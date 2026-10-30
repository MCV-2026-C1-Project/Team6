"""
Week 4 denoising search.

Compares classical denoisers on the impulse-noise images of QSD1-W4 against
their clean references (data/qsd1_w4/non_augmented). Only the impulse images
are scored: the detector in utils.denoise_image leaves every other image
untouched, so the filter choice only matters for these.

Families tried:
    single    - filters already used in previous weeks (mean, Gaussian, median,
                bilateral, NL-means)
    impulse   - impulse-specific nonlinear filters (iterated, adaptive and
                switching medians)
    two-stage - median 3x3 to remove impulses + an edge-preserving smoother for
                the JPEG residual (Gaussian, bilateral, guided, wavelet, NL-means)
    luma/chroma - two-stage in YCrCb with a different treatment for the
                luminance and the chroma channels

Output: results/denoising_search.csv (one row per image and method) and a
summary printed to the console / saved to results/denoising_search_summary.csv.
"""

import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
from skimage.restoration import denoise_wavelet

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
NOISY_DIR = DATA_DIR / "qsd1_w4"
CLEAN_DIR = NOISY_DIR / "non_augmented"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINE = "median 3x3"


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def read_bgr(path: Path) -> np.ndarray:
    """Read a BGR image ignoring EXIF rotation, so noisy/clean pairs stay aligned."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_impulse_pairs() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    with open(NOISY_DIR / "augmentations.pkl", "rb") as f:
        augmentations = pickle.load(f)
    ids = [i for i, aug in enumerate(augmentations) if "ImpulseNoise" in aug]
    return {i: (read_bgr(NOISY_DIR / f"{i:05d}.jpg"), read_bgr(CLEAN_DIR / f"{i:05d}.jpg")) for i in ids}


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def psnr(img: np.ndarray, ref: np.ndarray) -> float:
    mse = np.mean((img.astype(np.float64) - ref.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


def ssim(img: np.ndarray, ref: np.ndarray) -> float:
    return structural_similarity(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY),
                                 cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), data_range=255)


# -----------------------------------------------------------------------------
# Impulse-specific filters
# -----------------------------------------------------------------------------
def adaptive_median(img: np.ndarray, max_k: int = 7) -> np.ndarray:
    """
    Adaptive median filter (Gonzalez & Woods), per channel.

    The window grows (3, 5, 7) until its median is not an extreme value; then the
    center pixel is replaced by the median only if the center itself is an extreme.
    """
    out = img.copy()
    done = np.zeros(img.shape, dtype=bool)
    med = img
    for k in range(3, max_k + 1, 2):
        kernel = np.ones((k, k), np.uint8)
        med, zmin, zmax = cv2.medianBlur(img, k), cv2.erode(img, kernel), cv2.dilate(img, kernel)
        level_ok = (zmin < med) & (med < zmax) & ~done
        center_is_impulse = (img <= zmin) | (img >= zmax)
        replace = level_ok & center_is_impulse
        out[replace] = med[replace]
        done |= level_ok
    out[~done] = med[~done]
    return out


def road(gray: np.ndarray) -> np.ndarray:
    """Rank-Ordered Absolute Differences: sum of the 4 smallest |center - neighbor|."""
    g = gray.astype(np.int16)
    h, w = g.shape
    p = np.pad(g, 1, mode="edge")
    diffs = np.stack([np.abs(g - p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w])
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
    diffs.sort(axis=0)
    return diffs[:4].sum(axis=0)


def road_switching_median(img: np.ndarray, threshold: int = 100) -> np.ndarray:
    """Replace only the pixels whose ROAD score marks them as impulses by the 3x3 median."""
    med = cv2.medianBlur(img, 3)
    mask = road(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) > threshold
    out = img.copy()
    out[mask] = med[mask]
    return out


# -----------------------------------------------------------------------------
# Two-stage and luma/chroma filters
# -----------------------------------------------------------------------------
def median3(img):
    return cv2.medianBlur(img, 3)


def to_uint8(img_float: np.ndarray) -> np.ndarray:
    return (np.clip(img_float, 0, 1) * 255).round().astype(np.uint8)


def luma_chroma(img, luma_fn, chroma_fn):
    """Filter Y with luma_fn and Cr/Cb with chroma_fn."""
    y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
    return cv2.cvtColor(cv2.merge([luma_fn(y), chroma_fn(cr), chroma_fn(cb)]), cv2.COLOR_YCrCb2BGR)


def luma_nlm(h, search=11):
    return lambda y: cv2.fastNlMeansDenoising(cv2.medianBlur(y, 3), None, h, 7, search)


def chroma_median_gauss(k, sigma_k):
    return lambda c: cv2.GaussianBlur(cv2.medianBlur(c, k), (sigma_k, sigma_k), 0) if sigma_k else cv2.medianBlur(c, k)


METHODS = {
    # --- single filters (previous weeks) ---
    "none": ("baseline", lambda x: x),
    "mean 3x3": ("single", lambda x: cv2.blur(x, (3, 3))),
    "mean 5x5": ("single", lambda x: cv2.blur(x, (5, 5))),
    "gaussian 5x5": ("single", lambda x: cv2.GaussianBlur(x, (5, 5), 0)),
    "median 3x3": ("single", median3),
    "median 5x5": ("single", lambda x: cv2.medianBlur(x, 5)),
    "bilateral d9": ("single", lambda x: cv2.bilateralFilter(x, 9, 75, 75)),
    "nl-means h10": ("single", lambda x: cv2.fastNlMeansDenoisingColored(x, None, 10, 10, 7, 21)),
    # --- impulse-specific ---
    "median 3x3 twice": ("impulse", lambda x: median3(median3(x))),
    "adaptive median": ("impulse", adaptive_median),
    "ROAD switching median": ("impulse", road_switching_median),
    # --- two-stage: median 3x3 + smoother ---
    "median3 + gaussian3": ("two-stage", lambda x: cv2.GaussianBlur(median3(x), (3, 3), 0)),
    "median3 + bilateral d5": ("two-stage", lambda x: cv2.bilateralFilter(median3(x), 5, 30, 5)),
    "median3 + bilateral d9": ("two-stage", lambda x: cv2.bilateralFilter(median3(x), 9, 30, 9)),
    "median3 + guided r2": ("two-stage", lambda x: cv2.ximgproc.guidedFilter(median3(x), median3(x), 2, 100)),
    "median3 + wavelet": ("two-stage", lambda x: to_uint8(denoise_wavelet(
        median3(x), method="BayesShrink", mode="soft", channel_axis=-1, convert2ycbcr=True, rescale_sigma=True))),
    "median3 + nl-means h5": ("two-stage", lambda x: cv2.fastNlMeansDenoisingColored(median3(x), None, 5, 5, 7, 11)),
    "median3 + nl-means h5/10": ("two-stage", lambda x: cv2.fastNlMeansDenoisingColored(median3(x), None, 5, 10, 7, 11)),
    # --- luma/chroma: Y = median3 + smoother, CrCb = median + Gaussian ---
    "Y bilateral | C med5+g9": ("luma/chroma", lambda x: luma_chroma(
        x, lambda y: cv2.bilateralFilter(cv2.medianBlur(y, 3), 5, 30, 5), chroma_median_gauss(5, 9))),
    "Y nlm h4 | C med5+g9": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(4), chroma_median_gauss(5, 9))),
    "Y nlm h5 | C med5+g9": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(5), chroma_median_gauss(5, 9))),
    "Y nlm h6 | C med5+g9": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(6), chroma_median_gauss(5, 9))),
    "Y nlm h7 | C med5+g9": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(7), chroma_median_gauss(5, 9))),
    "Y nlm h5 s21 | C med5+g9": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(5, 21), chroma_median_gauss(5, 9))),
    "Y nlm h5 | C med3+g9": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(5), chroma_median_gauss(3, 9))),
    "Y nlm h5 | C med5": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(5), chroma_median_gauss(5, 0))),
    "Y nlm h5 | C med5+g15": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(5), chroma_median_gauss(5, 15))),
    "Y nlm h5 | C med7+g9": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(5), chroma_median_gauss(7, 9))),
    # --- refinement: lighter luma NL-means x stronger chroma smoothing ---
    "Y nlm h3 | C med5+g15": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(3), chroma_median_gauss(5, 15))),
    "Y nlm h4 | C med5+g15": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(4), chroma_median_gauss(5, 15))),
    "Y nlm h4 | C med5+g21": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(4), chroma_median_gauss(5, 21))),
    "Y nlm h4 | C med5+g31": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(4), chroma_median_gauss(5, 31))),
    "Y nlm h4 | C med7+g15": ("luma/chroma", lambda x: luma_chroma(x, luma_nlm(4), chroma_median_gauss(7, 15))),
    "Y guided | C med5+g15": ("luma/chroma", lambda x: luma_chroma(
        x, lambda y: cv2.ximgproc.guidedFilter(cv2.medianBlur(y, 3), cv2.medianBlur(y, 3), 2, 100),
        chroma_median_gauss(5, 15))),
}


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_impulse_pairs()
    print(f"Impulse images: {list(pairs)}")

    rows = []
    for i, (noisy, clean) in pairs.items():
        megapixels = noisy.shape[0] * noisy.shape[1] / 1e6
        print(f"\nImage {i:05d} ({megapixels:.1f} MP)")
        for name, (family, fn) in METHODS.items():
            start = time.perf_counter()
            out = fn(noisy)
            seconds = time.perf_counter() - start
            rows.append({"id": i, "method": name, "family": family, "psnr": psnr(out, clean),
                         "ssim": ssim(out, clean), "seconds": seconds, "megapixels": megapixels})
            print(f"  {name:28s} PSNR {rows[-1]['psnr']:6.2f}  SSIM {rows[-1]['ssim']:.3f}  {seconds:6.2f}s")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "denoising_search.csv", index=False)

    base = df[df["method"] == BASELINE].set_index("id")
    df["psnr_win"] = df["psnr"].values > base.loc[df["id"], "psnr"].values
    df["ssim_win"] = df["ssim"].values > base.loc[df["id"], "ssim"].values
    summary = (df.groupby(["method", "family"], sort=False)
                 .agg(psnr=("psnr", "mean"), ssim=("ssim", "mean"),
                      psnr_wins=("psnr_win", "sum"), ssim_wins=("ssim_win", "sum"),
                      seconds=("seconds", "sum"), megapixels=("megapixels", "sum"))
                 .reset_index())
    summary["s_per_mp"] = summary["seconds"] / summary["megapixels"]
    summary = summary.drop(columns=["seconds", "megapixels"]).sort_values("psnr", ascending=False)
    summary.to_csv(RESULTS_DIR / "denoising_search_summary.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(f"\nWins are counted against '{BASELINE}' (the Week 3 filter) over {len(pairs)} images.")
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
