"""
Week 4 denoising: impulse detector + luma/chroma two-stage filter.

The query images carry random-valued impulse noise that has been JPEG
compressed afterwards. After compression the noise has three parts, and each
stage of the filter targets one of them:

    Y (brightness)  1. median 3x3   -> removes the impulse cores
                    2. NL-means     -> removes the grainy JPEG halo around them
    Cr, Cb (colour) 3. median 5x5   -> removes colour impulses
                    4. Gaussian     -> flattens the low-frequency colour blotches

The filter is only applied to images flagged by the Week 3 impulse detector, so
clean and hue-shifted images are returned unchanged.

Usage in main.py (after `from utils import *`, so it replaces the Week 3 one):
    from denoising import denoise_image

Running this file denoises a query folder and, if it has a non_augmented/
subfolder, reports PSNR against the clean references:
    python denoising.py [query_folder]

See denoising_best.ipynb for the full explanation and searches/ for the
experiments behind the parameters.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Impulse detector (same as Week 3)
DETECT_DIFF = 30          # a pixel is isolated if it differs > 30 from all 8 neighbours
DETECT_FRACTION = 0.01    # image is noisy if > 1% of its pixels are isolated

# Luma (Y) branch
LUMA_MEDIAN = 3           # median kernel
LUMA_NLM_H = 4            # NL-means strength
NLM_TEMPLATE = 7          # NL-means patch size
NLM_SEARCH = 11           # NL-means search window

# Chroma (Cr, Cb) branch
CHROMA_MEDIAN = 5         # median kernel
CHROMA_GAUSS = 21         # Gaussian kernel size (sigma derived by OpenCV)


# =============================================================================
# DETECTION
# =============================================================================

def impulse_fraction(image_bgr: np.ndarray) -> float:
    """
    Compute the fraction of isolated pixels of an image.

    A pixel is isolated when its grayscale value differs by more than
    DETECT_DIFF from all eight neighbouring pixels.

    Args:
        image_bgr: Input image in BGR format.

    Returns:
        Fraction of isolated pixels, between 0 and 1.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    h, w = gray.shape
    padded = np.pad(gray, 1, mode="edge")
    isolated = np.ones((h, w), dtype=bool)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            isolated &= np.abs(gray - neighbor) > DETECT_DIFF

    return float(isolated.mean())


def has_impulse_noise(image_bgr: np.ndarray) -> bool:
    """Return True if more than DETECT_FRACTION of the pixels are isolated."""
    return impulse_fraction(image_bgr) > DETECT_FRACTION


# =============================================================================
# FILTERING
# =============================================================================

def denoise_luma(y: np.ndarray) -> np.ndarray:
    """
    Denoise the brightness channel: median for the impulses, NL-means for the halo.

    Args:
        y: Y channel (uint8) of a YCrCb image.

    Returns:
        Denoised Y channel.
    """
    y = cv2.medianBlur(y, LUMA_MEDIAN)
    return cv2.fastNlMeansDenoising(y, None, LUMA_NLM_H, NLM_TEMPLATE, NLM_SEARCH)


def denoise_chroma(channel: np.ndarray) -> np.ndarray:
    """
    Denoise a colour channel: median for the impulses, Gaussian for the blotches.

    Strong smoothing is safe here because JPEG already stores colour at half
    resolution, so the paintings have almost no fine detail in Cr/Cb.

    Args:
        channel: Cr or Cb channel (uint8) of a YCrCb image.

    Returns:
        Denoised channel.
    """
    channel = cv2.medianBlur(channel, CHROMA_MEDIAN)
    return cv2.GaussianBlur(channel, (CHROMA_GAUSS, CHROMA_GAUSS), 0)


def remove_impulse_noise(image_bgr: np.ndarray) -> np.ndarray:
    """
    Apply the luma/chroma two-stage filter, without checking for noise first.

    Args:
        image_bgr: Input image in BGR format.

    Returns:
        Filtered BGR image.
    """
    y, cr, cb = cv2.split(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb))
    ycrcb = cv2.merge([denoise_luma(y), denoise_chroma(cr), denoise_chroma(cb)])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def denoise_image(image_bgr: np.ndarray) -> np.ndarray:
    """
    Remove impulse noise only when the image is detected as noisy.

    Args:
        image_bgr: Input image in BGR format.

    Returns:
        Filtered BGR image, or an unchanged copy if no impulse noise is detected.
    """
    if not has_impulse_noise(image_bgr):
        return image_bgr.copy()
    return remove_impulse_noise(image_bgr)


# =============================================================================
# DEMO
# =============================================================================

def psnr(img: np.ndarray, ref: np.ndarray) -> float:
    mse = np.mean((img.astype(np.float64) - ref.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "qsd1_w4"
    clean_dir = folder / "non_augmented"
    read = lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    gains = []
    for path in sorted(folder.glob("*.jpg")):
        img = read(path)
        if not has_impulse_noise(img):
            continue
        out = denoise_image(img)
        line = f"{path.stem}: impulse noise detected ({impulse_fraction(img):.2%} isolated pixels)"
        if (clean_dir / path.name).exists():
            clean = read(clean_dir / path.name)
            before, after = psnr(img, clean), psnr(out, clean)
            gains.append(after - before)
            line += f"  PSNR {before:.2f} -> {after:.2f} dB"
        print(line)

    if gains:
        print(f"Mean PSNR gain on {len(gains)} noisy images: {np.mean(gains):+.2f} dB")
