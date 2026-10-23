"""Grid search for the painting detection of w3/utils.py (QSD2-W3).

Compares variations of the implemented edges-only pipeline:
colour space / channels of the gradient, the edge threshold (k * Otsu),
the closing and opening kernel sizes and the projection threshold of the
rectangle fit. Everything else is the code of utils.py.

Run from the repository root or from w3/:
    python w3/gridsearch_masking.py

Results are appended to w3/results/gridsearch_masking/grid_search.csv as the
search runs (so partial results survive an interruption), and the best
configurations are printed at the end.
"""

import csv
import itertools
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils as U


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "qsd2_w3"
IMAGE_DIR = DATA_DIR / "non_augmented"   # clean images; use DATA_DIR for the augmented ones
OUT_DIR = Path(__file__).resolve().parent / "results" / "gridsearch_masking"

# -----------------------------------------------------------------------------
# Grid (720 configurations). Values in utils.py are marked with "<-".
# -----------------------------------------------------------------------------
COLOR_OPTIONS = {
    "HSV S,V": ("HSV", [1, 2]),      # <-
    "HSV V": ("HSV", [2]),
    "LAB L,a,b": ("LAB", [0, 1, 2]),
    "LAB L": ("LAB", [0]),
    "GRAY": ("GRAY", [0]),
}
EDGE_K = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]   # <- 0.5 (1.0 = plain Otsu)
CLOSE_FRAC = [0.01, 0.02, 0.03, 0.04]     # <- 0.02
OPEN_FRAC = [0.0, 0.02, 0.04]             # <- 0.02 (0 = no opening)
PROJ_FRAC = [0.5, 0.7]                    # <- 0.7

COLOR_CONVERSIONS = {"HSV": cv2.COLOR_BGR2HSV, "LAB": cv2.COLOR_BGR2LAB, "GRAY": cv2.COLOR_BGR2GRAY}


# -----------------------------------------------------------------------------
# Pipeline with the searched parameters (same steps as utils.detect_paintings)
# -----------------------------------------------------------------------------
def gradient_magnitude(img_bgr: np.ndarray, color_space: str, channels: list[int]) -> np.ndarray:
    """utils.gradient_magnitude with a configurable colour space and channels."""
    space = cv2.cvtColor(img_bgr, COLOR_CONVERSIONS[color_space])
    if space.ndim == 2:
        space = space[..., None]
    mag = np.zeros(space.shape[:2], np.float32)
    for c in channels:
        channel = space[..., c].astype(np.float32)
        gx = cv2.filter2D(channel, cv2.CV_32F, U.PREWITT_X)
        gy = cv2.filter2D(channel, cv2.CV_32F, U.PREWITT_Y)
        g = np.sqrt(gx ** 2 + gy ** 2)
        mag = np.maximum(mag, g / (g.max() + 1e-6))
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def edge_segmentation(grad: np.ndarray, edge_k: float) -> np.ndarray:
    """utils.edge_segmentation with a configurable k."""
    t, _ = cv2.threshold(grad, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = (grad > edge_k * t).astype(np.uint8)
    edges[U.border_mask(*edges.shape, 2)] = 0
    return edges


def edges_to_regions(edges: np.ndarray, close_frac: float, open_frac: float) -> np.ndarray:
    """utils.edges_to_regions with configurable kernels (open_frac 0 = no opening)."""
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, U.square_kernel(close_frac, edges.shape))
    mask = U.fill_holes(mask)
    if open_frac > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, U.square_kernel(open_frac, mask.shape))
        mask = U.fill_holes(mask)
    return mask


def detect(edges: np.ndarray, close_frac: float, open_frac: float, proj_frac: float) -> list[np.ndarray]:
    """Rest of utils.detect_paintings, starting from the binary edges."""
    U.PROJ_FRAC = proj_frac  # read by utils.mask_to_rect
    regions = edges_to_regions(edges, close_frac, open_frac)
    return U.order_masks([U.mask_to_rect(c) for c in U.painting_components(regions)])


# -----------------------------------------------------------------------------
# Data and evaluation
# -----------------------------------------------------------------------------
def load_data() -> list[tuple[int, np.ndarray, np.ndarray, int]]:
    """(id, image, ground-truth mask, number of paintings) for every query."""
    data = []
    for path in sorted(IMAGE_DIR.glob("*.jpg")):
        img = cv2.imread(str(path))
        gt = cv2.imread(str(path.with_suffix(".png")), cv2.IMREAD_GRAYSCALE) > 0
        n_paintings = cv2.connectedComponents(gt.astype(np.uint8))[0] - 1
        data.append((int(path.stem), img, gt, n_paintings))
    assert data, f"No images found in {IMAGE_DIR}"
    return data


def evaluate(edges_by_image: list[np.ndarray], data, close_frac, open_frac, proj_frac) -> dict:
    """Mean mask metrics and painting-count accuracy over the dataset."""
    scores, count_ok, wrong = [], 0, []
    for edges, (idx, img, gt, n_paintings) in zip(edges_by_image, data):
        masks = detect(edges, close_frac, open_frac, proj_frac)
        union = np.zeros(gt.shape, bool)
        for m in masks:
            union |= m > 0
        scores.append(U.metrics(union, gt))
        if len(masks) == n_paintings:
            count_ok += 1
        else:
            wrong.append(idx)
    result = {k: float(np.mean([s[k] for s in scores])) for k in ("precision", "recall", "f1", "iou")}
    result["count_ok"] = count_ok
    result["wrong_count_ids"] = " ".join(map(str, wrong))
    return result


def main() -> None:
    data = load_data()
    print(f"{len(data)} images loaded from {IMAGE_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "grid_search.csv"
    fields = ["color", "edge_k", "close_frac", "open_frac", "proj_frac",
              "count_ok", "f1", "iou", "precision", "recall", "wrong_count_ids"]

    morph_grid = list(itertools.product(CLOSE_FRAC, OPEN_FRAC, PROJ_FRAC))
    total = len(COLOR_OPTIONS) * len(EDGE_K) * len(morph_grid)
    print(f"{total} configurations")

    results = []
    done = 0
    start = time.time()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for color_name, (color_space, channels) in COLOR_OPTIONS.items():
            # The gradient only depends on the colour option: compute it once.
            grads = [gradient_magnitude(img, color_space, channels) for _, img, _, _ in data]

            for edge_k in EDGE_K:
                edges_by_image = [edge_segmentation(g, edge_k) for g in grads]

                for close_frac, open_frac, proj_frac in morph_grid:
                    row = {"color": color_name, "edge_k": edge_k, "close_frac": close_frac,
                           "open_frac": open_frac, "proj_frac": proj_frac,
                           **evaluate(edges_by_image, data, close_frac, open_frac, proj_frac)}
                    writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
                    f.flush()
                    results.append(row)

                    done += 1
                    elapsed = time.time() - start
                    eta = elapsed / done * (total - done)
                    print(f"[{done}/{total}] {color_name:9s} k={edge_k:<4} close={close_frac:<5} "
                          f"open={open_frac:<5} proj={proj_frac:<4} -> paintings {row['count_ok']}/{len(data)} "
                          f"F1 {row['f1']:.4f} | elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min",
                          flush=True)

    # Best first: most paintings with the right count, then F1.
    results.sort(key=lambda r: (r["count_ok"], r["f1"]), reverse=True)
    print(f"\nResults saved to {csv_path}\n\nTop 10 configurations:")
    for r in results[:10]:
        print(f"  paintings {r['count_ok']}/{len(data)}  F1 {r['f1']:.4f}  IoU {r['iou']:.4f}  | "
              f"{r['color']}, k={r['edge_k']}, close={r['close_frac']}, open={r['open_frac']}, "
              f"proj={r['proj_frac']}  wrong: [{r['wrong_count_ids']}]")


if __name__ == "__main__":
    main()
