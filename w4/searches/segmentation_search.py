"""
Painting detection search for Week 4 (QSD1-W4, checked on QSD2-W3).

    python searches/segmentation_search.py          (from w4/)

Part 1 compares the segmentation methods of the previous weeks with the new one:

    W3 edges              Week 3 detect_paintings (edges + morphology)
    W3 edges + fallback   same, the whole image when nothing is found
    W2 wall model         Week 2 Mahalanobis distance to the border colour, several paintings
    W3 edges - wall       edge regions minus the wall-coloured pixels, opened and refilled
    W4 edges + split      segmentation.py: edge regions cut where there is only wall

Part 2 is a grid search of segmentation.py (like Week 3's gridsearch_masking.py).

Queries are denoised with denoising.denoise_image first, as in main.py. Metrics:
mean pixel precision / recall / F1 of the union of the masks against the
ground-truth mask, and the number of images with the right number of paintings.
Configurations are ranked by the right count on both datasets, then by F1.
Results go to searches/results/segmentation_methods.csv and segmentation_grid.csv.
"""

import csv
import itertools
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

W4 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W4))
import segmentation as S
from denoising import denoise_image

DATA = W4.parent / "data"
DATASETS = {"w4": DATA / "qsd1_w4", "w3": DATA / "qsd2_w3"}
OUT = W4 / "searches" / "results"

GRID = {  # values in segmentation.py: EDGE_K 0.5, CLOSE_FRAC 0.02, WALL_THR 4, SPLIT_THR 0.2, PROJ_FRAC 0.8
    "EDGE_K": [0.4, 0.5, 0.6],
    "CLOSE_FRAC": [0.01, 0.02, 0.03],
    "WALL_THR": [3.0, 4.0, 5.0],
    "SPLIT_THR": [0.1, 0.2, 0.3],
    "PROJ_FRAC": [0.7, 0.8],
}


# =============================================================================
# DATA
# =============================================================================

def load(folder: Path) -> list:
    """(id, denoised work-size image, ground-truth mask, number of paintings) of every query."""
    items = []
    for path in sorted(folder.glob("*.jpg")):
        img = S.resize_max_side(denoise_image(cv2.imread(str(path))))
        gt = cv2.imread(str(path.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        n = cv2.connectedComponents(gt.astype(np.uint8))[0] - 1
        items.append((int(path.stem), img, gt, n))
    return items


# =============================================================================
# METHODS (all on the work-size image)
# =============================================================================

def finish(regions: np.ndarray, fallback: bool) -> list:
    """Same selection, rectangle and order as segmentation.py, with each connected region as a part (no split)."""
    n, labels = cv2.connectedComponents(regions, connectivity=8)
    paintings = S.select_paintings([(labels == l).astype(np.uint8) for l in range(1, n)])
    rects = S.order_masks([S.mask_to_rect(p) for p in paintings])
    return rects if rects or not fallback else [np.ones(regions.shape, np.uint8)]


def w3_edges(img, fallback=False):
    return finish(S.edges_to_regions(S.edge_segmentation(S.gradient_magnitude(img))), fallback)


def not_wall_lab(img) -> np.ndarray:
    """Week 2 foreground: Lab L,a,b distance to the border colour > WALL_THR (L,a,b is best for these methods)."""
    old, S.WALL_CHANNELS = S.WALL_CHANNELS, [0, 1, 2]
    fg = (S.wall_distance(img) > S.WALL_THR).astype(np.uint8)
    S.WALL_CHANNELS = old
    fg[S.border_mask(*fg.shape, 2)] = 0
    return fg


def w2_wall(img):
    """Week 2 colour model, extended to several paintings (closing 1% was its best)."""
    old, S.CLOSE_FRAC = S.CLOSE_FRAC, 0.01
    regions = S.edges_to_regions(not_wall_lab(img))
    S.CLOSE_FRAC = old
    return finish(regions, True)


def w3_edges_minus_wall(img):
    """Week 3 regions without their wall-coloured pixels, opened to cut thin bridges and refilled."""
    regions = S.edges_to_regions(S.edge_segmentation(S.gradient_magnitude(img))) * not_wall_lab(img)
    regions = S.fill_holes(cv2.morphologyEx(regions, cv2.MORPH_OPEN, S.square_kernel(S.OPEN_FRAC, regions.shape)))
    return finish(regions, True)


def w4_split(img):
    return S.segment_steps(img)["rects"]


METHODS = {
    "W3 edges": lambda img: w3_edges(img),
    "W3 edges + fallback": lambda img: w3_edges(img, fallback=True),
    "W2 wall model": w2_wall,
    "W3 edges - wall": w3_edges_minus_wall,
    "W4 edges + split": w4_split,
}


# =============================================================================
# EVALUATION
# =============================================================================

def scores(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = np.count_nonzero(pred & gt)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(~pred & gt)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def evaluate(method, data: dict) -> dict:
    row = {}
    for name, items in data.items():
        values, ok, wrong = [], 0, []
        for idx, img, gt, n in items:
            masks = method(img)
            union = np.zeros(gt.shape, bool)
            for m in masks:
                union |= m > 0
            values.append(scores(union, gt))
            ok += len(masks) == n
            if len(masks) != n:
                wrong.append(idx)
        p, r, f1 = np.mean(values, axis=0)
        row.update({f"{name}_count": ok, f"{name}_f1": round(f1, 4), f"{name}_precision": round(p, 4),
                    f"{name}_recall": round(r, 4), f"{name}_wrong": " ".join(map(str, wrong))})
    return row


def rank(row: dict):
    return row["w4_count"] + row["w3_count"], row["w4_f1"] + row["w3_f1"]


_DATA = None


def _init(data):
    global _DATA
    _DATA = data
    cv2.setNumThreads(1)


def _evaluate_config(cfg: dict) -> dict:
    for k, v in cfg.items():
        setattr(S, k, v)
    return {**cfg, **evaluate(w4_split, _DATA)}


def save(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def show(row: dict, label: str) -> None:
    print(f"{label:22s} QSD1-W4 count {row['w4_count']}/30  F1 {row['w4_f1']:.4f} "
          f"(P {row['w4_precision']:.3f}, R {row['w4_recall']:.3f}) wrong [{row['w4_wrong']}] | "
          f"QSD2-W3 count {row['w3_count']}/30  F1 {row['w3_f1']:.4f} wrong [{row['w3_wrong']}]")


def main() -> None:
    t0 = time.time()
    data = {name: load(folder) for name, folder in DATASETS.items()}
    print(f"Data loaded in {time.time() - t0:.0f}s")

    print("\nPart 1: methods")
    rows = []
    for name, method in METHODS.items():
        row = {"method": name, **evaluate(method, data)}
        rows.append(row)
        show(row, name)
    save(rows, OUT / "segmentation_methods.csv")

    print("\nPart 2: grid search of segmentation.py")
    grid = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    print(f"{len(grid)} configurations")
    with Pool(None, _init, (data,)) as pool:
        rows = pool.map(_evaluate_config, grid)
    rows.sort(key=rank, reverse=True)
    save(rows, OUT / "segmentation_grid.csv")
    for row in rows[:10]:
        show(row, " ".join(f"{k.split('_')[0].lower()}={row[k]}" for k in GRID))
    print(f"\nResults saved under {OUT} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
