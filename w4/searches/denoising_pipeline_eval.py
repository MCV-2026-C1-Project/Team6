"""
Effect of the Week 4 denoiser on the Week 3 retrieval pipeline.

The Week 3 pipeline is run unchanged (w3/main.py + w3/utils.py: painting
detection, DCT-pyramid descriptor, Canberra distance) and only the denoising
step is swapped:

    none    - no denoising
    week3   - w3 utils.denoise_image (impulse detector + median 3x3 on BGR)
    week4   - impulse detector + luma/chroma two-stage filter
              (w4/denoising.py, explained in w4/denoising_best.ipynb)
    oracle  - impulse-noise queries replaced by their clean version
              (non_augmented): the ceiling any denoiser could reach

Datasets:
    QSD2-W3 - full Week 3 pipeline: denoise -> detect up to 2 paintings ->
              describe -> retrieve. mAP is computed per painting and a missed
              painting counts as AP = 0. Mask F1 is also reported, because the
              denoiser runs before painting detection.
    QSD1-W3 - one cropped painting per query: denoise -> describe -> retrieve.

Neither dataset was used to tune the Week 4 denoiser (it was tuned on QSD1-W4).

Output: results/denoising_pipeline_eval.csv (AP per painting and method) and
a summary printed to the console / saved to results/denoising_pipeline_summary.csv.
"""

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
W3_DIR = ROOT_DIR / "w3"
sys.path.insert(0, str(W3_DIR))
sys.path.append(str(ROOT_DIR))

import main as w3_main  # noqa: E402  (Week 3 pipeline; its denoise_image is swapped below)
import utils as w3_utils  # noqa: E402
from w4.denoising import denoise_image as denoise_week4, has_impulse_noise  # noqa: E402

DATA_DIR = ROOT_DIR / "data"
DB_DIR = DATA_DIR / "BBDD"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PARAMS = w3_main.DESC_PARAMS
KS = (1, 5, 10)


def identity(image_bgr: np.ndarray) -> np.ndarray:
    return image_bgr.copy()


DENOISERS = {
    "none": identity,
    "week3": w3_utils.denoise_image,
    "week4": denoise_week4,
}


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load_query_set(name: str):
    """Load a query set, its ground truth, augmentation labels and an oracle copy."""
    folder = DATA_DIR / name
    ids, images = w3_utils.load_images(folder)
    with open(folder / "gt_corresps.pkl", "rb") as f:
        gt = pickle.load(f)
    with open(folder / "augmentations.pkl", "rb") as f:
        augmentations = pickle.load(f)

    impulse = [("ImpulseNoise" in str(augmentations[i])) for i in ids]
    oracle = []
    for qid, img, is_impulse in zip(ids, images, impulse):
        if is_impulse:
            clean = cv2.imread(str(folder / "non_augmented" / f"{qid:05d}.jpg"))
            if clean.shape != img.shape:
                raise ValueError(f"{name} {qid}: clean {clean.shape} vs noisy {img.shape}")
            img = clean
        oracle.append(img)
    return ids, images, [gt[i] for i in ids], impulse, oracle


# -----------------------------------------------------------------------------
# Pipelines
# -----------------------------------------------------------------------------
def run_qsd2(images, db_ids, db_descs, denoise):
    """Week 3 main pipeline with the given denoiser. Returns predictions and masks."""
    w3_main.denoise_image = denoise
    descs, masks = w3_main.process_queries(images, db_descs.shape[1], PARAMS)
    dists = w3_utils.get_distances(descs, db_descs, metric=PARAMS["metric"])
    return w3_utils.retrieve_multi(dists, db_ids, k=max(KS)), masks


def run_qsd1(images, db_ids, db_descs, denoise):
    """QSD1-W3 pipeline (no painting detection): one ranking per query, wrapped as one painting."""
    descs = np.vstack([w3_utils.describe(denoise(img), PARAMS) for img in images])
    dists = w3_utils.get_distances(descs, db_descs, metric=PARAMS["metric"])
    return [[ranking] for ranking in w3_utils.retrieve(dists, db_ids, k=max(KS))]


def ap_rows(dataset, method, ids, gt, impulse, predictions):
    """One row per ground-truth painting with its AP@k (missed paintings get AP = 0)."""
    rows = []
    for qid, gt_q, is_impulse, pred_q in zip(ids, gt, impulse, predictions):
        for j, gt_painting in enumerate(gt_q):
            pred = pred_q[j] if j < len(pred_q) else []
            row = {"dataset": dataset, "method": method, "query": qid, "painting": j,
                   "impulse": is_impulse, "detected": j < len(pred_q)}
            row.update({f"ap@{k}": w3_utils.apk([gt_painting], pred, k) for k in KS})
            rows.append(row)
    return rows


def mask_f1(ids, images, masks):
    scores = {}
    for qid, img, m in zip(ids, images, masks):
        gt_mask = cv2.imread(str(DATA_DIR / "qsd2_w3" / f"{qid:05d}.png"), cv2.IMREAD_GRAYSCALE)
        scores[qid] = w3_utils.metrics(w3_utils.combined_mask(m, img.shape) > 0, gt_mask > 0)["f1"]
    return scores


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing BBDD descriptors...")
    db_ids, db_images = w3_utils.load_images(DB_DIR)
    db_descs = np.vstack([w3_utils.describe(img, PARAMS) for img in db_images])

    rows, f1_rows = [], []
    for dataset in ("qsd2_w3", "qsd1_w3"):
        ids, images, gt, impulse, oracle = load_query_set(dataset)
        flagged = [qid for qid, img in zip(ids, images) if has_impulse_noise(img)]
        print(f"\n{dataset}: impulse-noise queries {[q for q, i in zip(ids, impulse) if i]}, "
              f"flagged by the detector {flagged}")

        runs = {name: (images, fn) for name, fn in DENOISERS.items()}
        runs["oracle"] = (oracle, identity)
        for method, (query_images, denoise) in runs.items():
            print(f"  running {method}...")
            if dataset == "qsd2_w3":
                predictions, masks = run_qsd2(query_images, db_ids, db_descs, denoise)
                f1 = mask_f1(ids, query_images, masks)
                f1_rows += [{"method": method, "query": q, "impulse": imp, "f1": f1[q],
                             "n_detected": len(m)} for q, imp, m in zip(ids, impulse, masks)]
            else:
                predictions = run_qsd1(query_images, db_ids, db_descs, denoise)
            rows += ap_rows(dataset, method, ids, gt, impulse, predictions)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "denoising_pipeline_eval.csv", index=False)
    ap_cols = [f"ap@{k}" for k in KS]
    order = ["none", "week3", "week4", "oracle"]

    summary = (df.groupby(["dataset", "method"])[ap_cols].mean()
                 .rename(columns=lambda c: c.replace("ap", "mAP")).reindex(order, level=1))
    noisy_only = (df[df["impulse"]].groupby(["dataset", "method"])[ap_cols].mean()
                    .rename(columns=lambda c: c.replace("ap", "mAP")).reindex(order, level=1))
    summary.to_csv(RESULTS_DIR / "denoising_pipeline_summary.csv")

    f1 = pd.DataFrame(f1_rows)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n=== mAP over all paintings ===")
        print(summary.round(4))
        print("\n=== mAP over the paintings of impulse-noise queries only ===")
        print(noisy_only.round(4))
        print("\n=== QSD2-W3 painting detection (mask F1) ===")
        print(f1.groupby(["method", "impulse"])["f1"].mean().unstack().reindex(order).round(4))

        print("\n=== Per painting, impulse-noise queries (AP@5) ===")
        noisy = df[df["impulse"]].pivot_table(index=["dataset", "query", "painting"], columns="method",
                                              values="ap@5").reindex(columns=order)
        print(noisy.round(3))

        clean_changes = df[~df["impulse"]].pivot_table(index=["dataset", "query", "painting"],
                                                       columns="method", values="ap@5")
        changed = clean_changes[clean_changes.nunique(axis=1) > 1]
        print(f"\nPaintings of non-noisy queries whose AP@5 changed between methods: {len(changed)}")
        if len(changed):
            print(changed.round(3))


if __name__ == "__main__":
    main()
