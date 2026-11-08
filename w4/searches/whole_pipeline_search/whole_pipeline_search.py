"""
Grid search on QSD1-W4: descriptor x detector x CLAHE x max keypoints.

    python grid_search.py

The inliers of every configuration are cached in RESULTS_DIR/inliers.pkl, so an
interrupted search resumes where it stopped. A painting is predicted unknown (-1)
when its best match has fewer inliers than min_inliers, which is chosen per
configuration on the same queries (optimistic).
"""
import pickle
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

import main
from utils import UNKNOWN, apk, load_images, load_pickle


RESULTS_DIR = Path(__file__).resolve().parent / "searches" / "grid_search" / "results"
CACHE_PATH = RESULTS_DIR / "inliers.pkl"

DESCRIPTORS = ["rootsift", "akaze", "orb"]
DETECTORS = [None, "dog", "harris_laplace", "hessian_laplace", "fast"]
COLOR_OPTIONS = ["gray", "gray_saturation", "opponent"]
CLAHE = [True, False]
MAX_KEYPOINTS = [500, 1000]

N_PLOTTED = 5
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def configurations() -> list[dict]:
    """Custom detectors only exist for rootsift."""
    return [dict(descriptor=descriptor, detector=detector, color=color, clahe=clahe, max_keypoints=max_keypoints)
            for descriptor, color, clahe, max_keypoints in product(DESCRIPTORS, COLOR_OPTIONS, CLAHE, MAX_KEYPOINTS)
            for detector in (DETECTORS if descriptor == "rootsift" else [None])]


def name(config: dict) -> str:
    """
    Build a readable label for a configuration.

    Args:
        config: Configuration with descriptor, detector, color, clahe and max_keypoints.

    Returns:
        Label such as "rootsift+dog color=gray clahe=True kps=500".
    """
    detector = f"+{config['detector']}" if config["detector"] else ""
    return (f"{config['descriptor']}{detector} color={config['color']} "
            f"clahe={config['clahe']} kps={config['max_keypoints']}")


def search(query_images, db_images, db_ids, cache_path=CACHE_PATH) -> dict:
    """{name: inlier matrix (n_paintings, n_database)} of every configuration, cached on disk."""
    cache = load_pickle(cache_path) if cache_path.exists() else {}
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    progress = tqdm(configurations(), desc="Grid search")
    for config in progress:
        progress.set_postfix_str(name(config))
        if name(config) in cache:
            continue
        results = main.run_pipeline(query_images, db_images, db_ids, **{**main.PARAMS, **config},
                                    predictions=False)
        cache[name(config)] = np.vstack(results["inliers"])
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    return {name(config): cache[name(config)] for config in configurations()}


def evaluate(inliers: np.ndarray, labels: np.ndarray, db_ids: np.ndarray) -> dict:
    """mAP@5 with unknowns for every threshold, the best one, and the ROC known vs unknown."""
    known = labels != UNKNOWN
    best_match = inliers.max(axis=1)
    rankings = db_ids[np.argsort(-inliers, axis=1, kind="stable")[:, :5]]
    ap5 = np.array([apk([label], list(ranking), 5) for label, ranking in zip(labels, rankings)])

    thresholds = np.arange(best_match.max() + 2)
    map5 = np.array([np.where(known, ap5 * (best_match >= t), best_match < t).mean() for t in thresholds])

    best = np.flatnonzero(map5 == map5.max())
    longest_run = max(np.split(best, np.flatnonzero(np.diff(best) > 1) + 1), key=len)
    threshold = thresholds[longest_run[len(longest_run) // 2]]
    fpr, tpr, _ = roc_curve(known, best_match)

    return {"thresholds": thresholds, "map5": map5, "fpr": fpr, "tpr": tpr,
            "best_map5": map5.max(), "threshold": threshold,
            "auc": auc(fpr, tpr)}


def plot(curves: list, title: str, xlabel: str, ylabel: str, path: Path, xlim=None):
    """One line per (label, x, y) in curves."""
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for color, (label, x, y) in zip(COLORS, curves):
        ax.plot(x, y, color=color, linewidth=2, label=label)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=(0, 1.02))
    ax.grid(color="#e4e3df")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    db_ids, db_images = load_images(main.BBDD_DIR)
    query_ids, query_images = load_images(main.QUERY_DIR)
    ground_truth = load_pickle(main.GROUND_TRUTH_PATH)
    labels = np.array([painting for query_id in query_ids for painting in ground_truth[query_id]])
    db_ids = np.array(db_ids)

    inliers_by_config = search(query_images, db_images, db_ids)
    evaluations = {config_name: evaluate(inliers, labels, db_ids)
                   for config_name, inliers in inliers_by_config.items()}

    table = pd.DataFrame([{"name": name(config), **config, "detector": config["detector"] or "default",
                           "map5_unknowns": round(evaluations[name(config)]["best_map5"], 4),
                           "threshold": int(evaluations[name(config)]["threshold"]),
                           "roc_auc": round(evaluations[name(config)]["auc"], 4)}
                          for config in configurations()])
    table = table.sort_values(["map5_unknowns", "roc_auc"], ascending=False)
    table.drop(columns="name").to_csv(RESULTS_DIR / "grid_search.csv", index=False)

    best = table["name"].head(N_PLOTTED)
    plot([(f"{n} (AUC {evaluations[n]['auc']:.3f})", evaluations[n]["fpr"], evaluations[n]["tpr"]) for n in best],
         "ROC: known vs unknown paintings (score = best inliers)",
         "False positive rate (unknowns accepted)", "True positive rate (knowns accepted)",
         RESULTS_DIR / "roc.png", xlim=(0, 1))
    plot([(f"{n} (best {evaluations[n]['best_map5']:.3f} @ {evaluations[n]['threshold']})",
           evaluations[n]["thresholds"], evaluations[n]["map5"]) for n in best],
         "mAP@5 with unknowns vs min_inliers", "min_inliers", "mAP@5 (unknowns included)",
         RESULTS_DIR / "map_vs_threshold.png", xlim=(0, 100))

    print(table.drop(columns="name").head(10).to_string(index=False))
    print(f"\nResults saved to {RESULTS_DIR}")
