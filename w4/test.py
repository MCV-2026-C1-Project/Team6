"""Run the selected Week 4 local-feature method on the blind QST1-W4 set.

Run from the repository root after QST1-W4 has been released:
    python w4/test.py

Expected input:
    data/BBDD/*.jpg
    data/qst1_w4/*.jpg
    w4/results/task2_task3_results.csv

Output:
    w4/results/QST1/method1/result.pkl

Each query result is a list of paintings; each painting contains its ranked
museum IDs, or [-1] when the painting is rejected as unknown.
"""
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np

from grid_search import DATA, describe, match_score, museum_id
from cv_descriptors import NORMS
from w3 import utils as w3u

RESULTS_CSV = DATA.parent / "w4" / "results" / "task2_task3_results.csv"
QST1_DIR = DATA / "qst1_w4"
OUTPUT = DATA.parent / "w4" / "results" / "QST1" / "method1" / "result.pkl"
K = 10


def read_best_method():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_CSV}. Run python w4/grid_search.py first."
        )

    with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
        rows = [
            row for row in csv.DictReader(f)
            if row.get("method") == "week4_local"
        ]

    if not rows:
        raise ValueError(f"No Week 4 local-method rows in {RESULTS_CSV}")

    best = max(
        rows,
        key=lambda row: (
            float(row["map@5"]),
            float(row["unknown_f1"]),
            float(row["map@1"]),
        ),
    )
    best["clahe"] = best["clahe"].strip().lower() == "true"
    best["unknown_threshold"] = float(best["unknown_threshold"])
    if best["keypoint_detector"] == "descriptor_default":
        best["keypoint_detector"] = None
    return best


def main():
    config = read_best_method()
    descriptor = config["descriptor"]
    color = config["color"]
    clahe = config["clahe"]
    keypoint_detector = config["keypoint_detector"]
    threshold = config["unknown_threshold"]
    norm = NORMS[descriptor]

    museum_paths = sorted((DATA / "BBDD").glob("*.jpg"))
    if not museum_paths:
        raise FileNotFoundError(f"No museum images found in {DATA / 'BBDD'}")
    museum_ids = [museum_id(path) for path in museum_paths]

    museum_features = []
    for path in museum_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read museum image: {path}")
        museum_features.append(
            describe(image, descriptor, color, clahe, keypoint_detector)
        )

    query_paths = sorted(QST1_DIR.glob("*.jpg"))
    if not query_paths:
        raise FileNotFoundError(
            f"No QST1-W4 JPGs found in {QST1_DIR}. "
            "Copy the released test images there first."
        )

    all_predictions = []
    for image_number, path in enumerate(query_paths, start=1):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read query image: {path}")

        image = w3u.denoise_image(image)
        masks = w3u.detect_paintings(image)
        query_predictions = []

        for mask in masks:
            crop = w3u.crop_to_mask(image, mask)
            query_feature = describe(
                crop, descriptor, color, clahe, keypoint_detector
            )
            scores = np.asarray(
                [
                    match_score(query_feature, museum_feature, norm)
                    for museum_feature in museum_features
                ],
                dtype=float,
            )

            if not len(scores) or float(scores.max()) < threshold:
                query_predictions.append([-1])
            else:
                ranked = np.argsort(-scores)[:K]
                query_predictions.append(
                    [museum_ids[index] for index in ranked]
                )

        if not masks:
            print(f"{path.name}: no paintings detected")
        else:
            print(
                f"{path.name}: detected {len(masks)} painting(s)",
                flush=True,
            )
        all_predictions.append(query_predictions)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as f:
        pickle.dump(all_predictions, f)

    print(f"Selected configuration: {config}")
    print(f"Saved {len(all_predictions)} query results to {OUTPUT}")


if __name__ == "__main__":
    main()
