"""Week 4 local-feature matching and QSD1-W4 evaluation.

Run from the repository root:
    python w4/grid_search.py

Requires:
    data/BBDD/*.jpg
    data/qsd1_w4/*.jpg
    data/qsd1_w4/frames.pkl
    data/qsd1_w4/gt_corresps.pkl

The QSD1 frames are used to crop one painting at a time. Unknown-painting
thresholds are selected by F1 on QSD1-W4. mAP is reported on known paintings;
unknown rejection F1 is reported separately.
"""
import csv
import itertools
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import w4.descriptors as desc
from w4.descriptors import COLORS
from keypoints import detect, resize_max_side
from w3 import utils as w3u

DATA = ROOT / "data"
OUT = ROOT / "w4" / "results" / "task2_task3_results.csv"

DESCRIPTORS = ("rootsift", "akaze", "orb")
KEYPOINTS = (None, "dog", "harris_laplace", "hybrid")
CLAHE_VALUES = (True, False)
MAX_KPS = 1000
TOP_K_VALUES = (1, 5)
MAX_WORKERS = 8


def denoise(img):
    """Apply the Week 3 impulse-noise check and median filter."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    padded = np.pad(gray, 1, mode="edge")
    h, w = gray.shape
    isolated = np.ones_like(gray, dtype=bool)
    for dy, dx in itertools.product((-1, 0, 1), repeat=2):
        if dy or dx:
            neighbor = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            isolated &= np.abs(gray - neighbor) > 30
    return cv2.medianBlur(img, 3) if isolated.mean() > 0.01 else img


def museum_id(path):
    """Get a numeric museum ID from names such as bbdd_00150.jpg or 00150.jpg."""
    return int(path.stem.split("_")[-1])


def load_data():
    """Load museum images and crop each QSD1 painting using its provided frame."""
    museum_paths = sorted((DATA / "BBDD").glob("*.jpg"))
    qsd = DATA / "qsd1_w4"
    if not museum_paths:
        raise FileNotFoundError(f"No museum JPGs found in {DATA / 'BBDD'}")

    museum_ids = [museum_id(path) for path in museum_paths]
    museum_images = []
    museum_images_raw = []
    for path in museum_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read museum image: {path}")
        museum_images_raw.append(img)
        museum_images.append(resize_max_side(img))

    with open(qsd / "gt_corresps.pkl", "rb") as f:
        ground_truth = pickle.load(f)
    with open(qsd / "frames.pkl", "rb") as f:
        frames = pickle.load(f)

    if len(ground_truth) != len(frames):
        raise ValueError("gt_corresps.pkl and frames.pkl have different query counts")

    crops, labels = [], []
    for image_index, (ids, image_frames) in enumerate(zip(ground_truth, frames)):
        path = qsd / f"{image_index:05d}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read query image: {path}")
        img = denoise(img)

        if len(ids) != len(image_frames):
            raise ValueError(f"Query {image_index}: GT IDs and frames do not line up")

        for painting_id, frame in zip(ids, image_frames):
            _, quad = frame
            x, y, w, h = cv2.boundingRect(np.asarray(quad, dtype=np.int32))
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                raise ValueError(f"Query {image_index}: empty crop from frame")
            crops.append(crop)
            labels.append(int(painting_id))

    return museum_ids, museum_images, crops, np.asarray(labels, dtype=int), museum_images_raw


def config_grid():
    for descriptor, color, clahe in itertools.product(DESCRIPTORS, COLORS, CLAHE_VALUES):
        detectors = KEYPOINTS if descriptor == "rootsift" else (None,)
        for keypoint_detector in detectors:
            yield descriptor, color, clahe, keypoint_detector


def describe(img, descriptor, color, clahe, keypoint_detector):
    img = resize_max_side(img)
    if keypoint_detector is None:
        return desc.DESCRIPTORS[descriptor](img, MAX_KPS, color, clahe)
    keypoints = detect(img, keypoint_detector, max_kps=MAX_KPS, clahe=clahe)
    return desc.rootsift(img, MAX_KPS, color, clahe, kps=keypoints)


def match_score(query, museum, norm, ratio=0.75, reprojection_error=5.0):
    """Count descriptor matches that agree with one planar RANSAC homography."""
    query_points, query_desc = query
    museum_points, museum_desc = museum
    if query_desc is None or museum_desc is None:
        return 0
    if len(query_desc) < 4 or len(museum_desc) < 2:
        return 0

    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(query_desc, museum_desc, k=2)
    good = [first for pair in pairs if len(pair) == 2
            for first, second in [pair] if first.distance < ratio * second.distance]
    if len(good) < 4:
        return 0

    src = query_points[[m.queryIdx for m in good]]
    dst = museum_points[[m.trainIdx for m in good]]
    try:
        _, inlier_mask = cv2.findHomography(
            src, dst, cv2.RANSAC, reprojection_error
        )
    except cv2.error:
        return 0
    return int(inlier_mask.sum()) if inlier_mask is not None else 0


def _init_workers(museum_features, norm):
    global _MUSEUM_FEATURES, _NORM
    _MUSEUM_FEATURES = museum_features
    _NORM = norm
    cv2.setNumThreads(1)


def _score_one_query(query):
    return [match_score(query, museum, _NORM) for museum in _MUSEUM_FEATURES]


def local_score_matrix(query_features, museum_features, norm):
    workers = max(1, min(MAX_WORKERS, mp.cpu_count() or 1))
    with mp.Pool(workers, _init_workers, (museum_features, norm)) as pool:
        return np.asarray(pool.map(_score_one_query, query_features), dtype=float)


def choose_unknown_threshold(best_scores, labels, higher_is_known):
    """Choose the threshold with the best unknown-class F1 on QSD1."""
    truth_unknown = labels < 0

    values = np.unique(np.asarray(best_scores, dtype=float))
    if len(values) == 1:
        candidates = np.array(
            [values[0] - 1, values[0], values[0] + 1],
            dtype=float,
        )
    else:
        midpoints = (values[:-1] + values[1:]) / 2
        candidates = np.concatenate(
            ([values[0] - 1], midpoints, [values[-1] + 1])
        )

    best_threshold, best_f1 = candidates[0], -1.0

    for threshold in candidates:
        predicted_unknown = (
            best_scores < threshold
            if higher_is_known
            else best_scores > threshold
        )

        tp = np.count_nonzero(predicted_unknown & truth_unknown)
        fp = np.count_nonzero(predicted_unknown & ~truth_unknown)
        fn = np.count_nonzero(~predicted_unknown & truth_unknown)

        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 0.0

        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1

    return float(best_threshold), float(best_f1)


def map_known(labels, rankings, best_scores, threshold, higher_is_known):
    """Compute mAP@1 and mAP@5 over known paintings after unknown rejection."""
    accepted = (best_scores >= threshold if higher_is_known
                else best_scores <= threshold)
    known_rows = np.flatnonzero(labels >= 0)
    if not len(known_rows):
        return {k: 0.0 for k in TOP_K_VALUES}

    result = {}
    for k in TOP_K_VALUES:
        ap_values = []
        for row in known_rows:
            if not accepted[row]:
                ap_values.append(0.0)
                continue
            top = rankings[row, :k]
            places = np.flatnonzero(top == labels[row])
            ap_values.append(1.0 / (int(places[0]) + 1) if len(places) else 0.0)
        result[k] = float(np.mean(ap_values))
    return result


def evaluate_rankings(labels, rankings, best_scores, higher_is_known):
    threshold, unknown_f1 = choose_unknown_threshold(
        best_scores, labels, higher_is_known
    )
    maps = map_known(labels, rankings, best_scores, threshold, higher_is_known)
    return threshold, unknown_f1, maps


def rank_local(S, db_ids):
    order = np.argsort(-S, axis=1)
    rankings = np.asarray(db_ids, dtype=int)[order]
    return rankings, S.max(axis=1)


def run_w3_baseline(museum_images_raw, query_images, labels, db_ids):
    """Run Week 3's best DCT-pyramid configuration on the same QSD1 frame crops."""
    params = {
        "descriptor": w3u.dct_pyramid_descriptor,
        "color_code": cv2.COLOR_BGR2YCrCb,
        "channels": [0, 1, 2],
        "size": 672,
        "kwargs": {"levels": [1, 2, 3], "n_coef": 32, "skip_dc": False},
    }
    db_desc = np.vstack([w3u.describe(img, params) for img in museum_images_raw])
    query_desc = np.vstack([w3u.describe(img, params) for img in query_images])
    distances = w3u.get_distances(query_desc, db_desc, metric="canberra")
    order = np.argsort(distances, axis=1)
    rankings = np.asarray(db_ids, dtype=int)[order]
    best_distances = distances.min(axis=1)
    threshold, unknown_f1, maps = evaluate_rankings(
        labels, rankings, best_distances, higher_is_known=False
    )
    return threshold, unknown_f1, maps


def main():
    db_ids, museum_images, query_images, labels, museum_images_raw = load_data()
    results = []

    for descriptor, color, clahe, keypoint_detector in config_grid():
        museum_features = [
            describe(img, descriptor, color, clahe, keypoint_detector)
            for img in museum_images
        ]
        query_features = [
            describe(img, descriptor, color, clahe, keypoint_detector)
            for img in query_images
        ]
        norm = desc.NORMS[descriptor]
        scores = local_score_matrix(query_features, museum_features, norm)
        rankings, best_scores = rank_local(scores, db_ids)
        threshold, unknown_f1, maps = evaluate_rankings(
            labels, rankings, best_scores, higher_is_known=True
        )

        row = {
            "method": "week4_local",
            "descriptor": descriptor,
            "color": color,
            "clahe": clahe,
            "keypoint_detector": keypoint_detector or "descriptor_default",
            "unknown_threshold": threshold,
            "unknown_f1": unknown_f1,
            "map@1": maps[1],
            "map@5": maps[5],
        }
        results.append(row)
        print(row, flush=True)

    print("Running Week 3 DCT baseline on the same QSD1 frame crops...")
    threshold, unknown_f1, maps = run_w3_baseline(
        museum_images_raw, query_images, labels, db_ids
    )
    baseline = {
        "method": "week3_dct_baseline",
        "descriptor": "DCT pyramid",
        "color": "YCrCb",
        "clahe": "",
        "keypoint_detector": "",
        "unknown_threshold": threshold,
        "unknown_f1": unknown_f1,
        "map@1": maps[1],
        "map@5": maps[5],
    }
    results.append(baseline)
    print(baseline, flush=True)

    results.sort(key=lambda row: (row["method"] != "week3_dct_baseline",
                                 -row["map@5"], -row["unknown_f1"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    local_rows = [row for row in results if row["method"] == "week4_local"]
    best_local = max(local_rows, key=lambda row: (row["map@5"], row["unknown_f1"]))
    print("\nBest Week 4 local method:", best_local)
    print("Week 3 baseline:", baseline)
    print(f"Saved results to {OUT}")


if __name__ == "__main__":
    main()
