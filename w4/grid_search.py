"""Grid search on QSD1-W4: descriptor x colour x CLAHE (x keypoint detector, rootsift only).

    python grid_search.py

Query paintings are cropped with the ground-truth frames and median-filtered if they have impulse noise.
Score of a (query, museum) pair = RANSAC inliers. A query is predicted unknown (-1) when its best
score is below a threshold, chosen here on the same queries (optimistic).
Results are printed and saved to results/grid_search.csv.
"""
import csv
import itertools
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import cv_descriptors as cvd
from cv_descriptors import COLORS
from keypoints import detect, resize_max_side

DESCRIPTORS = ["rootsift", "akaze", "orb"]
KEYPOINTS = [None, "dog", "harris_laplace", "hybrid"]  # None = the descriptor's own detector
CLAHE = [True, False]
MAX_KPS = 1000
OUT = "results/grid_search.csv"
DATA = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------- data
def denoise(img):
    """3x3 median filter if > 1% of the pixels differ by > 30 from all 8 neighbours (impulse noise, as in W3)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    p = np.pad(g, 1, mode="edge")
    h, w = g.shape
    isolated = np.ones_like(g, bool)
    for dy, dx in itertools.product((-1, 0, 1), repeat=2):
        if dy or dx:
            isolated &= np.abs(g - p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]) > 30
    return cv2.medianBlur(img, 3) if isolated.mean() > 0.01 else img


def load():
    """Museum images, one crop per query painting (ground-truth frame) and its museum id (-1 = unknown)."""
    museum = [resize_max_side(cv2.imread(str(p))) for p in sorted((DATA / "BBDD").glob("*.jpg"))]
    qsd = DATA / "qsd1_w4"
    gt = pickle.load(open(qsd / "gt_corresps.pkl", "rb"))
    frames = pickle.load(open(qsd / "frames.pkl", "rb"))
    crops, labels = [], []
    for i, (ids, quads) in enumerate(zip(gt, frames)):
        img = denoise(cv2.imread(str(qsd / f"{i:05d}.jpg")))
        for painting_id, (_, quad) in zip(ids, quads):
            x, y, w, h = cv2.boundingRect(np.int32(quad))
            crops.append(resize_max_side(img[max(y, 0):y + h, max(x, 0):x + w]))
            labels.append(painting_id)
    return museum, crops, np.array(labels)


def grid():
    for desc, color, clahe in itertools.product(DESCRIPTORS, COLORS, CLAHE):
        for kp in KEYPOINTS if desc == "rootsift" else [None]:
            yield desc, color, clahe, kp


def features(img, desc, color, clahe, kp):
    if kp is None:
        return cvd.DESCRIPTORS[desc](img, MAX_KPS, color, clahe)
    return cvd.rootsift(img, MAX_KPS, color, clahe, kps=detect(img, kp, max_kps=MAX_KPS, clahe=clahe))


# ---------------------------------------------------------------- matching (in parallel)
def score(q, m, norm, ratio=0.75, reproj=5.0):
    """Inliers of a RANSAC homography fitted on the matches that pass Lowe's ratio test."""
    (pq, dq), (pm, dm) = q, m
    if dq is None or dm is None or len(dq) < 4 or len(dm) < 2:
        return 0
    good = [a for a, b in (x for x in cv2.BFMatcher(norm).knnMatch(dq, dm, k=2) if len(x) == 2)
            if a.distance < ratio * b.distance]
    if len(good) < 4:
        return 0
    _, inliers = cv2.findHomography(pq[[g.queryIdx for g in good]], pm[[g.trainIdx for g in good]],
                                    cv2.RANSAC, reproj)
    return int(inliers.sum()) if inliers is not None else 0


def _init(museum, norm):
    global MUSEUM, NORM
    MUSEUM, NORM = museum, norm
    cv2.setNumThreads(1)


def _scores(query):
    return [score(query, m, NORM) for m in MUSEUM]


def score_matrix(queries, museum, norm):
    """S[i, j] = inliers between query painting i and museum image j."""
    with Pool(os.cpu_count(), _init, (museum, norm)) as pool:
        return np.array(pool.map(_scores, queries))


# ---------------------------------------------------------------- metrics
def ap5(gt, ranking):
    top = list(ranking[:5])
    return 1 / (top.index(gt) + 1) if gt in top else 0.0


def evaluate(S, labels):
    """mAP@5 of the known paintings, and mAP@5 of all paintings with the best -1 threshold."""
    known = labels >= 0
    ap = np.array([ap5(gt, r) for gt, r in zip(labels, np.argsort(-S, axis=1))])
    best = S.max(axis=1)

    def map_with_unknowns(t):  # unknown painting: correct if best < t; known: needs best >= t
        return np.where(known, ap * (best >= t), best < t).mean()

    t = max(range(1, 101), key=map_with_unknowns)
    return ap[known].mean(), map_with_unknowns(t), t


def main():
    museum_imgs, query_imgs, labels = load()
    rows = []
    for desc, color, clahe, kp in grid():
        museum = [features(im, desc, color, clahe, kp) for im in museum_imgs]
        queries = [features(im, desc, color, clahe, kp) for im in query_imgs]
        S = score_matrix(queries, museum, cvd.NORMS[desc])
        map5, map5_all, thr = evaluate(S, labels)
        row = dict(descriptor=desc, color=color, clahe=clahe, keypoints=kp or "own",
                   map5=round(map5, 3), map5_unknowns=round(map5_all, 3), threshold=thr)
        rows.append(row)
        print(row, flush=True)

    rows.sort(key=lambda r: (-r["map5_unknowns"], -r["map5"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("\nBest 10:")
    for r in rows[:10]:
        print(r)


if __name__ == "__main__":
    main()
