"""Demo: image descriptor (keypoints + local descriptors) -> matching -> retrieval -> map@k.

    python main.py --museum data/BBDD --queries data/qsd1_w4 [--gt data/qsd1_w4/gt_corresps.pkl] [--k 5]

Several configurations are compared (max 3 descriptors, as required)."""
import argparse
import glob
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from keypoints import resize_max_side
from local_descriptors import extract

Feat = Tuple[np.ndarray, np.ndarray]  # (keypoint coordinates Nx2, descriptors NxD)

# name -> (detector, descriptor, detector kwargs, descriptor kwargs)
CONFIGS: Dict[str, Tuple[str, str, dict, dict]] = {
    "dog+rootsift": ("dog", "sift", {"max_kps": 1500}, {}),
    "hybrid+colorsift": ("hybrid", "color_sift", {"max_kps": 2000}, {}),
    "harlap+orb": ("harris_laplace", "orb", {"max_kps": 1500}, {}),
}


def load_features(paths: Sequence[str], det: str, desc: str, det_kw: dict, desc_kw: dict,
                  max_side: int = 800) -> List[Feat]:
    """Extract (points, descriptors) for every image. Images are resized so scales/thresholds are consistent."""
    feats = []
    for p in paths:
        kps, des = extract(resize_max_side(cv2.imread(p), max_side), det, desc, det_kw, desc_kw)
        feats.append((np.float32([k.pt for k in kps]).reshape(-1, 2), des))
    return feats


def match_score(q: Feat, m: Feat, norm: int, ratio: float = 0.75, reproj: float = 5.0) -> int:
    """Similarity = number of geometrically consistent matches.
    1) Lowe's ratio test on 2-NN matches (appearance), 2) RANSAC homography (paintings are planar) -> inlier count."""
    (pq, dq), (pm, dm) = q, m
    if len(dq) < 4 or len(dm) < 2:
        return 0
    knn = cv2.BFMatcher(norm).knnMatch(dq, dm, k=2)
    good = [a for a, b in (x for x in knn if len(x) == 2) if a.distance < ratio * b.distance]
    if len(good) < 4:
        return 0
    src = pq[[g.queryIdx for g in good]]
    dst = pm[[g.trainIdx for g in good]]
    _, inl = cv2.findHomography(src, dst, cv2.RANSAC, reproj)
    return int(inl.sum()) if inl is not None else 0


def retrieve(q: Feat, museum: Sequence[Feat], norm: int, k: int = 5, min_inliers: int = 15) -> List[int]:
    """Top-k museum indices by score. If even the best has < min_inliers the painting is assumed
    not to be in the museum -> [-1]. Tune min_inliers (and `ratio`) on QSD1: it is the key to unknown queries."""
    scores = np.array([match_score(q, m, norm) for m in museum])
    top = np.argsort(-scores)[:k]
    return [-1] if scores[top[0]] < min_inliers else top.tolist()


def apk(actual: List[int], pred: List[int], k: int) -> float:
    """Average precision at k for one query."""
    pred, hits, score = pred[:k], 0, 0.0
    for i, p in enumerate(pred):
        if p in actual and p not in pred[:i]:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual), k) if actual else 0.0


def mapk(actual: List[List[int]], pred: List[List[int]], k: int) -> float:
    """Mean average precision at k over all queries."""
    return float(np.mean([apk(a, p, k) for a, p in zip(actual, pred)]))


def main(museum_dir: str, query_dir: str, gt_path: Optional[str], k: int) -> None:
    mus_paths = sorted(glob.glob(f"{museum_dir}/*.jpg"))
    qry_paths = sorted(glob.glob(f"{query_dir}/*.jpg"))
    gt = pickle.load(open(gt_path, "rb")) if gt_path else None
    for name, (det, desc, det_kw, desc_kw) in CONFIGS.items():
        norm = cv2.NORM_HAMMING if desc == "orb" else cv2.NORM_L2
        museum = load_features(mus_paths, det, desc, det_kw, desc_kw)
        queries = load_features(qry_paths, det, desc, det_kw, desc_kw)
        pred = [retrieve(q, museum, norm, k) for q in queries]
        if gt:
            print(f"{name:18s} map@1={mapk(gt, pred, 1):.3f}  map@{k}={mapk(gt, pred, k):.3f}")
        else:
            print(name, pred[:5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--museum", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gt", default=None, help="pickle with a list of lists of ground-truth museum ids")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    main(a.museum, a.queries, a.gt, a.k)
