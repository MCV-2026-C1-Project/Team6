from utils import *
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from segmentation import detect_paintings

import descriptors as desc_lib
from denoising import denoise_image
from keypoints import detect, resize_max_side


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QUERY_DIR = DATA_DIR / "qsd1_w4"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
GROUND_TRUTH_DIR = QUERY_DIR / "gt_corresps.pkl"

RUN_TESTS = True
TEST_DIR = DATA_DIR / "qst1_w4"
K = 10
N_WORKERS = os.cpu_count()

UNKNOWN = -1  # prediction for a painting that is not in the database

PARAMS = {
    "descriptor": "rootsift",   # key of descriptors.DESCRIPTORS
    "keypoints": None,          # key of keypoints.DETECTORS, None = descriptor's own detector
    "color": "gray",            # one of descriptors.COLORS
    "clahe": True,
    "max_kps": 1000,
    "max_side": 800,            # images are resized so the scales are comparable
    "ratio": 0.75,              # Lowe's ratio test
    "reproj": 5.0,              # RANSAC reprojection threshold (px)
    "threshold": 25,            # min inliers of the best match, below it -> UNKNOWN
}


# =============================================================================
# DESCRIPTION
# =============================================================================

def describe(img: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the local features of an image.

    Args:
        img: BGR image.
        params: Pipeline parameters (descriptor, keypoints, color, clahe,
            max_kps, max_side).

    Returns:
        Tuple (points, descriptors): keypoint coordinates (N, 2) and their
        descriptors (N, D), or None descriptors if no keypoint was found.
    """
    img = resize_max_side(img, params["max_side"])
    name, kp = params["descriptor"], params["keypoints"]
    color, clahe, max_kps = params["color"], params["clahe"], params["max_kps"]

    if kp is None:
        return desc_lib.DESCRIPTORS[name](img, max_kps, color, clahe)
    if name != "rootsift":
        raise ValueError("A custom keypoint detector is only supported with rootsift")
    kps = detect(img, kp, max_kps=max_kps, clahe=clahe)
    return desc_lib.rootsift(img, max_kps, color, clahe, kps=kps)


def describe_database(db_images: list[np.ndarray], params: dict) -> list:
    """Local features of every database image."""
    return [describe(img, params) for img in tqdm(db_images, desc="BBDD")]


def process_queries(query_images: list[np.ndarray], params: dict):
    """
    Detect up to MAX_PAINTINGS paintings per query and describe each of them.

    Each query is denoised, its paintings are detected and every painting is
    cropped to its mask and described with the given params.

    Args:
        query_images: Query images in BGR format.
        params: Pipeline parameters passed to describe.

    Returns:
        Tuple (feats_by_query, masks_by_query) where feats_by_query[i] is the
        list of features (one per detected painting) of query i, and
        masks_by_query[i] is the list of detected masks of query i.
    """
    feats_by_query, masks_by_query = [], []

    with open(QUERY_DIR / "frames.pkl", "rb") as f:  # TEMPORARY: GT frames
        gt_frames = pickle.load(f)

    for img in tqdm(query_images, desc="Queries"):
        img = denoise_image(img)
        masks = detect_paintings(img)
        masks_by_query.append(masks)
        feats_by_query.append([describe(crop_to_mask(img, m), params) for m in masks])

    return feats_by_query, masks_by_query


# =============================================================================
# MATCHING
# =============================================================================

def match_score(q, m, norm: int, ratio: float = 0.75, reproj: float = 5.0) -> int:
    """
    Number of geometrically consistent matches between two images.

    1) Lowe's ratio test on the 2-NN matches, 2) RANSAC homography (paintings
    are planar) on the surviving matches.

    Args:
        q, m: Features (points, descriptors) of the query and database image.
        norm: OpenCV norm for the descriptors (NORM_L2 / NORM_HAMMING).

    Returns:
        RANSAC inlier count (0 if there are not enough matches).
    """
    (pq, dq), (pm, dm) = q, m
    if dq is None or dm is None or len(dq) < 4 or len(dm) < 2:
        return 0
    knn = cv2.BFMatcher(norm).knnMatch(dq, dm, k=2)
    good = [a for a, b in (x for x in knn if len(x) == 2) if a.distance < ratio * b.distance]
    if len(good) < 4:
        return 0
    src = pq[[g.queryIdx for g in good]]
    dst = pm[[g.trainIdx for g in good]]
    _, inliers = cv2.findHomography(src, dst, cv2.RANSAC, reproj)
    return int(inliers.sum()) if inliers is not None else 0


def _init_worker(db_feats, norm, ratio, reproj):
    global _DB, _NORM, _RATIO, _REPROJ
    _DB, _NORM, _RATIO, _REPROJ = db_feats, norm, ratio, reproj
    cv2.setNumThreads(1)


def _score_row(q):
    return [match_score(q, m, _NORM, _RATIO, _REPROJ) for m in _DB]


def score_matrix(query_feats: list, db_feats: list, params: dict, n_workers: int = N_WORKERS) -> np.ndarray:
    """
    S[i, j] = inliers between painting i and database image j.

    Args:
        query_feats: Flat list of painting features.
        db_feats: Features of the database images.
        params: Pipeline parameters (descriptor, ratio, reproj).
        n_workers: Number of processes (1 = no multiprocessing).

    Returns:
        Integer array of shape (len(query_feats), len(db_feats)).
    """
    if not query_feats:
        return np.zeros((0, len(db_feats)), int)

    init_args = (db_feats, desc_lib.NORMS[params["descriptor"]], params["ratio"], params["reproj"])
    if n_workers == 1:
        _init_worker(*init_args)
        rows = [_score_row(q) for q in tqdm(query_feats, desc="Matching")]
    else:
        with Pool(n_workers, _init_worker, init_args) as pool:
            rows = list(tqdm(pool.imap(_score_row, query_feats), total=len(query_feats), desc="Matching"))
    return np.array(rows, int)


def query_scores(feats_by_query: list, db_feats: list, params: dict, n_workers: int = N_WORKERS) -> list:
    """
    Score every detected painting of every query against the database.

    Returns:
        One array of shape (n_paintings_i, n_database) per query.
    """
    flat = [f for feats in feats_by_query for f in feats]
    S = score_matrix(flat, db_feats, params, n_workers)
    splits = np.cumsum([len(f) for f in feats_by_query])[:-1]
    return np.split(S, splits)


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve_unknown(scores_by_query: list, db_ids: list[int], k: int, threshold: float) -> list:
    """
    Top-k database IDs per painting, or [UNKNOWN] if the painting is not in it.

    A painting is considered unknown when its best score is below threshold.

    Args:
        scores_by_query: One array (n_paintings_i, n_database) per query
            (higher is better).
        db_ids: Database IDs of the score columns.
        k: Number of IDs retrieved per painting.
        threshold: Min best score to accept the retrieval.

    Returns:
        [[[ids painting 1], [ids painting 2]], [[-1]], ...]
    """
    db_ids = np.asarray(db_ids)
    preds = []
    for S in scores_by_query:
        pred_q = []
        for row in S:
            if row.max() < threshold:
                pred_q.append([UNKNOWN])
            else:
                pred_q.append(db_ids[np.argsort(-row, kind="stable")[:k]].tolist())
        preds.append(pred_q)
    return preds


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(
    query_images: list[np.ndarray],
    db_images: list[np.ndarray],
    db_ids: list[int],
    params: dict,
    k: int = K,
    ground_truth: list | None = None,
    db_feats: list | None = None,
    n_workers: int = N_WORKERS,
) -> dict:
    """
    Full W4 pipeline: denoise -> detect paintings -> local features ->
    matching (ratio test + RANSAC) -> retrieval with unknowns (-1) -> mAP@k.

    Reusable from the grid search: pass db_feats to avoid recomputing the
    database when only the matching/threshold params change, and reuse the
    returned "scores" with retrieve_unknown to sweep the threshold for free.

    Args:
        query_images: Query images in BGR format.
        db_images: Database images in BGR format.
        db_ids: Database IDs.
        params: Pipeline parameters (see PARAMS).
        k: Number of IDs retrieved per painting.
        ground_truth: Optional GT per query (e.g. [[-1], [150], [48, 251]]).
        db_feats: Optional precomputed database features for these params.
        n_workers: Number of processes for the matching.

    Returns:
        Dict with "predictions", "scores", "masks", "db_feats" and, if
        ground_truth is given, "map1" and "map5".
    """
    if db_feats is None:
        db_feats = describe_database(db_images, params)

    feats_by_query, masks_by_query = process_queries(query_images, params)
    scores = query_scores(feats_by_query, db_feats, params, n_workers)
    predictions = retrieve_unknown(scores, db_ids, k, params["threshold"])

    out = {"predictions": predictions, "scores": scores, "masks": masks_by_query, "db_feats": db_feats}
    if ground_truth is not None:
        out["map1"] = mapk_multi(ground_truth, predictions, k=1)
        out["map5"] = mapk_multi(ground_truth, predictions, k=5)
    return out


if __name__ == "__main__":
    db_ids, db_images = load_images(BBDD_DIR)
    query_ids, query_images = load_images(QUERY_DIR)

    with open(GROUND_TRUTH_DIR, "rb") as f:
        ground_truth = pickle.load(f)

    ground_truth = [ground_truth[qid] for qid in query_ids]

    res = run_pipeline(query_images, db_images, db_ids, PARAMS, K, ground_truth)

    n_detected = [len(m) for m in res["masks"]]
    n_unknown = sum(p == [UNKNOWN] for q in res["predictions"] for p in q)
    n_gt = [len(g) for g in ground_truth]
    n_gt_unknown = sum(p == UNKNOWN for g in ground_truth for p in g)
    print("\nResults:")
    sep = "-" * 32
    print(sep)
    print(f"Descriptor: {PARAMS['descriptor']} ({PARAMS['keypoints'] or 'own'} keypoints)")
    print(f"Detections:   0 -> {n_detected.count(0)}, 1 -> {n_detected.count(1)}, "
          f"2 -> {n_detected.count(2)}")
    print(f"Ground truth: 0 -> {n_gt.count(0)}, 1 -> {n_gt.count(1)}, "
          f"2 -> {n_gt.count(2)}")
    print(f"Unknown: predicted {n_unknown} / ground truth {n_gt_unknown} "
          f"(of {sum(n_gt)} paintings)")
    print(f"{'mAP@1':>10}{'mAP@5':>10}")
    print(sep)
    print(f"{res['map1']:>10.4f}{res['map5']:>10.4f}")
    print(sep)

    if RUN_TESTS and TEST_DIR.exists():
        name = "QST1"
        print(f"\nRunning test {name}...")
        test_ids, test_images = load_images(TEST_DIR)
        test = run_pipeline(test_images, db_images, db_ids, PARAMS, K, db_feats=res["db_feats"])

        out_test = RESULTS_DIR / name / "method1"
        out_test.mkdir(parents=True, exist_ok=True)
        with open(out_test / "result.pkl", "wb") as f:
            pickle.dump(test["predictions"], f)

        print(f"{name} results saved to {out_test}")
