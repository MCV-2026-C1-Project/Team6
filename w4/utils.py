import os
import pickle
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# =============================================================================
# PREPROCESSING
# =============================================================================

def to_gray(image: np.ndarray, clahe: bool = True) -> np.ndarray:
    """
    Convert an image to gray, optionally equalising its local contrast.

    Args:
        image: BGR or gray image.
        clahe: Apply CLAHE (clip limit 2, 8x8 tiles) to the gray image.

    Returns:
        uint8 gray image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if not clahe:
        return gray
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def resize_max_side(image: np.ndarray, max_side: int = 800) -> np.ndarray:
    """
    Downscale an image so its longest side is at most max_side.

    Args:
        image: Input image.
        max_side: Maximum length of the longest side, in pixels.

    Returns:
        The resized image, or the same image if it is already small enough
        (it is never upscaled).
    """
    scale = max_side / max(image.shape[:2])
    if scale >= 1:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


# =============================================================================
# TOOLS
# =============================================================================

def load_images(folder: str, color_space: int = None) -> tuple[list[int], list[np.ndarray]]:
    """
    Load the JPG images of a folder, optionally converting their color space.

    Only .jpg files are read because they are the only ones that contain the
    required images.

    Args:
        folder: Path to the folder containing the images.
        color_space: OpenCV color conversion code (e.g. cv2.COLOR_BGR2Lab).
            If None, the images are kept in BGR.

    Returns:
        Tuple (ids, images) where ids[i] is the ID of images[i].
    """
    ids: list[int] = []
    images: list[np.ndarray] = []

    for path in sorted(Path(folder).glob("*.jpg")):
        img = cv2.imread(str(path))

        if img is None:
            print(f"Could not read {path}")
            continue

        if color_space:
            img = cv2.cvtColor(img, color_space)
        image_id = int(path.stem.split("_")[-1])

        ids.append(image_id)
        images.append(img)

    return ids, images


def load_pickle(path: str | Path):
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

# =============================================================================
# EVALUATION
# =============================================================================

# APK and MAPK are adapted from https://github.com/benhamner/Metrics.
# The ml_metrics package is no longer compatible with modern versions of setuptools, so the required functions are included here directly.
def apk(actual: list[int], predicted: list[int], k: int = 10) -> float:
    """
    Compute the average precision at k (AP@k) for one query.

    The order of items in ``actual`` does not matter, while the order in
    ``predicted`` determines the ranking used to compute the average precision.

    Args:
        actual: Ground-truth BBDD IDs of the query.
        predicted: Ranked BBDD IDs retrieved for the query.
        k: Maximum number of top-ranked predictions to consider.

    Returns:
        The average precision at k (AP@k), as a float between 0 and 1.
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual: list[list[int]], predicted: list[list[int]], k: int = 10) -> float:
    """
    Compute the mean average precision at k (mAP@k) for retrieval results.

    The metric compares the ranked retrieval results against the ground-truth
    items for each query. The order of items in ``actual`` does not matter,
    while the order in ``predicted`` determines the ranking used to compute
    average precision.

    Args:
        actual: Ground-truth BBDD IDs for each query. Each inner list contains
            the correct BBDD IDs for the corresponding query.
        predicted: Ranked BBDD IDs retrieved for each query. The order of
            elements determines their ranking.
        k: Maximum number of top-ranked predictions to consider for each query.

    Returns:
        The mean average precision at k (mAP@k), as a float between 0 and 1.
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def mapk_multi(actual: list, predicted: list[list[list[int]]], k: int) -> float:
    """
    Compute the mean average precision at k (mAP@k) per painting.

    A painting in the ground truth that was not detected gets AP = 0.

    Args:
        actual: Ground truth of each query, with one BBDD ID per painting in the
            same order as the detections, e.g. [120] or [120, 170].
        predicted: For each query, one ranked list of BBDD IDs per detected
            painting.
        k: Maximum number of top-ranked predictions to consider per painting.

    Returns:
        The mean of the AP@k over all ground-truth paintings, as a float
        between 0 and 1.
    """
    aps = []
    for gt_q, pred_q in zip(actual, predicted):
        for j, gt_painting in enumerate(gt_q):
            gt_painting = list(gt_painting) if isinstance(gt_painting, (list, tuple)) else [gt_painting]
            pred = pred_q[j] if j < len(pred_q) else []
            aps.append(apk(gt_painting, pred, k))
    return float(np.mean(aps)) if aps else 0.0


def evaluate(actual: list[list[int]], predicted: list[list[int]], k: int):
    """
    Compute the mean average precision at k (mAP@k) for retrieval results.

    The query folder can have fewer images than the ground truth (qsd1_w1 has
    24 images for 30 entries), so predictions can be given as a dict to match
    them by query ID.

    Args:
        actual: Ground-truth BBDD IDs for each query, indexed by query ID.
        predicted: Ranked BBDD IDs retrieved for each query, as a list in the
            same order as ``actual`` or as a dict {query_id: [ids]}.
        k: Maximum number of top-ranked predictions to consider for each query.

    Returns:
        The mean average precision at k (mAP@k), as a float between 0 and 1.

    Raises:
        ValueError: If ``predicted`` is a list whose length differs from
            ``actual``.
    """
    if isinstance(predicted, dict):
        query_ids = sorted(predicted)
        actual = [actual[qid] for qid in query_ids]
        predicted = [list(predicted[qid]) for qid in query_ids]

    elif len(actual) != len(predicted):
        raise ValueError(
            f'Got {len(predicted)} predictions for {len(actual)} ground truth '
            'entries. Pass predictions as a dict {query_id: [ids]}.'
        )

    return mapk(actual, predicted, k)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute pixel-level metrics of a predicted mask.

    Args:
        pred: Predicted binary mask.
        gt: Ground-truth binary mask of the same shape.

    Returns:
        Dictionary with the keys "precision", "recall", "f1", "iou" and
        "error_rate".
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.count_nonzero(pred & gt)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(~pred & gt)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "error_rate": (fp + fn) / gt.size,
    }


# =============================================================================
# MATCHING
# =============================================================================

def count_inliers(query_features, db_features, norm, lowe_ratio=0.75, ransac_reproj_px=5.0) -> int:
    """Inliers of a RANSAC homography fitted on the matches that pass Lowe's ratio test."""
    query_points, query_descriptors = query_features
    db_points, db_descriptors = query_features, db_features

    if query_descriptors is None or db_descriptors is None:
        return 0
    if len(query_descriptors) < 4 or len(db_descriptors) < 2:
        return 0

    pairs = cv2.BFMatcher(norm).knnMatch(query_descriptors, db_descriptors, k=2)
    good = [best for best, second in (pair for pair in pairs if len(pair) == 2)
            if best.distance < lowe_ratio * second.distance]
    if len(good) < 4:
        return 0

    _, inliers = cv2.findHomography(query_points[[match.queryIdx for match in good]],
                                    db_points[[match.trainIdx for match in good]],
                                    cv2.RANSAC, ransac_reproj_px)
    return int(inliers.sum()) if inliers is not None else 0


def _init_worker(db_features, norm, lowe_ratio, ransac_reproj_px):
    global DB_FEATURES, MATCH_ARGS
    DB_FEATURES, MATCH_ARGS = db_features, (norm, lowe_ratio, ransac_reproj_px)
    cv2.setNumThreads(1)


def _inliers_row(query_features):
    return [count_inliers(query_features, db, *MATCH_ARGS) for db in DB_FEATURES]


def inlier_matrix(paintings_features, db_features, norm, lowe_ratio=0.75, ransac_reproj_px=5.0,
                  n_workers=None) -> np.ndarray:
    """M[i, j] = inliers between painting i and database image j (in parallel)."""
    init_args = (db_features, norm, lowe_ratio, ransac_reproj_px)
    with Pool(n_workers or os.cpu_count(), _init_worker, init_args) as pool:
        rows = list(tqdm(pool.imap(_inliers_row, paintings_features),
                         total=len(paintings_features), desc="Matching"))
    return np.array(rows, int).reshape(len(paintings_features), len(db_features))


def inliers_by_query(features_by_query, db_features, norm, lowe_ratio=0.75, ransac_reproj_px=5.0,
                     n_workers=None) -> list[np.ndarray]:
    """Inlier matrix (n_paintings, n_database) of every query."""
    paintings = [painting for query in features_by_query for painting in query]
    inliers = inlier_matrix(paintings, db_features, norm, lowe_ratio, ransac_reproj_px, n_workers)
    query_starts = np.cumsum([len(query) for query in features_by_query])[:-1]
    return np.split(inliers, query_starts)


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve(distances: np.ndarray, db_ids: list[int], k: int = 5) -> list[list[int]]:
    """
    Return the top-k closest database image IDs for each query.

    Args:
        distances: Distance matrix of shape (n_queries, n_database).
            Lower values are considered better matches.
        db_ids: Database image IDs corresponding to the columns
            of the distance matrix.
        k: Number of database images to retrieve per query.

    Returns:
        List of lists containing the top-k database image IDs
        for each query, ordered from smallest to largest distance.

    Raises:
        ValueError: If distances is not 2D, if its number of columns does not
            match db_ids, or if k is not in [1, len(db_ids)].
    """
    if distances.ndim != 2:
        raise ValueError("distances must be a 2D array")

    if distances.shape[1] != len(db_ids):
        raise ValueError(
            "Number of database IDs must match the number of distance columns"
        )

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if k > len(db_ids):
        raise ValueError("k cannot be larger than the database")

    sorted_indices = np.argsort(distances, axis=1)
    top_k_indices = sorted_indices[:, :k]

    results = []

    for query_indices in top_k_indices:
        query_results = [db_ids[i] for i in query_indices]
        results.append(query_results)

    return results

def retrieve_multi(dists: np.ndarray, db_ids: list[int], k: int) -> list[list[list[int]]]:
    """
    Return the top-k closest database image IDs for each painting of each query.

    Args:
        dists: Distance array of shape (n_queries, MAX_PAINTINGS, n_database),
            with NaN in the rows of missing paintings.
        db_ids: Database image IDs corresponding to the last axis of dists.
        k: Number of database images to retrieve per painting.

    Returns:
        One list per query with one top-k list per existing painting, e.g.
        [[[ids painting 1], [ids painting 2]], [[ids painting 1]], ...].
    """
    preds = []
    for q in dists:
        valid = ~np.isnan(q).all(axis=1)
        preds.append(retrieve(q[valid], db_ids, k) if valid.any() else [])
    return preds


UNKNOWN = -1


def rank_painting(painting_inliers, db_ids, k, min_inliers) -> list[int]:
    """Top-k database IDs of a painting, or [UNKNOWN] if its best match has < min_inliers."""
    if painting_inliers.max() < min_inliers:
        return [UNKNOWN]
    return [db_ids[i] for i in np.argsort(-painting_inliers, kind="stable")[:k]]


def retrieve_with_unknowns(inliers_by_query, db_ids, k, min_inliers) -> list[list[list[int]]]:
    """[[[ids painting 1], [ids painting 2]], [[-1]], ...] for every query."""
    return [[rank_painting(painting, db_ids, k, min_inliers) for painting in query]
            for query in inliers_by_query]
