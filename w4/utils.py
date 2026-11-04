from pathlib import Path
import numpy as np
import cv2

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