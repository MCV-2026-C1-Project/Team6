from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
import cv2
import scipy.spatial.distance as sd
from typing import Literal
import pywt
from tqdm import tqdm

from segmentation import MAX_PAINTINGS  # used by visualize_painting_count

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
# SEGMENTATION
# =============================================================================
# Painting detection is in segmentation.py (detect_paintings, up to
# MAX_PAINTINGS paintings). The Week 3 version is still in w3/utils.py.

def crop_to_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crop an image to the bounding box of a mask.

    Args:
        img: Input image of shape (h, w) or (h, w, C).
        mask: Binary mask of shape (h, w).

    Returns:
        Cropped image, or the whole image if the mask is empty.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


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

# =============================================================================
# VISUALIZATION
# =============================================================================

MASK_COLORS = [(60, 60, 255), (255, 160, 60), (0, 200, 255)]  # BGR: painting 1 red, 2 blue, 3 yellow


def overlay_masks(img_bgr: np.ndarray, masks: list[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    """
    Draw each painting mask in its own colour with its order number.

    Args:
        img_bgr: Input image in BGR format.
        masks: List of binary masks, one per painting, in detection order.
        alpha: Opacity of the colour overlay, in [0, 1].

    Returns:
        BGR image with the masks, their contours and their numbers drawn.
    """
    out = img_bgr.copy()
    for k, mask in enumerate(masks):
        mask = (np.asarray(mask) > 0).astype(np.uint8)
        color = np.array(MASK_COLORS[k % len(MASK_COLORS)])
        out[mask > 0] = ((1 - alpha) * out[mask > 0] + alpha * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color.tolist(), 3)
        x, y, w, h = cv2.boundingRect(mask)
        cv2.putText(out, str(k + 1), (x + w // 2, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
    return out


def visualize_masks(
    query_images: list[np.ndarray],
    masks_by_query: list[list[np.ndarray]],
    output_dir: Path,
) -> None:
    """
    Save one overlay image per query with its detected painting masks.

    Args:
        query_images: Query images in BGR format.
        masks_by_query: For each query, the list of its painting masks.
        output_dir: Folder where the images are saved as 00000.png, 00001.png, ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (img, masks) in enumerate(zip(query_images, masks_by_query)):
        cv2.imwrite(str(output_dir / f"{idx:05d}.png"), overlay_masks(img, masks))
    print(f"Mask visualizations saved under: {output_dir}")


def visualize_painting_count(
    query_images: list[np.ndarray],
    masks_by_query: list[list[np.ndarray]],
    output_dir: Path,
) -> None:
    """
    Save a bar chart with how many queries have 0, 1, ... MAX_PAINTINGS paintings.

    Args:
        query_images: Query images (only their number is used).
        masks_by_query: For each query, the list of its painting masks.
        output_dir: Folder where the chart is saved as painting_count.png.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = [len(masks) for masks in masks_by_query]
    values = list(range(MAX_PAINTINGS + 1))
    totals = [counts.count(v) for v in values]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([str(v) for v in values], totals, color="#4c72b0")
    ax.bar_label(bars)
    ax.set_xlabel("Detected paintings")
    ax.set_ylabel("Number of query images")
    ax.set_title(f"Paintings detected per image ({len(query_images)} images)")
    fig.tight_layout()
    fig.savefig(output_dir / "painting_count.png", dpi=120)
    plt.close(fig)
    print(f"Painting count chart saved under: {output_dir}")
