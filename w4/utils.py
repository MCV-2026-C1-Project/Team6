from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
import cv2
import scipy.spatial.distance as sd
from typing import Literal
import pywt
from scipy import ndimage as ndi
from tqdm import tqdm

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
# DISTANCES
# =============================================================================

def chi2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Chi-squared distance between descriptors.

    Args:
        x: 2D NumPy array of descriptors for the first set of images.
        y: 2D NumPy array of descriptors for the second set of images.

    Returns:
        Distance matrix of shape (len(x), len(y)).
    """
    num = (x[:, None, :] - y[None, :, :]) ** 2
    den = x[:, None, :] + y[None, :, :]
    return 0.5 * np.sum(np.divide(num, den, out=np.zeros_like(num), where=den > 0), axis=2)


def hellinger(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Hellinger distance between descriptors.

    Args:
        x: 2D NumPy array of descriptors for the first set of images.
        y: 2D NumPy array of descriptors for the second set of images.

    Returns:
        Distance matrix of shape (len(x), len(y)).
    """
    return np.linalg.norm(
        np.sqrt(x[:, None, :]) - np.sqrt(y[None, :, :]),
        axis=2
    ) / np.sqrt(2.0)


def bhattacharyya(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Bhattacharyya distance between descriptors.

    Args:
        x: 2D NumPy array of descriptors for the first set of images.
        y: 2D NumPy array of descriptors for the second set of images.

    Returns:
        Distance matrix of shape (len(x), len(y)).
    """
    coefficient = np.sum(
        np.sqrt(x[:, None, :] * y[None, :, :]),
        axis=2
    )
    return -np.log(np.clip(coefficient, np.finfo(float).tiny, 1.0))


def histogram_intersection(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise histogram intersection distance.

    Args:
        x: 2D NumPy array of descriptors for the first set of images.
        y: 2D NumPy array of descriptors for the second set of images.

    Returns:
        Distance matrix of shape (len(x), len(y)).
    """
    x = x / x.sum(axis=1, keepdims=True)
    y = y / y.sum(axis=1, keepdims=True)
    return 1.0 - np.sum(
        np.minimum(x[:, None, :], y[None, :, :]),
        axis=2
    )


def total_variation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise total variation distance between descriptors.

    Args:
        x: 2D NumPy array of descriptors for the first set of images.
        y: 2D NumPy array of descriptors for the second set of images.

    Returns:
        Distance matrix of shape (len(x), len(y)).
    """
    return 0.5 * np.sum(
        np.abs(x[:, None, :] - y[None, :, :]),
        axis=2
    )


DISTANCES = ["bhattacharyya", 'braycurtis', 'canberra', 'chebyshev', "chi2", 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', "hellinger", "histogram_intersection", 'jaccard', 'jensenshannon', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalsneath', 'sqeuclidean', "total_variation", 'yule']


def get_distances(
        query_desc: np.ndarray,
        db_desc: np.ndarray,
        metric: Literal[
            "bhattacharyya", 'braycurtis', 'canberra', 'chebyshev', "chi2", 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', "hellinger", "histogram_intersection", 'jaccard', 'jensenshannon', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalsneath', 'sqeuclidean', "total_variation", 'yule'
        ] = 'euclidean'
    ) -> np.ndarray:
    """
    Compute pairwise distances between query and database descriptors.

    Args:
        query_desc: Array of shape (n_queries, n_features) or
            (n_queries, max_paintings, n_features). Missing paintings must be
            filled with NaN.
        db_desc: 2D NumPy array of shape (n_database, n_features).
        metric: Distance metric to use. Custom metrics are used for chi2,
            hellinger, bhattacharyya, histogram_intersection and
            total_variation; the rest are computed with scipy cdist.

    Returns:
        Distance matrix of shape (n_queries, n_database) if the input is 2D, or
        (n_queries, max_paintings, n_database) if it is 3D, with NaN in the rows
        of missing paintings.
    """
    other_metrics = {
        "chi2": chi2,
        "hellinger": hellinger,
        "bhattacharyya": bhattacharyya,
        "histogram_intersection": histogram_intersection,
        "total_variation": total_variation,
    }

    query_desc = np.asarray(query_desc, dtype=float)
    is_3d = query_desc.ndim == 3
    n_feat = query_desc.shape[-1]

    flat = query_desc.reshape(-1, n_feat)
    valid = ~np.isnan(flat).any(axis=1)

    dists = np.full((flat.shape[0], db_desc.shape[0]), np.nan)
    if valid.any():
        if metric in other_metrics:
            dists[valid] = other_metrics[metric](flat[valid], db_desc)
        else:
            dists[valid] = sd.cdist(flat[valid], db_desc, metric)

    return dists.reshape(*query_desc.shape[:-1], -1) if is_3d else dists


# =============================================================================
# SEGMENTATION
# =============================================================================

EDGE_CHANNELS = [1, 2]   # HSV channels used for the gradient (S, V)
EDGE_K = 0.5             # edge threshold = EDGE_K * Otsu threshold
CLOSE_FRAC = 0.02        # closing kernel, fraction of the smallest image side
OPEN_FRAC = 0.02         # opening kernel, fraction of the smallest image side
MAX_PAINTINGS = 2
MIN_AREA_FRAC = 0.02     # min component area, fraction of the image
MIN_REL_AREA = 0.15      # min component area, fraction of the largest component
MIN_EXTENT = 0.6         # min component area / bounding-box area
PROJ_FRAC = 0.7          # row/column kept if > PROJ_FRAC * max projection

PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)


def border_mask(h: int, w: int, b: int) -> np.ndarray:
    """
    Build a mask of a frame along the image border.

    Args:
        h: Height of the mask.
        w: Width of the mask.
        b: Width of the frame in pixels, clipped to [1, min(h, w) // 2].

    Returns:
        Boolean mask of shape (h, w), True on the frame.
    """
    b = max(1, min(b, h // 2, w // 2))
    m = np.zeros((h, w), bool)
    m[:b, :] = m[-b:, :] = True
    m[:, :b] = m[:, -b:] = True
    return m


def square_kernel(frac: float, shape: tuple[int, int]) -> np.ndarray:
    """
    Build a square structuring element proportional to the image size.

    Args:
        frac: Side of the kernel as a fraction of the smallest side of the image.
        shape: Shape (h, w) of the image.

    Returns:
        Rectangular structuring element of side max(2, int(frac * min(shape))).
    """
    n = max(2, int(frac * min(shape)))
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill the holes of every blob in a binary mask.

    Args:
        mask: Binary mask (any non-zero value is foreground).

    Returns:
        uint8 mask of the same shape with the holes filled (1 inside).
    """
    return ndi.binary_fill_holes(mask > 0).astype(np.uint8)


def gradient_magnitude(img_bgr: np.ndarray) -> np.ndarray:
    """
    Compute the Prewitt gradient magnitude of the HSV S and V channels.

    Each channel response is normalised to [0, 1] and the maximum over the
    channels is kept, so an edge visible in either channel is detected.

    Args:
        img_bgr: Input image in BGR format.

    Returns:
        uint8 array of shape (h, w) with values in [0, 255].
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mag = np.zeros(hsv.shape[:2], np.float32)
    for c in EDGE_CHANNELS:
        channel = hsv[..., c].astype(np.float32)
        gx = cv2.filter2D(channel, cv2.CV_32F, PREWITT_X)
        gy = cv2.filter2D(channel, cv2.CV_32F, PREWITT_Y)
        g = np.sqrt(gx ** 2 + gy ** 2)
        mag = np.maximum(mag, g / (g.max() + 1e-6))
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def edge_segmentation(grad: np.ndarray) -> np.ndarray:
    """
    Binarize a gradient image with EDGE_K times its Otsu threshold.

    Otsu alone is dominated by the strongest painting in the image, so a faint
    frame next to it would be lost. The image border is wall and is cleared.

    Args:
        grad: uint8 gradient magnitude of shape (h, w).

    Returns:
        uint8 binary edge map of shape (h, w), 1 on the edges.
    """
    t, _ = cv2.threshold(grad, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = (grad > EDGE_K * t).astype(np.uint8)
    edges[border_mask(*edges.shape, 2)] = 0
    return edges


def edges_to_regions(edges: np.ndarray) -> np.ndarray:
    """
    Turn an edge map into solid regions with closing and Week 2 post-processing.

    The closing joins the gaps of each frame outline so that filling the holes
    turns it into a solid painting. Then holes are filled, an opening removes
    thin structures and separates paintings joined by a thin bridge, and holes
    are filled again.

    Args:
        edges: uint8 binary edge map of shape (h, w).

    Returns:
        uint8 binary mask of shape (h, w), 1 inside the regions.
    """
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, square_kernel(CLOSE_FRAC, edges.shape))
    mask = fill_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, square_kernel(OPEN_FRAC, mask.shape))
    return fill_holes(mask)


def painting_components(mask: np.ndarray) -> list[np.ndarray]:
    """
    Select up to MAX_PAINTINGS connected components as painting candidates.

    A component is kept when it is at least MIN_AREA_FRAC of the image, at
    least MIN_REL_AREA of the largest component, and fills at least MIN_EXTENT
    of its bounding box. Week 2 kept only the largest component.

    Args:
        mask: uint8 binary region mask of shape (h, w).

    Returns:
        List of uint8 masks of shape (h, w), one per kept component, ordered
        from largest to smallest area.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return []
    largest = stats[1:, cv2.CC_STAT_AREA].max()

    kept = []
    for label in range(1, n):
        _, _, w, h, area = stats[label]
        if (area >= MIN_AREA_FRAC * mask.size
                and area >= MIN_REL_AREA * largest
                and area / (w * h) >= MIN_EXTENT):
            kept.append((area, label))

    kept = sorted(kept, reverse=True)[:MAX_PAINTINGS]
    return [fill_holes(labels == label) for _, label in kept]


def mask_to_rect(mask: np.ndarray) -> np.ndarray:
    """
    Compute the Week 2 projection rectangle of one painting.

    A column (row) belongs to the painting if its fraction of foreground pixels
    exceeds PROJ_FRAC times the maximum over all columns (rows). With two
    paintings it is applied to each one separately so their projections do not
    mix.

    Args:
        mask: Binary mask of one painting, shape (h, w).

    Returns:
        uint8 mask of shape (h, w), 1 inside the rectangle. All zeros if the
        input mask is empty.
    """
    rect = np.zeros(mask.shape, np.uint8)
    if not mask.any():
        return rect
    col_prof, row_prof = mask.mean(axis=0), mask.mean(axis=1)
    cols = col_prof > PROJ_FRAC * col_prof.max()
    rows = row_prof > PROJ_FRAC * row_prof.max()
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0, y1 = np.where(rows)[0][[0, -1]]
    rect[y0:y1 + 1, x0:x1 + 1] = 1
    return rect


def order_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    """
    Order painting masks left-to-right, or top-to-bottom when they are stacked.

    Two paintings are considered stacked when their horizontal overlap is more
    than half the width of the narrower one.

    Args:
        masks: List of binary masks (at most two are compared).

    Returns:
        The same masks in reading order.
    """
    if len(masks) < 2:
        return masks
    boxes = [cv2.boundingRect(m) for m in masks]
    (x0, _, w0, _), (x1, _, w1, _) = boxes[0], boxes[1]
    overlap_x = min(x0 + w0, x1 + w1) - max(x0, x1)
    axis = 1 if overlap_x > 0.5 * min(w0, w1) else 0  # 1 -> sort by y
    return [masks[i] for i in sorted(range(len(masks)), key=lambda i: boxes[i][axis])]


def detect_paintings(img_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Detect up to two paintings in an image.

    Args:
        img_bgr: Input BGR image (already denoised).

    Returns:
        List with 0, 1 or 2 binary uint8 masks (one per painting, 1 inside),
        ordered left-to-right / top-to-bottom, each the size of the image.
    """
    edges = edge_segmentation(gradient_magnitude(img_bgr))
    regions = edges_to_regions(edges)
    return order_masks([mask_to_rect(c) for c in painting_components(regions)])


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


def combined_mask(masks: list[np.ndarray], image_shape: tuple[int, ...]) -> np.ndarray:
    """
    Combine separate painting masks into one binary image mask.

    Args:
        masks: List of binary masks, each of shape image_shape[:2].
        image_shape: Shape of the image the masks belong to.

    Returns:
        uint8 mask of shape image_shape[:2], 255 inside any painting.

    Raises:
        ValueError: If a mask does not match the image dimensions.
    """
    result = np.zeros(image_shape[:2], dtype=np.uint8)
    for mask in masks:
        if np.asarray(mask).shape != result.shape:
            raise ValueError("A painting mask does not match its image dimensions.")
        result[np.asarray(mask) > 0] = 255
    return result


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

MASK_COLORS = [(60, 60, 255), (255, 160, 60)]  # BGR: painting 1 red, painting 2 blue


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
