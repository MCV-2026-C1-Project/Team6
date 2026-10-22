from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
import cv2
import scipy.spatial.distance as sd
from typing import Literal
import pywt



# =============================================================================
# TOOLS
# =============================================================================

# We only use .jpg because these are the only ones that contain the required images.
def load_images(folder: str, color_space: int = None) -> tuple[list[int], list[np.ndarray]]:
    """
    Load JPG images from a folder and convert them to CIE Lab.

    Args:
        folder: Path to the folder containing the images.

    Returns:
        Tuple (ids, images) where ids[i] is the ID of images[i].
        Images are stored in Lab color space.
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
# DESCRIPTORS
# =============================================================================

def preprocess(img_bgr, color_space=cv2.COLOR_BGR2HSV, channels=None, size=672):
    """
    Crop the central square of an image, resize it and convert its color space.

    The square side is the minimum of the image height and width.

    Args:
        img_bgr: Input image in BGR format.
        color_space: OpenCV color conversion code (e.g. cv2.COLOR_BGR2HSV).
            If None, the image is kept in BGR.
        channels: Indices of the channels to keep, e.g. (0,) to keep only L in Lab.
            If None, all channels are kept.
        size: Side length of the output square image.

    Returns:
        Float32 array of shape (size, size, C).
    """
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    img = img_bgr[y0:y0 + s, x0:x0 + s]
    interpolation = cv2.INTER_AREA if size < s else cv2.INTER_CUBIC
    img = cv2.resize(img, (size, size), interpolation=interpolation)

    if color_space is not None:
        img = cv2.cvtColor(img, color_space)
    if img.ndim == 2:
        img = img[:, :, None]
    if channels is not None:
        img = img[:, :, channels]
    return img.astype(np.float32)

def compute_descriptors(descriptor, images, color_code=cv2.COLOR_BGR2HSV, channels=[1,2], size=672, **kargs):
    """
    Preprocess a list of images and compute a descriptor for each of them.

    Args:
        descriptor: Descriptor function that takes a preprocessed image.
        images: List of BGR images.
        color_code: OpenCV color conversion code passed to preprocess.
        channels: Indices of the channels to keep, passed to preprocess.
        size: Side length of the preprocessed images.
        **kargs: Extra keyword arguments passed to the descriptor.

    Returns:
        2D NumPy array of shape (n_images, n_features) with one descriptor per row.
    """
    return np.vstack([descriptor(preprocess(im, color_code, channels, size), **kargs) for im in images])

def zigzag_indices(s=8, n=None):
    """
    Return the (row, col) indices of an s x s block in JPEG zig-zag order.

    Args:
        s: Side length of the block.
        n: Number of indices to return. If None, all s * s indices are returned.

    Returns:
        List of (row, col) tuples ordered from low to high frequency.
    """
    idxs = sorted(((i, j) for i in range(s) for j in range(s)),
                  key=lambda p: (p[0] + p[1], p[0] if (p[0] + p[1]) % 2 else p[1]))
    if not n:
        return idxs
    return idxs[:n]

def dct_block_descriptor(img, n_blocks=4, n_coef=16, skip_dc=True, normalize=True):
    """
    Compute a block-based DCT descriptor.

    For each channel, the image is split into n_blocks x n_blocks blocks, the DCT of
    each block is computed and the magnitudes of the first n_coef coefficients in
    zig-zag order are concatenated. The channel vectors are then concatenated.

    Args:
        img: Preprocessed float32 image of shape (S, S, C).
        n_blocks: Number of blocks per side.
        n_coef: Number of DCT coefficients kept per block.
        skip_dc: If True, the DC coefficient is discarded.
        normalize: If True, the final vector is L2-normalized.

    Returns:
        Float32 descriptor of length C * n_blocks**2 * n_coef.
        Recommended distance: euclidean.
    """
    size = img.shape[0]
    B = size // n_blocks
    rows, cols = np.array(zigzag_indices(B, int(skip_dc) + n_coef)[int(skip_dc):]).T

    channel_feats = []
    for c in range(img.shape[2]):
        ch = np.ascontiguousarray(img[:, :, c])
        feats = []
        for by in range(n_blocks):
            for bx in range(n_blocks):
                d = cv2.dct(ch[by * B:(by + 1) * B, bx * B:(bx + 1) * B])
                coeffs = np.abs(d[rows, cols])
                feats.append(coeffs)
        feat = np.concatenate(feats)
        channel_feats.append(feat)
    channel_feat = np.concatenate(channel_feats)
    if normalize:
        channel_feat /= np.linalg.norm(channel_feat) + 1e-8
    return channel_feat.astype(np.float32)


def dct_histogram_descriptor(img, block_size=8, n_coef=12, n_regions=4, n_bins=16,
                             skip_dc=True, value_range=None):
    """
    Compute a DCT coefficient histogram descriptor.

    For each channel, the image is split into small blocks of block_size pixels and
    the DCT of each block is computed. The blocks are grouped into n_regions x n_regions
    regions and, for each region and each of the first n_coef coefficients, a histogram
    of log(1 + |coef|) over all the blocks of the region is computed. All the histograms
    are concatenated.

    Args:
        img: Preprocessed float32 image of shape (S, S, C).
        block_size: Side length of the DCT blocks, in pixels.
        n_coef: Number of DCT coefficients used per block (zig-zag order).
        n_regions: Number of regions per side.
        n_bins: Number of bins of each histogram.
        skip_dc: If True, the DC coefficient is discarded.
        value_range: (lo, hi) range of the histograms. It must be the same for the
            database and the queries. If None, the theoretical maximum is used.

    Returns:
        Float32 descriptor of length C * n_regions**2 * n_coef * n_bins.
        Recommended distances: chi2 or hellinger.
    """
    size = img.shape[0]
    B = block_size
    n_blocks = size // B
    rows, cols = np.array(zigzag_indices(B, int(skip_dc) + n_coef)[int(skip_dc):]).T

    if value_range is None:
        value_range = (0.0, float(np.log1p(255.0 * B)))
    lo, hi = value_range
    bounds = np.linspace(0, n_blocks, n_regions + 1).astype(int)

    feats = []
    for c in range(img.shape[2]):
        ch = np.ascontiguousarray(img[:, :, c])
        coefs = np.empty((n_blocks, n_blocks, n_coef), np.float32)
        for by in range(n_blocks):
            for bx in range(n_blocks):
                d = cv2.dct(ch[by * B:(by + 1) * B, bx * B:(bx + 1) * B])
                coefs[by, bx] = d[rows, cols]
        coefs = np.clip(np.log1p(np.abs(coefs)), lo, hi)

        for ry in range(n_regions):
            for rx in range(n_regions):
                cell = coefs[bounds[ry]:bounds[ry + 1], bounds[rx]:bounds[rx + 1]].reshape(-1, n_coef)
                for k in range(n_coef):
                    hist, _ = np.histogram(cell[:, k], bins=n_bins, range=(lo, hi))
                    feats.append(hist / max(hist.sum(), 1))
    return np.concatenate(feats).astype(np.float32)


def dct_pyramid_descriptor(img, levels=(1, 2, 3, 4), n_coef=10, skip_dc=True, weights=None):
    """
    Compute a spatial pyramid of block-based DCT descriptors.

    The block DCT descriptor is computed at each pyramid level (1x1, 2x2, 3x3, ...),
    each level is multiplied by its weight and the levels are concatenated.

    Args:
        img: Preprocessed float32 image of shape (S, S, C).
        levels: Number of blocks per side at each pyramid level.
        n_coef: Number of DCT coefficients kept per block.
        skip_dc: If True, the DC coefficient is discarded.
        weights: Weight of each level. If None, all levels have weight 1.

    Returns:
        Float32 descriptor of length C * n_coef * sum(n**2 for n in levels).
        Recommended distance: euclidean.
    """
    if weights is None:
        weights = [1.0] * len(levels)
    feats = [w * dct_block_descriptor(img, n_blocks=n, n_coef=n_coef, skip_dc=skip_dc)
             for n, w in zip(levels, weights)]
    return np.concatenate(feats).astype(np.float32)

def dwt_block_descriptor(img, n_blocks=4, wavelet="haar", levels=3, normalize=True):
    """
    Compute a block-based wavelet (DWT) statistics descriptor.

    For each channel, the image is split into n_blocks x n_blocks blocks and a
    multi-level 2D DWT of each block is computed. For each detail subband (H, V, D
    at each level), the mean absolute value and the standard deviation of its
    coefficients are kept. The channel vectors are then concatenated.

    Args:
        img: Preprocessed float32 image of shape (S, S, C).
        n_blocks: Number of blocks per side.
        wavelet: Wavelet name accepted by PyWavelets (e.g. "haar", "db2").
        levels: Number of decomposition levels.
        normalize: If True, the final vector is L2-normalized.

    Returns:
        Float32 descriptor of length C * n_blocks**2 * levels * 3 * 2.
        Recommended distance: euclidean.

    Raises:
        ValueError: If the blocks are too small for the requested number of levels.
    """
    size = img.shape[0]
    B = size // n_blocks
    if levels > pywt.dwt_max_level(B, wavelet):
        raise ValueError(f"Blocks of {B} px are too small for {levels} levels.")

    channel_feats = []
    for c in range(img.shape[2]):
        ch = img[:n_blocks * B, :n_blocks * B, c]
        blocks = ch.reshape(n_blocks, B, n_blocks, B).transpose(0, 2, 1, 3)
        coeffs = pywt.wavedec2(blocks, wavelet, level=levels, axes=(-2, -1))

        feats = []
        for detail in coeffs[1:]:
            for band in detail:
                feats.append(np.abs(band).mean(axis=(-2, -1)))
                feats.append(band.std(axis=(-2, -1)))
        channel_feats.append(np.stack(feats, axis=-1).reshape(-1))

    feat = np.concatenate(channel_feats)
    if normalize:
        feat /= np.linalg.norm(feat) + 1e-8
    return feat.astype(np.float32)


def dwt_histogram_descriptor(img, n_blocks=4, wavelet="haar", levels=3, n_bins=16):
    """
    Compute a block-based wavelet (DWT) coefficient histogram descriptor.

    For each channel, the image is split into n_blocks x n_blocks blocks and a
    multi-level 2D DWT of each block is computed. For each detail subband (H, V, D
    at each level) and each block, a histogram of log(1 + |coef|) is computed.
    All the histograms are concatenated.

    Args:
        img: Preprocessed float32 image of shape (S, S, C).
        n_blocks: Number of blocks per side.
        wavelet: Wavelet name accepted by PyWavelets (e.g. "haar", "db2").
        levels: Number of decomposition levels.
        n_bins: Number of bins of each histogram.

    Returns:
        Float32 descriptor of length C * n_blocks**2 * levels * 3 * n_bins.
        Recommended distances: chi2 or hellinger.

    Raises:
        ValueError: If the blocks are too small for the requested number of levels.
    """
    size = img.shape[0]
    B = size // n_blocks
    if levels > pywt.dwt_max_level(B, wavelet):
        raise ValueError(f"Blocks of {B} px are too small for {levels} levels.")

    feats = []
    for c in range(img.shape[2]):
        ch = img[:n_blocks * B, :n_blocks * B, c]
        blocks = ch.reshape(n_blocks, B, n_blocks, B).transpose(0, 2, 1, 3)
        coeffs = pywt.wavedec2(blocks, wavelet, level=levels, axes=(-2, -1))

        for i, detail in enumerate(coeffs[1:]):
            hi = np.log1p(255.0 * 2 ** (levels - i))
            for band in detail:
                v = np.clip(np.log1p(np.abs(band)), 0, hi)
                for by in range(n_blocks):
                    for bx in range(n_blocks):
                        hist, _ = np.histogram(v[by, bx], bins=n_bins, range=(0, hi))
                        feats.append(hist / hist.sum())
    return np.concatenate(feats).astype(np.float32)


def lbp_block_descriptor(img, n_blocks=4, scales=((8, 1.0),), method="uniform"):
    """
    Compute a block-based Local Binary Pattern (LBP) histogram descriptor.

    For each channel and each scale (P, R), the LBP code of every pixel is computed,
    the image is split into n_blocks x n_blocks blocks and a normalized histogram of
    the codes of each block is computed. All the histograms are concatenated.

    Args:
        img: Preprocessed float32 image of shape (S, S, C) with values in [0, 255].
        n_blocks: Number of blocks per side.
        scales: Sequence of (P, R) pairs: number of neighbors and radius.
        method: LBP method passed to skimage: "default" (2**P codes) or
            "uniform" (P + 2 codes).

    Returns:
        Float32 descriptor of length C * n_blocks**2 * sum(n_codes of each scale).
        Recommended distances: chi2 or hellinger.
    """
    size = img.shape[0]
    B = size // n_blocks
    block_id = np.arange(n_blocks * B) // B
    block_id = block_id[:, None] * n_blocks + block_id[None, :]

    feats = []
    for c in range(img.shape[2]):
        ch = np.clip(img[:n_blocks * B, :n_blocks * B, c], 0, 255).astype(np.uint8)
        for P, R in scales:
            n_codes = 2 ** P if method == "default" else P + 2
            codes = local_binary_pattern(ch, P, R, method).astype(np.int64)
            hist = np.bincount((block_id * n_codes + codes).ravel(),
                               minlength=n_blocks ** 2 * n_codes).reshape(n_blocks ** 2, n_codes)
            feats.append((hist / hist.sum(axis=1, keepdims=True)).ravel())
    return np.concatenate(feats).astype(np.float32)


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
        query_desc: 2D NumPy array of shape (n_queries, n_features) with query descriptors as rows.
        db_desc: 2D NumPy array of shape (n_database, n_features) with database descriptors as rows.
        metric: Distance metric to use.

    Returns:
        Distance matrix of shape (n_queries, n_database).
    """
    other_metrics = {
        "chi2": chi2,
        "hellinger": hellinger,
        "bhattacharyya": bhattacharyya,
        "histogram_intersection": histogram_intersection,
        "total_variation": total_variation,
    }

    if metric in other_metrics:
        return other_metrics[metric](query_desc, db_desc)

    return sd.cdist(query_desc, db_desc, metric)


# =============================================================================
# SEGMENTATION
# =============================================================================

def crop_to_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# =============================================================================
# EVALUATION
# =============================================================================

# APK and MAPK are adapted from https://github.com/benhamner/Metrics.
# The ml_metrics package is no longer compatible with modern versions of setuptools, so the required functions are included here directly.
def apk(actual: list[list[int]], predicted: list[list[int]], k: int = 10) -> float:
    """
    Compute the average precision at k (AP@k) for retrieval results.

    The metric compares the ranked retrieval results against the ground-truth
    items for a query. The order of items in ``actual`` does not matter, while
    the order in ``predicted`` determines the ranking used to compute the
    average precision.

    Args:
        actual: Ground-truth BBDD IDs for each query. Each inner list contains
            the correct BBDD IDs for the corresponding query.
        predicted: Ranked BBDD IDs retrieved for each query. The order of
            elements determines their ranking.
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


def evaluate(actual: list[list[int]], predicted: list[list[int]], k: int):
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
    # The query folder can have fewer images than the ground truth
    # (qsd1_w1 has 24 images for 30 entries), so match them by query id
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


def metrics(pred, gt):
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
