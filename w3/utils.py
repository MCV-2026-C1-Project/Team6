from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
import cv2
import scipy.spatial.distance as sd
from typing import Literal
import pywt
from scipy import ndimage as ndi

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
# DENOISING
# =============================================================================

def denoise_image(image_bgr):
    """Apply a 5x5 mean filter only when impulse-like noise is detected.

    The detector flags an image when more than 1% of its grayscale pixels
    differ by over 30 intensity levels from all eight neighboring pixels.
    Otherwise, return an unchanged copy.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    height, width = gray.shape
    padded = np.pad(gray, 1, mode="edge")
    isolated = np.ones((height, width), dtype=bool)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue

            neighbor = padded[
                1 + dy : 1 + dy + height,
                1 + dx : 1 + dx + width,
            ]
            isolated &= np.abs(gray - neighbor) > 30

    impulse_fraction = float(isolated.mean())

    if impulse_fraction <= 0.01:
        return image_bgr.copy()

    return cv2.blur(image_bgr, (5, 5))

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
        query_desc: (n_queries, n_features) o (n_queries, max_paintings, n_features).
            missing paintings must be filled with NaN.
        db_desc: (n_database, n_features).
        metric: Distance metric to use.

    Returns:
        (n_queries, n_database) si l'input és 2D, o
        (n_queries, max_paintings, n_database) si és 3D, amb NaN a les
        files corresponents a quadres inexistents.
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


# =============================================================================
# PAINTING DETECTION (QSD2-W3: up to 2 paintings per image)
# =============================================================================
#
# Edges-only pipeline (images are assumed to be already denoised), selected in
# w3/edges_no_hysteresis.ipynb:
#   1. Gradient: Prewitt on the HSV S and V channels (no blur), keeping the
#      maximum of the normalised channel responses.
#   2. Edge segmentation: threshold at EDGE_K * Otsu, so that faint frames
#      next to a strong painting are not lost.
#   3. Closing, so that every frame becomes a closed outline.
#   4. Week 2 post-processing: fill holes, opening (separates paintings joined
#      by thin bridges) and fill holes again.
#   5. Connected components: keep up to 2 big components that fill their
#      bounding box.
#   6. Week 2 projection rectangle for each painting, ordered left-to-right
#      (top-to-bottom when stacked).

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
    """Boolean mask of shape (h, w), True on a frame of b pixels along the border."""
    b = max(1, min(b, h // 2, w // 2))
    m = np.zeros((h, w), bool)
    m[:b, :] = m[-b:, :] = True
    m[:, :b] = m[:, -b:] = True
    return m


def square_kernel(frac: float, shape: tuple[int, int]) -> np.ndarray:
    """Square structuring element whose side is a fraction of the smallest side."""
    n = max(2, int(frac * min(shape)))
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill the holes of every blob in a binary mask."""
    return ndi.binary_fill_holes(mask > 0).astype(np.uint8)


def gradient_magnitude(img_bgr: np.ndarray) -> np.ndarray:
    """
    Prewitt gradient magnitude of the HSV S and V channels.

    Each channel response is normalised to [0, 1] and the maximum over the
    channels is kept, so an edge visible in either channel is detected.

    Returns:
        uint8 array of shape (h, w) in [0, 255].
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
    Binary edges: gradient above EDGE_K times its Otsu threshold.

    Otsu alone is dominated by the strongest painting in the image, so a faint
    frame next to it would be lost. The image border is wall and is cleared.
    """
    t, _ = cv2.threshold(grad, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = (grad > EDGE_K * t).astype(np.uint8)
    edges[border_mask(*edges.shape, 2)] = 0
    return edges


def edges_to_regions(edges: np.ndarray) -> np.ndarray:
    """
    Closing + Week 2 post-processing (fill holes, opening, fill holes).

    The closing joins the gaps of each frame outline so that filling the holes
    turns it into a solid painting. The opening removes thin structures and
    separates paintings joined by a thin bridge.
    """
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, square_kernel(CLOSE_FRAC, edges.shape))
    mask = fill_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, square_kernel(OPEN_FRAC, mask.shape))
    return fill_holes(mask)


def painting_components(mask: np.ndarray) -> list[np.ndarray]:
    """
    Keep up to MAX_PAINTINGS connected components, the largest first.

    Week 2 kept only the largest component. Here a component is kept when it
    is at least MIN_AREA_FRAC of the image, at least MIN_REL_AREA of the
    largest component, and fills at least MIN_EXTENT of its bounding box.
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
    Week 2 projection rectangle of ONE painting.

    A column (row) belongs to the painting if its fraction of foreground pixels
    exceeds PROJ_FRAC times the maximum over all columns (rows). With two
    paintings it is applied to each one separately so their projections do not
    mix.
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
    """Order paintings left-to-right, or top-to-bottom when they are stacked."""
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
    """Crop the image to the bounding box of a mask."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def combined_mask(masks: list[np.ndarray], image_shape: tuple[int, ...]) -> np.ndarray:
    """Combine separate painting masks into one binary image mask."""
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


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Pixel-level precision, recall, F1, IoU and error rate of a mask."""
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


# =============================================================================
# VISUALIZATION
# =============================================================================

MASK_COLORS = [(60, 60, 255), (255, 160, 60)]  # BGR: painting 1 red, painting 2 blue


def overlay_masks(img_bgr: np.ndarray, masks: list[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    """Draw each painting mask in its own colour with its order number (BGR)."""
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
    """Save one overlay image per query with its detected painting masks."""
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
    """Save a bar chart with how many queries have 0, 1 and 2 detected paintings."""
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
