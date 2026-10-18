from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.spatial.distance as sd
from typing import Literal
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d


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


def show_histogram(img: np.ndarray) -> None:
    """Display the CIELAB channel histograms.

    Args:
        img: CIELAB image as a NumPy array with three channels corresponding
            to L*, a*, and b*.

    Returns:
        None.
    """
    colors = ("black", "red", "blue")
    channel_names = ("L*", "a*", "b*")

    plt.figure(figsize=(10, 5))

    for i, (name, color) in enumerate(zip(channel_names, colors)):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=name)

    plt.title("CIELAB Color Histograms")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def normalize(hist: np.ndarray) -> np.ndarray:
    """
    Normalize a histogram so its values sum to 1.

    Args:
        hist: Input histogram.

    Returns:
        Normalized histogram.
    """

    return cv2.normalize(hist, None, alpha=1, beta=0, norm_type=cv2.NORM_L1)

def normalize_lab_image(img: np.ndarray, normalize_l: bool = True, normalize_chroma: bool = True,
                        low: float = 1, high: float = 99, chroma_target: float = 40) -> np.ndarray:
    """
    Reduce lighting and saturation differences between images (per-image normalization).

    L* is stretched so its low/high percentiles map to 0 and 255. The chroma (distance of
    (a*, b*) to the neutral point 128) is rescaled so its `high` percentile equals
    chroma_target, keeping the hue of every pixel unchanged.

    Args:
        img: CIELAB image (uint8), as returned by load_images.
        normalize_l: Whether to stretch the lightness channel.
        normalize_chroma: Whether to rescale the chroma.
        low: Lower percentile used for the L* stretch.
        high: Upper percentile used for the L* stretch and chroma scaling.
        chroma_target: Value the `high` percentile of the chroma is scaled to.

    Returns:
        Normalized CIELAB image (uint8) with the same shape as img.
    """
    L, a, b = [c.astype(np.float32) for c in cv2.split(img)]

    if normalize_l:
        lo, hi = np.percentile(L, [low, high])
        L = (L - lo) / max(hi - lo, 1) * 255

    if normalize_chroma:
        a, b = a - 128, b - 128
        scale = chroma_target / max(np.percentile(np.hypot(a, b), high), 1)
        a, b = a * scale + 128, b * scale + 128

    return cv2.merge([np.clip(c, 0, 255).astype(np.uint8) for c in (L, a, b)])


# =============================================================================
# DESCRIPTORS
# =============================================================================

def simple_descriptor(img: np.ndarray) -> np.ndarray:
    """
    Compute a normalized color histogram descriptor for an image.

    Args:
        img: Input image.

    Returns:
        NumPy array containing the normalized histograms
        of all image channels concatenated together.
    """
    hists = []
    for i in range(img.shape[2]):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist = normalize(hist)
        hists.append(hist.flatten())

    return np.concatenate(hists)


def simple_descriptors(images: list[np.ndarray]) -> np.ndarray:
    """
    Compute simple color histogram descriptors for multiple images.

    Args:
        images: List of input images.

    Returns:
        2D NumPy array containing one simple descriptor per input image.
    """
    return np.array([simple_descriptor(img) for img in images])


def complex_descriptor(img: np.ndarray) -> np.ndarray:
    """
    Compute a descriptor for each of the four image quadrants.

    Args:
        img: Input image.
    
    Returns:
        Concatenated descriptors for the four quadrants.
    """
    rows = np.array_split(img, 3, axis = 0)
    regions = [cell for row in rows for cell in np.array_split(row, 3, axis = 1)]

    return np.concatenate([
        simple_descriptor(subimage)
        for subimage in regions
    ])


def complex_descriptors(images: list[np.ndarray]) -> np.ndarray:
    """
    Compute spatial color histogram descriptors for multiple images.

    Args:
        images: List of input images.

    Returns:
        2D NumPy array containing one complex descriptor per input image.
    """
    return np.array([complex_descriptor(img) for img in images])


def color_light_block_descriptor(img: np.ndarray, grid: int = 4, ab_bins: int = 48, l_bins: int = 16,
                                 mask: np.ndarray | None = None) -> np.ndarray:
    """
    Compute a color + lightness histogram for each cell of a grid x grid partition.

    For each cell, a joint 2D a*b* histogram (color) and a 1D L* histogram (light)
    are L1-normalized and weighted 2/3 and 1/3, so color accounts for 2/3 of the
    information and light for 1/3. The cell descriptors are concatenated and
    divided by the number of cells, so the result sums to 1.

    The default parameters are the best configuration found in grid_search.py.

    Args:
        img: CIELAB image (uint8), as returned by load_images.
        grid: Number of rows and columns of the partition.
        ab_bins: Bins per color channel (a*, b*); each cell has ab_bins**2 color values.
        l_bins: Bins for the L* channel.
        mask: Optional uint8 mask with the same height and width as img (non-zero = painting).
            Only the masked pixels are counted. Cells without masked pixels are all zeros.

    Returns:
        1D descriptor of length grid**2 * (ab_bins**2 + l_bins).
    """
    rows = np.array_split(img, grid, axis=0)
    regions = [cell for row in rows for cell in np.array_split(row, grid, axis=1)]

    if mask is None:
        cell_masks = [None] * len(regions)
    else:
        mask_rows = np.array_split(mask.astype(np.uint8), grid, axis=0)
        cell_masks = [cell for row in mask_rows for cell in np.array_split(row, grid, axis=1)]

    cell_descriptors = []
    for cell, cell_mask in zip(regions, cell_masks):
        color = cv2.calcHist([cell], [1, 2], cell_mask, [ab_bins, ab_bins], [0, 256, 0, 256])
        light = cv2.calcHist([cell], [0], cell_mask, [l_bins], [0, 256])
        cell_descriptors.append(normalize(color).flatten() * 2/3)
        cell_descriptors.append(normalize(light).flatten() * 1/3)

    return np.concatenate(cell_descriptors) / grid**2


def color_light_block_descriptors(images: list[np.ndarray], grid: int = 4, ab_bins: int = 48, l_bins: int = 16,
                                  masks: list[np.ndarray] | None = None) -> np.ndarray:
    """
    Compute color + lightness block descriptors for multiple images.

    Args:
        images: List of BGR images, as returned by load_images.
        grid: Number of rows and columns of the partition.
        ab_bins: Bins per color channel (a*, b*).
        l_bins: Bins for the L* channel.
        masks: Optional list of masks, one per image (see color_light_block_descriptor).

    Returns:
        2D NumPy array containing one descriptor per input image.
    """
    if masks is None:
        masks = [None] * len(images)

    return np.array([
        color_light_block_descriptor(
            cv2.cvtColor(img, cv2.COLOR_BGR2LAB),
            grid,
            ab_bins,
            l_bins,
            mask)
        for img, mask in zip(images, masks)
    ])


def histogram_3d(image: np.ndarray, bins: int = 8) -> np.ndarray:
    """
    Compute a normalized joint 3D histogram from a Lab image.

    Args:
        image: Non-empty uint8 Lab image of shape (height, width, 3).
            Uses OpenCV's 8-bit encoding, with all channels in [0, 255].
        bins: Number of bins per channel. Must be a positive integer.

    Returns:
        A flattened histogram containing bins ** 3 values.
        The values sum to 1 and represent pixel proportions.
    """
    if image.dtype != np.uint8:
        raise ValueError("Expected a uint8 image.")

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a non-empty three-channel image.")

    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer.")

    # Count L, a and b combinations together.
    histogram = cv2.calcHist(
        [image],
        [0, 1, 2],
        None,
        [bins, bins, bins],
        [0, 256, 0, 256, 0, 256],
    )

    # Convert counts into proportions.
    histogram /= histogram.sum()

    return histogram.flatten()

def spatial_descriptor_3d(
    image: np.ndarray,
    bins: tuple[int, int, int] = (4, 8, 8),
    grids: tuple[int, ...] = (3,),
    weights: tuple[float, ...] | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute a global, block-based or hierarchical 3D Lab descriptor.

    Args:
        image: Non-empty uint8 OpenCV Lab image of shape (H, W, 3).
        bins: Positive bin counts for L, a and b respectively.
        grids: Grid widths. (4,) means 4x4 blocks; (1, 2, 4)
            means a three-level spatial pyramid.
        weights: Non-negative weights for the grid levels.
            None gives equal weight to each level.
        mask: Optional binary foreground mask of shape (H, W).
            Non-zero pixels are included; zero pixels are ignored.

    Returns:
        A flattened float32 descriptor with
        product(bins) * sum(grid ** 2 for grid in grids) values.
        Non-empty cells have equal weight within each level.
        The final descriptor sums to 1.

    Raises:
        ValueError: If the image, settings or mask are invalid,
            or the mask contains no foreground pixels.
    """
    if image.dtype != np.uint8:
        raise ValueError("Expected a uint8 Lab image.")

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a non-empty three-channel image.")

    if len(bins) != 3 or any(
        not isinstance(b, int) or b < 1 for b in bins
    ):
        raise ValueError("Provide three positive integer bin counts.")

    if not grids or any(
        not isinstance(g, int) or g < 1
        or g > min(image.shape[:2]) for g in grids
    ):
        raise ValueError("Grid sizes must be positive and fit the image.")

    level_weights = (
        np.ones(len(grids), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )

    if (
        level_weights.shape != (len(grids),)
        or not np.all(np.isfinite(level_weights))
        or np.any(level_weights < 0)
        or level_weights.sum() <= 0
    ):
        raise ValueError(
            "Provide one finite, non-negative weight per grid, "
            "with at least one positive weight."
        )

    level_weights /= level_weights.sum()

    if mask is not None:
        mask = np.asarray(mask)

        if mask.shape != image.shape[:2]:
            raise ValueError("Mask shape must match image height and width.")

        if not np.all(np.isfinite(mask)):
            raise ValueError("Mask must contain only finite values.")

        mask = np.ascontiguousarray(
            (mask != 0).astype(np.uint8) * 255
        )

        if not np.any(mask):
            raise ValueError("Mask contains no foreground pixels.")

    levels = []

    for grid, weight in zip(grids, level_weights):
        cells = []

        image_rows = np.array_split(image, grid, axis=0)
        mask_rows = (
            np.array_split(mask, grid, axis=0)
            if mask is not None
            else [None] * grid
        )

        for image_row, mask_row in zip(image_rows, mask_rows):
            image_blocks = np.array_split(image_row, grid, axis=1)
            mask_blocks = (
                np.array_split(mask_row, grid, axis=1)
                if mask_row is not None
                else [None] * grid
            )

            for block, block_mask in zip(image_blocks, mask_blocks):
                histogram = cv2.calcHist(
                    [np.ascontiguousarray(block)],
                    [0, 1, 2],
                    (
                        np.ascontiguousarray(block_mask)
                        if block_mask is not None
                        else None
                    ),
                    list(bins),
                    [0, 256, 0, 256, 0, 256],
                ).flatten()

                # Empty foreground cells remain zero.
                count = histogram.sum()
                if count > 0:
                    histogram /= count

                cells.append(histogram)

        level = np.concatenate(cells)
        level /= level.sum()
        levels.append(level * weight)

    descriptor = np.concatenate(levels).astype(np.float32)
    descriptor /= descriptor.sum()

    return descriptor

def spatial_descriptors_3d(
    images: list[np.ndarray],
    bins: tuple[int, int, int] = (4, 8, 8),
    grids: tuple[int, ...] = (3,),
    weights: tuple[float, ...] | None = None,
    masks: list[np.ndarray] | None = None,
) -> np.ndarray:
    """
    Compute 3D spatial descriptors for multiple images.

    Args:
        images: List of BGR images.
        bins: Number of bins for the L*, a*, and b* channels.
        grids: Grid sizes for each spatial level.
        weights: Optional weights for each grid level.
        masks: Optional list of masks, one per image.

    Returns:
        2D NumPy array containing one descriptor per input image.
    """
    if masks is None:
        masks = [None] * len(images)

    return np.array([
        spatial_descriptor_3d(
            cv2.cvtColor(image, cv2.COLOR_BGR2LAB),
            bins=bins,
            grids=grids,
            weights=weights,
            mask=mask
        )
        for image, mask in zip(images, masks)
    ])


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

BORDER_SIZE = 15
MEDIAN_KSIZE = 5
WALLTHR_K = 6
GMM_SAMPLE_SIZE = 14
GMM_SMOOTH_SIGMA = 2
PROJ_FRAC = 0.5


def prefilter_image(bgr: np.ndarray, name: Literal["none", "median"]) -> np.ndarray:
    """
    Apply an optional denoising filter to an image.

    Args:
        bgr: Input image in BGR color space.
        name: Filter to apply. "none" returns the image unchanged and
            "median" applies a median blur of size MEDIAN_KSIZE.

    Returns:
        Filtered image with the same shape as the input.
    """
    if name == "none":
        return bgr
    if name == "median":
        return cv2.medianBlur(bgr, MEDIAN_KSIZE)
    raise ValueError(f"Unknown prefilter: {name}")


def border_mask(h: int, w: int, b: int = BORDER_SIZE) -> np.ndarray:
    """
    Build a mask selecting a frame of pixels along the image border.

    The border is assumed to contain only wall, so it is used to model
    the background color distribution.

    Args:
        h: Image height in pixels.
        w: Image width in pixels.
        b: Border width in pixels. It is clipped so that it never exceeds
            half of the image height or width.

    Returns:
        Boolean mask of shape (h, w), True on the border pixels.
    """
    b = max(1, min(b, h // 2, w // 2))
    m = np.zeros((h, w), bool)
    m[:b, :] = m[-b:, :] = True
    m[:, :b] = m[:, -b:] = True
    return m


def mahalanobis(color: np.ndarray, border: np.ndarray, channels: list[int]) -> np.ndarray:
    """
    Compute the Mahalanobis distance of each pixel to the background.

    The background is modeled as a Gaussian fitted to the border pixels,
    so pixels that look like the wall get small distances and pixels
    from the painting get large ones.

    Args:
        color: Image in the chosen color space (HSV or Lab).
        border: Boolean mask of the pixels used to model the background.
        channels: Indices of the color channels to use.

    Returns:
        Float32 array of shape (h, w) with the distance of each pixel.
    """
    color = color[..., channels].astype(np.float64)
    background_pixels = color[border]
    mean = background_pixels.mean(axis=0)

    covariance = np.atleast_2d(np.cov(background_pixels, rowvar=False))
    covariance += np.eye(len(channels)) * 1e-3
    inverse_covariance = np.linalg.pinv(covariance)

    delta = color - mean
    distance_squared = np.einsum(
        "...i,ij,...j->...",
        delta,
        inverse_covariance,
        delta,
    )
    distance = np.sqrt(np.maximum(distance_squared, 0)).astype(np.float32)
    return distance


def threshold_value(
    distance: np.ndarray,
    border: np.ndarray,
    method: Literal["otsu", "triangle", "gmm", "wallthr"],
) -> float:
    """
    Compute the threshold that separates the painting from the wall.

    Pixels with a distance greater than the threshold are foreground.
    "wallthr" uses a robust statistic of the border distances (median +
    WALLTHR_K * MAD). "otsu", "triangle" and "gmm" work on the distance
    clipped to [0, 255]; "gmm" fits two Gaussians and takes the histogram
    minimum between their means.

    Args:
        distance: Distance map returned by mahalanobis.
        border: Boolean mask of the border pixels.
        method: Thresholding method to use.

    Returns:
        Threshold value in distance units.
    """
    if method == "wallthr":
        wall_d = distance[border].astype(np.float64)
        med = np.median(wall_d)
        return med + WALLTHR_K * 1.4826 * np.median(np.abs(wall_d - med))

    src = np.clip(distance, 0, 255).astype(np.uint8)

    if method == "otsu":
        t, _ = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return t
    if method == "triangle":
        t, _ = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return t
    if method == "gmm":
        from sklearn.mixture import GaussianMixture
        sample = cv2.resize(src, (GMM_SAMPLE_SIZE, GMM_SAMPLE_SIZE),
                            interpolation=cv2.INTER_CUBIC).reshape(-1, 1).astype(np.float64)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(sample)
        m1, m2 = np.sort(gmm.means_.flatten())
        hist, bins = np.histogram(src, bins=256, range=(0, 256))
        hist_smooth = gaussian_filter1d(hist.astype(np.float64), sigma=GMM_SMOOTH_SIGMA)
        centers = (bins[:-1] + bins[1:]) / 2
        between = (centers > m1) & (centers < m2)
        if not between.any():   # the two gaussians are too close
            return (m1 + m2) / 2
        return centers[between][np.argmin(hist_smooth[between])]
    raise ValueError(f"Unknown threshold: {method}")


def close(mask: np.ndarray, frac: float) -> np.ndarray:
    """
    Apply a morphological closing to fill small gaps in a binary mask.

    Args:
        mask: Binary mask as a uint8 array.
        frac: Size of the square kernel as a fraction of the smallest
            image side. Kernels smaller than 2 pixels are skipped.

    Returns:
        Closed binary mask with the same shape as the input.
    """
    h, w = mask.shape
    n = int(frac * min(h, w))
    if n < 2:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def largest_component_filled(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component and fill its holes.

    Args:
        mask: Binary mask as a uint8 array.

    Returns:
        Binary uint8 mask containing only the largest component, with its
        holes filled. All zeros if the input has no foreground.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return np.zeros_like(mask)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return ndi.binary_fill_holes(labels == largest).astype(np.uint8)


def mask_to_rect(mask: np.ndarray) -> np.ndarray:
    """
    Convert a mask into an axis-aligned rectangle using its projections.

    A column (row) belongs to the painting if its fraction of foreground
    pixels exceeds PROJ_FRAC times the maximum over all columns (rows).
    This ignores thin protrusions that a plain bounding box would include.

    Args:
        mask: Binary mask as a uint8 array.

    Returns:
        Binary uint8 mask of the same shape, with ones inside the rectangle.
        All zeros if the input has no foreground.
    """
    H, W = mask.shape
    mask_rect = np.zeros((H, W), np.uint8)
    if not mask.any():
        return mask_rect
    col_prof, row_prof = mask.mean(axis=0), mask.mean(axis=1)
    cols = col_prof > PROJ_FRAC * col_prof.max()
    rows = row_prof > PROJ_FRAC * row_prof.max()
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0, y1 = np.where(rows)[0][[0, -1]]
    mask_rect[y0:y1 + 1, x0:x1 + 1] = 1
    return mask_rect

# Pipeline
def segment(
    img_bgr: np.ndarray,
    prefilter: Literal["none", "median"],
    color_space: Literal["HSV", "LAB"],
    channels: list[int],
    threshold: Literal["otsu", "triangle", "gmm", "wallthr"],
    close_frac: float,
) -> np.ndarray:
    """
    Segment the painting hanging on a wall in an image.

    The wall is modeled from the image border, each pixel's Mahalanobis
    distance to it is thresholded, and the result is cleaned up and
    converted into a rectangle.

    Args:
        img_bgr: Input image in BGR color space, as returned by cv2.imread.
        prefilter: Denoising filter applied before the color conversion.
        color_space: Color space used to compute the distance.
        channels: Indices of the color channels to use
            (HSV: 0=H, 1=S, 2=V; Lab: 0=L, 1=a, 2=b).
        threshold: Thresholding method applied to the distance map.
        close_frac: Closing kernel size as a fraction of the smallest
            image side.

    Returns:
        Binary uint8 mask of shape (h, w), with ones on the painting.
    """
    COLOR_SPACES = {"HSV": cv2.COLOR_BGR2HSV, "LAB": cv2.COLOR_BGR2LAB}

    h, w = img_bgr.shape[:2]
    border = border_mask(h, w)
    color = cv2.cvtColor(prefilter_image(img_bgr, prefilter), COLOR_SPACES[color_space])
    dist = mahalanobis(color, border, channels)
    t = threshold_value(dist, border, threshold)
    mask = (dist > t).astype(np.uint8)
    mask = close(mask, close_frac)
    mask = largest_component_filled(mask)
    return mask_to_rect(mask)


def segments(
    images: list[np.ndarray],
    prefilter: Literal["none", "median"],
    color_space: Literal["HSV", "LAB"],
    channels: list[int],
    threshold: Literal["otsu", "triangle", "gmm", "wallthr"],
    close_frac: float,
) -> np.ndarray:
    """
    Apply segment() to a list of images.

    Args:
        images: Input images to segment.
        Other arguments: Passed directly to segment().

    Returns:
        Array of binary masks, one for each image.
    """
    return [
        segment(img, prefilter, color_space, channels, threshold, close_frac)
        for img in images
    ]


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
