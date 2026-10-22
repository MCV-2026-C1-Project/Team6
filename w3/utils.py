from typing import Literal

import cv2
import numpy as np
from scipy import ndimage as ndi


# =============================================================================
# PAINTING DETECTION (QSD2-W3: up to 2 paintings per image)
# =============================================================================
#
# Pipeline (images are assumed to be already denoised):
#   1. Edge input: the image in the chosen colour space and channels. With
#      use_mahalanobis=1 those channels are turned into the Mahalanobis
#      distance of every pixel to the wall (modelled on the border pixels).
#   2. Edges: high-pass filter (Prewitt by default) on that map, thresholded
#      with Otsu and closed so each frame becomes a closed outline.
#   3. Week 2 post-processing: fill holes (closed outlines become solid) and
#      an opening to remove the edges that do not enclose anything.
#   4. Largest connected components with holes filled (Week 2), extended to
#      keep up to 2 paintings that are big enough and fill their bounding box.
#   5. Week 2 projection rectangle for each painting separately, and ordering
#      left-to-right (or top-to-bottom when stacked).

BORDER_SIZE = 15
MAX_PAINTINGS = 2
PROJ_FRAC = 0.5

EDGE_KERNELS = {
    "prewitt": (
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32),
        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32),
    ),
    "sobel": (
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32),
    ),
}

DEFAULT_PARAMS = {
    # Input of the edge detection
    "color_space": "HSV",
    "channels": [1, 2],
    "use_mahalanobis": 0,  # 1: Mahalanobis distance to the wall, 0: raw channels
    # Edge detection
    "edge_operator": "prewitt",
    "blur_ksize": 5,
    "edge_close_frac": 0.02,
    # Clean-up
    "open_frac": 0.02,
    # Component selection
    "min_area_frac": 0.02,
    "min_rel_area": 0.15,
    "min_extent": 0.6,
    # Shape fitting
    "fit_rect": True,  # Week 2 projection rectangle per painting
    "proj_frac": PROJ_FRAC,
}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def border_mask(h: int, w: int, b: int = BORDER_SIZE) -> np.ndarray:
    """
    Build a boolean mask selecting a frame of pixels along the image border.

    Args:
        h: Image height in pixels.
        w: Image width in pixels.
        b: Border width in pixels.

    Returns:
        Boolean mask of shape (h, w), True on the border pixels.
    """
    b = max(1, min(b, h // 2, w // 2))
    m = np.zeros((h, w), bool)
    m[:b, :] = m[-b:, :] = True
    m[:, :b] = m[:, -b:] = True
    return m


def kernel(mask_shape: tuple[int, int], frac: float) -> np.ndarray | None:
    """
    Square structuring element whose side is a fraction of the smallest side.

    Returns None when the kernel would be smaller than 2 pixels.
    """
    n = int(frac * min(mask_shape))
    if n < 2:
        return None
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


def morph(mask: np.ndarray, op: int, frac: float) -> np.ndarray:
    """Apply a morphological operation with a kernel sized by `frac`."""
    k = kernel(mask.shape, frac)
    if k is None:
        return mask
    return cv2.morphologyEx(mask, op, k)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill the holes of every blob in a binary mask."""
    return ndi.binary_fill_holes(mask > 0).astype(np.uint8)


# -----------------------------------------------------------------------------
# Step 1: input map for the edge detection
# -----------------------------------------------------------------------------
def mahalanobis(color: np.ndarray, border: np.ndarray, channels: list[int]) -> np.ndarray:
    """
    Mahalanobis distance of each pixel to a Gaussian fitted on the border.

    Args:
        color: Image in the chosen color space.
        border: Boolean mask of the pixels used to model the wall.
        channels: Indices of the color channels to use.

    Returns:
        Float32 array of shape (h, w).
    """
    color = color[..., channels].astype(np.float64)
    background = color[border]
    mean = background.mean(axis=0)
    covariance = np.atleast_2d(np.cov(background, rowvar=False))
    covariance += np.eye(len(channels)) * 1e-3
    inverse = np.linalg.pinv(covariance)

    delta = color - mean
    d2 = np.einsum("...i,ij,...j->...", delta, inverse, delta)
    return np.sqrt(np.maximum(d2, 0)).astype(np.float32)


COLOR_SPACES = {"HSV": cv2.COLOR_BGR2HSV, "LAB": cv2.COLOR_BGR2LAB}


def edge_input(
    img_bgr: np.ndarray,
    color_space: Literal["HSV", "LAB"] = "HSV",
    channels: list[int] = (1, 2),
    use_mahalanobis: int = 1,
) -> np.ndarray:
    """
    Map on which the edges are detected.

    Args:
        img_bgr: Input BGR image.
        color_space: Color space the image is converted to.
        channels: Channels of that color space that are used.
        use_mahalanobis: 1 -> Mahalanobis distance of every pixel to the wall
            (modelled on the border pixels), a single-channel map where the
            paintings stand out from the wall. 0 -> the selected channels
            themselves.

    Returns:
        Float32 array of shape (h, w) if use_mahalanobis else (h, w, C).
    """
    color = cv2.cvtColor(img_bgr, COLOR_SPACES[color_space])
    if use_mahalanobis:
        h, w = img_bgr.shape[:2]
        return mahalanobis(color, border_mask(h, w), list(channels))
    return color[..., list(channels)].astype(np.float32)


# -----------------------------------------------------------------------------
# Step 2: edge detection (high-pass filters)
# -----------------------------------------------------------------------------
def gradient_magnitude(
    image: np.ndarray,
    operator: Literal["prewitt", "sobel", "laplacian"] = "prewitt",
    blur_ksize: int = 5,
) -> np.ndarray:
    """
    Gradient magnitude of a single or multi-channel map, normalised to
    [0, 255]. For several channels the maximum response is kept per pixel.

    A Gaussian blur is applied first so that the wall/painting texture does
    not dominate over the frame edges.
    """
    image = image.astype(np.float32)
    if image.ndim == 2:
        image = image[..., None]

    mag = np.zeros(image.shape[:2], np.float32)
    for c in range(image.shape[2]):
        channel = image[..., c]
        if blur_ksize > 1:
            channel = cv2.GaussianBlur(channel, (blur_ksize, blur_ksize), 0)

        if operator == "laplacian":
            channel_mag = np.abs(cv2.Laplacian(channel, cv2.CV_32F, ksize=3))
        else:
            kx, ky = EDGE_KERNELS[operator]
            gx = cv2.filter2D(channel, cv2.CV_32F, kx)
            gy = cv2.filter2D(channel, cv2.CV_32F, ky)
            channel_mag = np.sqrt(gx ** 2 + gy ** 2)
        # Each channel has its own range, so normalise before combining.
        channel_mag /= channel_mag.max() + 1e-6
        mag = np.maximum(mag, channel_mag)

    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def edge_cue(
    image: np.ndarray,
    operator: str = "prewitt",
    blur_ksize: int = 5,
    close_frac: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Binary edge map of the edge input.

    Args:
        image: Map returned by edge_input (Mahalanobis or colour channels).
        operator: High-pass filter used to compute the gradient.
        blur_ksize: Gaussian blur applied before the filter.
        close_frac: Closing kernel size as a fraction of the smallest image
            side. It joins the edges of each frame into a closed outline.

    Returns:
        (gradient magnitude, binary uint8 edge map after Otsu and closing)
    """
    mag = gradient_magnitude(image, operator, blur_ksize)
    _, edges = cv2.threshold(mag, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # The border itself is wall: drop edges touching it (image borders, crops).
    edges[border_mask(*edges.shape, b=2)] = 0
    edges = morph(edges, cv2.MORPH_CLOSE, close_frac)
    return mag, edges


# -----------------------------------------------------------------------------
# Step 3: regions (Week 2 post-processing)
# -----------------------------------------------------------------------------
def clean_mask(edges: np.ndarray, open_frac: float = 0.02) -> np.ndarray:
    """
    Turn closed frame outlines into filled regions.

    Holes are filled, so a closed outline becomes a solid painting. Edges that
    do not enclose anything stay thin and are removed by the opening, which
    also separates two paintings joined by a thin bridge.
    """
    mask = fill_holes(edges)
    mask = morph(mask, cv2.MORPH_OPEN, open_frac)
    return fill_holes(mask)


# -----------------------------------------------------------------------------
# Step 4: largest components (Week 2, extended to 2 paintings)
# -----------------------------------------------------------------------------
def largest_components_filled(
    mask: np.ndarray,
    min_area_frac: float = 0.02,
    min_rel_area: float = 0.15,
    min_extent: float = 0.6,
    max_paintings: int = MAX_PAINTINGS,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Keep the largest connected components and fill their holes.

    Week 2 kept only the largest component. Here every component is a
    candidate and it is kept when:
      - its area is at least `min_area_frac` of the image,
      - its area is at least `min_rel_area` times the largest component
        (a second painting must not be a tiny blob),
      - its extent (area / bounding box area) is at least `min_extent`
        (paintings fill their bounding box, furniture or shadows do not).
    At most `max_paintings` components are kept, the largest first.

    Returns:
        (kept component masks, info about every component)
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return [], []

    image_area = mask.size
    largest = stats[1:, cv2.CC_STAT_AREA].max()

    info = []
    for label in range(1, n):
        x, y, w, h, area = stats[label]
        extent = area / (w * h)
        reasons = []
        if area < min_area_frac * image_area:
            reasons.append("too small")
        if area < min_rel_area * largest:
            reasons.append("small vs largest")
        if extent < min_extent:
            reasons.append("low extent")
        info.append({
            "label": label,
            "area": int(area),
            "extent": float(extent),
            "bbox": [int(x), int(y), int(w), int(h)],
            "kept": not reasons,
            "reasons": reasons,
            "mask": fill_holes(labels == label),
        })

    kept = sorted((c for c in info if c["kept"]), key=lambda c: -c["area"])
    kept = kept[:max_paintings]
    kept_labels = {c["label"] for c in kept}
    for c in info:
        if c["kept"] and c["label"] not in kept_labels:
            c["kept"] = False
            c["reasons"].append(f"more than {max_paintings}")
    return [c["mask"] for c in kept], info


# -----------------------------------------------------------------------------
# Step 5: shape fitting (Week 2) and ordering
# -----------------------------------------------------------------------------
def mask_to_rect(mask: np.ndarray, proj_frac: float = PROJ_FRAC) -> np.ndarray:
    """
    Convert the mask of ONE painting into an axis-aligned rectangle using its
    projections (Week 2 method).

    A column (row) belongs to the painting if its fraction of foreground
    pixels exceeds `proj_frac` times the maximum over all columns (rows).
    This ignores thin protrusions that a plain bounding box would include.
    With two paintings it must be applied to each component separately,
    otherwise the projections of both paintings would be merged.

    Returns:
        Binary uint8 mask with ones inside the rectangle (all zeros if the
        input is empty).
    """
    H, W = mask.shape
    rect = np.zeros((H, W), np.uint8)
    if not mask.any():
        return rect
    col_prof, row_prof = mask.mean(axis=0), mask.mean(axis=1)
    cols = col_prof > proj_frac * col_prof.max()
    rows = row_prof > proj_frac * row_prof.max()
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0, y1 = np.where(rows)[0][[0, -1]]
    rect[y0:y1 + 1, x0:x1 + 1] = 1
    return rect


def order_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    """
    Order paintings left-to-right, or top-to-bottom when they overlap
    horizontally (stacked vertically).
    """
    if len(masks) < 2:
        return masks
    boxes = [cv2.boundingRect(m) for m in masks]
    (x0, _, w0, _), (x1, _, w1, _) = boxes[0], boxes[1]
    overlap_x = min(x0 + w0, x1 + w1) - max(x0, x1)
    axis = 1 if overlap_x > 0.5 * min(w0, w1) else 0  # 1 -> sort by y
    order = sorted(range(len(masks)), key=lambda i: boxes[i][axis])
    return [masks[i] for i in order]


# -----------------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------------
def detect_paintings_debug(img_bgr: np.ndarray, **params) -> dict:
    """
    Run the whole detection pipeline and return every intermediate result,
    so that each step can be visualised (see w3/segmentation.ipynb).

    Args:
        img_bgr: Input BGR image (already denoised).
        **params: Overrides of DEFAULT_PARAMS.

    Returns:
        Dictionary with the intermediate maps, the candidate info and the
        final list of painting masks under "masks".
    """
    p = {**DEFAULT_PARAMS, **params}

    edge_map = edge_input(img_bgr, p["color_space"], p["channels"], p["use_mahalanobis"])
    grad, edges = edge_cue(
        edge_map, p["edge_operator"], p["blur_ksize"], p["edge_close_frac"]
    )
    cleaned = clean_mask(edges, p["open_frac"])
    components, info = largest_components_filled(
        cleaned, p["min_area_frac"], p["min_rel_area"], p["min_extent"]
    )
    if p["fit_rect"]:
        components = [mask_to_rect(c, p["proj_frac"]) for c in components]
    masks = order_masks(components)

    return {
        "params": p,
        "edge_input": edge_map,
        "gradient": grad,
        "edges": edges,
        "cleaned": cleaned,
        "candidates": info,
        "masks": masks,
    }


def detect_paintings(img_bgr: np.ndarray, **params) -> list[np.ndarray]:
    """
    Detect up to two paintings in an image.

    Args:
        img_bgr: Input BGR image (already denoised).
        **params: Overrides of DEFAULT_PARAMS.

    Returns:
        List with 0, 1 or 2 binary uint8 masks (one per painting), ordered
        left-to-right / top-to-bottom, each with the same size as the image.
    """
    return detect_paintings_debug(img_bgr, **params)["masks"]


def combined_mask(masks: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    """Union of the painting masks as a 0/255 uint8 image."""
    out = np.zeros(shape[:2], np.uint8)
    for m in masks:
        out[m > 0] = 255
    return out


def crop_to_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop the image to the bounding box of a mask."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# =============================================================================
# EVALUATION
# =============================================================================
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
