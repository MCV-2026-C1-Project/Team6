from typing import Literal

import cv2
import numpy as np
from scipy import ndimage as ndi


# =============================================================================
# PAINTING DETECTION (QSD2-W3: up to 2 paintings per image)
# =============================================================================
#
# Pipeline (images are assumed to be already denoised):
#   1. Colour cue: Mahalanobis distance of every pixel to the wall, modelled
#      from the image border (same idea as Week 2), thresholded with Otsu.
#   2. Edge cue: high-pass filter (Prewitt by default) on the grey image,
#      thresholded with Otsu. Each connected group of edges (a frame outline)
#      is filled with its convex hull ("hull"), or the regions enclosed by
#      edges are found with a flood fill from the border ("flood").
#   3. Fusion of both cues, morphological clean-up and hole filling.
#   4. Connected components -> keep at most 2 that are big and rectangular.
#   5. Fit a quadrilateral to each one (paintings may be slightly rotated) and
#      order the paintings left-to-right (or top-to-bottom when stacked).

BORDER_SIZE = 15
MAX_PAINTINGS = 2

EDGE_KERNELS = {
    "prewitt": (
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32),
        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32),
    ),
    "sobel": (
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32),
    ),
    "scharr": (
        np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], np.float32),
        np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], np.float32),
    ),
}

DEFAULT_PARAMS = {
    # Colour cue
    "color_space": "HSV",
    "channels": [1, 2],
    # Edge cue
    "edge_operator": "prewitt",
    "blur_ksize": 5,
    "edge_close_frac": 0.01,
    "edge_fill": "hull",
    # Fusion and clean-up
    "fusion": "or",
    "open_frac": 0.02,
    # Candidate selection
    "min_area_frac": 0.02,
    "min_rel_area": 0.15,
    "min_rectangularity": 0.75,
    # Shape fitting
    "fit": "poly",
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
# Step 1: colour cue (Week 2 background model)
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


def color_cue(
    img_bgr: np.ndarray,
    color_space: Literal["HSV", "LAB"] = "HSV",
    channels: list[int] = (1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Foreground mask from the colour distance to the wall.

    Returns:
        (distance map, binary uint8 mask)
    """
    spaces = {"HSV": cv2.COLOR_BGR2HSV, "LAB": cv2.COLOR_BGR2LAB}
    h, w = img_bgr.shape[:2]
    color = cv2.cvtColor(img_bgr, spaces[color_space])
    dist = mahalanobis(color, border_mask(h, w), list(channels))
    src = np.clip(dist, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(src, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return dist, mask


# -----------------------------------------------------------------------------
# Step 2: edge cue (high-pass filters)
# -----------------------------------------------------------------------------
def gradient_magnitude(
    img_bgr: np.ndarray,
    operator: Literal["prewitt", "sobel", "laplacian"] = "prewitt",
    blur_ksize: int = 5,
) -> np.ndarray:
    """
    Gradient magnitude of the grey image, normalised to [0, 255].

    A Gaussian blur is applied first so that the wall/painting texture does
    not dominate over the frame edges.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    if operator == "laplacian":
        mag = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    else:
        kx, ky = EDGE_KERNELS[operator]
        gx = cv2.filter2D(gray, cv2.CV_32F, kx)
        gy = cv2.filter2D(gray, cv2.CV_32F, ky)
        mag = np.sqrt(gx ** 2 + gy ** 2)

    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def fill_from_border(edges: np.ndarray) -> np.ndarray:
    """
    Regions that cannot be reached from the image border without crossing
    an edge pixel (i.e. regions enclosed by edges, plus the edges themselves).
    """
    h, w = edges.shape
    # Pad with a 1-pixel background frame so the flood fill starts everywhere
    # on the border at once.
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = edges
    flood = padded.copy()
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 1)
    # Pixels still at 0 were not reached -> enclosed.
    enclosed = (flood == 0)[1:-1, 1:-1]
    return (enclosed | (edges > 0)).astype(np.uint8)


def fill_hulls(edges: np.ndarray, min_area_frac: float = 0.01) -> np.ndarray:
    """
    Fill the convex hull of every connected group of edges.

    A painting frame is usually one connected group of edges, so its hull is
    the whole painting even when the frame outline has small gaps (where a
    flood fill would leak). Tiny groups are ignored.
    """
    out = np.zeros_like(edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        hull = cv2.convexHull(contour)
        if cv2.contourArea(hull) >= min_area_frac * edges.size:
            cv2.fillPoly(out, [hull], 1)
    return out


def edge_cue(
    img_bgr: np.ndarray,
    operator: str = "prewitt",
    blur_ksize: int = 5,
    close_frac: float = 0.01,
    fill: Literal["flood", "hull"] = "hull",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Foreground mask from the edges.

    Args:
        fill: "flood" keeps the regions enclosed by edges (flood fill from
            the border), "hull" fills the convex hull of each edge group.

    Returns:
        (gradient magnitude, binary edge map, filled regions mask)
    """
    mag = gradient_magnitude(img_bgr, operator, blur_ksize)
    _, edges = cv2.threshold(mag, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # The border itself is wall: drop edges touching it (image borders, crops).
    edges[border_mask(*edges.shape, b=2)] = 0
    edges = morph(edges, cv2.MORPH_CLOSE, close_frac)
    if fill == "flood":
        return mag, edges, fill_from_border(edges)
    if fill == "hull":
        return mag, edges, fill_hulls(edges)
    raise ValueError(f"Unknown edge fill: {fill}")


# -----------------------------------------------------------------------------
# Step 3: fusion and clean-up
# -----------------------------------------------------------------------------
def fuse(
    color_mask: np.ndarray,
    edge_mask: np.ndarray,
    method: Literal["or", "and", "color", "edge"] = "or",
) -> np.ndarray:
    """Combine the colour and the edge foreground masks."""
    if method == "or":
        return color_mask | edge_mask
    if method == "and":
        return color_mask & edge_mask
    if method == "color":
        return color_mask.copy()
    if method == "edge":
        return edge_mask.copy()
    raise ValueError(f"Unknown fusion: {method}")


def clean_mask(mask: np.ndarray, open_frac: float = 0.02) -> np.ndarray:
    """
    Fill holes and remove thin structures (edges of furniture, frame
    shadows...) with an opening. The opening also separates two paintings
    that are only joined by a thin bridge.
    """
    mask = fill_holes(mask)
    mask = morph(mask, cv2.MORPH_OPEN, open_frac)
    return fill_holes(mask)


# -----------------------------------------------------------------------------
# Step 4: candidate selection
# -----------------------------------------------------------------------------
def rectangularity(component: np.ndarray) -> float:
    """Area of the blob divided by the area of its minimum rotated rectangle."""
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    if rw * rh == 0:
        return 0.0
    return float(component.sum()) / (rw * rh)


def select_candidates(
    mask: np.ndarray,
    min_area_frac: float = 0.02,
    min_rel_area: float = 0.15,
    min_rectangularity: float = 0.75,
    max_paintings: int = MAX_PAINTINGS,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Split a mask into connected components and keep the painting-like ones.

    A component is kept when it is larger than `min_area_frac` of the image,
    at least `min_rel_area` times the largest component and its
    rectangularity is above `min_rectangularity`.

    Returns:
        (kept component masks sorted by area, info about every component)
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return [], []

    image_area = mask.size
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = areas.max()

    info = []
    for label in range(1, n):
        component = (labels == label).astype(np.uint8)
        area = stats[label, cv2.CC_STAT_AREA]
        rect = rectangularity(component)
        reasons = []
        if area < min_area_frac * image_area:
            reasons.append("too small")
        if area < min_rel_area * largest:
            reasons.append("small vs largest")
        if rect < min_rectangularity:
            reasons.append("not rectangular")
        info.append({
            "label": label,
            "area": int(area),
            "rectangularity": rect,
            "bbox": stats[label, :4].tolist(),
            "kept": not reasons,
            "reasons": reasons,
            "mask": component,
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
# Step 5: shape fitting and ordering
# -----------------------------------------------------------------------------
def fit_shape(
    component: np.ndarray,
    method: Literal["min_area_rect", "bbox", "poly", "none"] = "min_area_rect",
) -> np.ndarray:
    """
    Replace a blob by a clean quadrilateral.

    "min_area_rect": rotated rectangle (paintings are slightly rotated).
    "bbox": axis-aligned bounding box.
    "poly": convex hull simplified with approxPolyDP.
    "none": keep the blob as is.
    """
    if method == "none":
        return component
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return component
    contour = max(contours, key=cv2.contourArea)

    if method == "min_area_rect":
        points = cv2.boxPoints(cv2.minAreaRect(contour))
    elif method == "bbox":
        x, y, w, h = cv2.boundingRect(contour)
        points = np.array([[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]])
    elif method == "poly":
        hull = cv2.convexHull(contour)
        points = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    else:
        raise ValueError(f"Unknown fit: {method}")

    out = np.zeros_like(component)
    cv2.fillPoly(out, [np.round(points).astype(np.int32).reshape(-1, 1, 2)], 1)
    return out


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

    color_dist, color_mask = color_cue(img_bgr, p["color_space"], p["channels"])
    grad, edges, edge_mask = edge_cue(
        img_bgr, p["edge_operator"], p["blur_ksize"], p["edge_close_frac"],
        p["edge_fill"],
    )
    fused = fuse(color_mask, edge_mask, p["fusion"])
    cleaned = clean_mask(fused, p["open_frac"])
    candidates, info = select_candidates(
        cleaned, p["min_area_frac"], p["min_rel_area"], p["min_rectangularity"]
    )
    masks = order_masks([fit_shape(c, p["fit"]) for c in candidates])

    return {
        "params": p,
        "color_dist": color_dist,
        "color_mask": color_mask,
        "gradient": grad,
        "edges": edges,
        "edge_mask": edge_mask,
        "fused": fused,
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
