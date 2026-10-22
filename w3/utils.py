from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi


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
