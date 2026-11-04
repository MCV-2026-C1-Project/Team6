"""
Week 4 painting detection.

Combines the Week 3 edge segmentation with the Week 2 wall colour model:

    1. resize      work on an image of at most WORK_SIDE px (masks are scaled back)
    2. edges       Prewitt gradient of HSV S,V thresholded at EDGE_K * Otsu         (W3)
    3. regions     closing + hole filling + opening of the edges                     (W3)
    4. wall        Mahalanobis distance of the Lab chroma to the border pixels      (W2)
    5. split       cut a region where a band is wall-coloured AND has no edges      (new)
    6. selection   keep regions by area / rectangularity, at most MAX_PAINTINGS     (W3)
    7. rectangle   projection rectangle of every painting                           (W2)
    8. fallback    no painting found -> the painting fills the whole image          (new)
    9. order       by columns: left-to-right, top-to-bottom inside a column         (W3 rule)

Step 5 fixes the Week 3 failures on QSD1-W4: two stacked paintings whose regions
merge because closing / hole filling also fills the strip of wall between them.
Inside a painting there is always colour that is not wall or an edge, while the
strip of wall between two paintings has neither, so the region is cut there.

Usage:
    from segmentation import detect_paintings
    masks = detect_paintings(img_bgr)   # list of 1 to MAX_PAINTINGS uint8 masks (1 inside)

The parameters come from searches/segmentation_search.py (QSD1-W4 + QSD2-W3). Images
with 3 paintings are tested on synthetic data in searches/segmentation_3paintings_test.py.
"""

import cv2
import numpy as np
from scipy import ndimage as ndi

WORK_SIDE = 1024          # the image is processed with its longest side at most this size

# Edges (Week 3)
EDGE_CHANNELS = [1, 2]    # HSV channels of the gradient (S, V)
EDGE_K = 0.5              # edge threshold = EDGE_K * Otsu threshold
CLOSE_FRAC = 0.02         # closing kernel, fraction of the smallest image side
OPEN_FRAC = 0.02          # opening kernel, fraction of the smallest image side

# Wall colour model (Week 2)
WALL_CHANNELS = [1, 2]    # Lab channels of the model (a, b): ignores shadows, which only change L
WALL_BORDER_FRAC = 0.02   # width of the border used as wall sample, fraction of the smallest side
WALL_TRIM = 0.2           # drop this fraction of the border pixels farthest from the median
WALL_THR = 4.0            # Mahalanobis distance above which a pixel is not wall

# Split of merged paintings (new)
SPLIT_THR = 0.2           # cut if the band is < SPLIT_THR * the content of both sides
SPLIT_MARGIN = 0.15       # no cut in the outer 15% of a region (thin borders are not paintings)
SPLIT_DEPTH = 2           # recursion depth of the cuts: one region can become up to 4 parts
SPLIT_MAX_ASPECT = 8      # no cut that leaves a part more elongated than 8:1 (museum paintings are <= 5.8:1)

# Selection and rectangle (Week 3)
MAX_PAINTINGS = 3         # at most 3 paintings per image
MIN_AREA_FRAC = 0.02      # min region area, fraction of the image
MIN_REL_AREA = 0.15       # min region area, fraction of the largest region
MIN_EXTENT = 0.6          # min region area / bounding-box area
PROJ_FRAC = 0.8           # row/column kept if > PROJ_FRAC * max projection

PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
PREWITT_Y = PREWITT_X.T.copy()


# =============================================================================
# HELPERS
# =============================================================================

def resize_max_side(img: np.ndarray, max_side: int = WORK_SIDE) -> np.ndarray:
    """Downscale an image so its longest side is at most max_side (never upscale)."""
    s = max_side / max(img.shape[:2])
    return img if s >= 1 else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def border_mask(h: int, w: int, b: int) -> np.ndarray:
    """Boolean mask of a frame of width b along the image border."""
    b = max(1, min(b, h // 2, w // 2))
    m = np.zeros((h, w), bool)
    m[:b, :] = m[-b:, :] = True
    m[:, :b] = m[:, -b:] = True
    return m


def square_kernel(frac: float, shape: tuple[int, int]) -> np.ndarray:
    """Square structuring element with side frac * smallest image side (at least 2 px)."""
    n = max(2, int(frac * min(shape)))
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill the holes of every blob of a binary mask (uint8, 1 inside)."""
    return ndi.binary_fill_holes(mask > 0).astype(np.uint8)


# =============================================================================
# 1-3. EDGES AND REGIONS (Week 3)
# =============================================================================

def gradient_magnitude(img_bgr: np.ndarray) -> np.ndarray:
    """
    Prewitt gradient magnitude of the HSV EDGE_CHANNELS.

    Each channel response is normalised to [0, 1] and the maximum over the
    channels is kept, so an edge visible in either channel is detected.

    Returns:
        uint8 array of shape (h, w) with values in [0, 255].
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mag = np.zeros(hsv.shape[:2], np.float32)
    for c in EDGE_CHANNELS:
        channel = hsv[..., c].astype(np.float32)
        g = np.hypot(cv2.filter2D(channel, cv2.CV_32F, PREWITT_X), cv2.filter2D(channel, cv2.CV_32F, PREWITT_Y))
        mag = np.maximum(mag, g / (g.max() + 1e-6))
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def edge_segmentation(grad: np.ndarray) -> np.ndarray:
    """
    Binarize the gradient with EDGE_K times its Otsu threshold.

    Otsu alone is dominated by the strongest painting, so a faint frame next to
    it would be lost. The image border is wall and is cleared.

    Returns:
        uint8 edge map (1 on the edges).
    """
    t, _ = cv2.threshold(grad, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = (grad > EDGE_K * t).astype(np.uint8)
    edges[border_mask(*edges.shape, 2)] = 0
    return edges


def edges_to_regions(edges: np.ndarray) -> np.ndarray:
    """
    Turn the edge map into solid regions.

    The closing joins the gaps of each frame outline so filling the holes turns
    it into a solid painting; the opening removes thin structures (wires,
    labels) and the holes are filled again.

    Returns:
        uint8 mask (1 inside the regions).
    """
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, square_kernel(CLOSE_FRAC, edges.shape))
    mask = fill_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, square_kernel(OPEN_FRAC, mask.shape))
    return fill_holes(mask)


# =============================================================================
# 4. WALL COLOUR MODEL (Week 2)
# =============================================================================

def wall_distance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Mahalanobis distance of every pixel to the wall colour.

    The wall is modelled as a Gaussian on the Lab WALL_CHANNELS of the border
    pixels. The WALL_TRIM border pixels farthest from the median are dropped
    first, so a painting touching the border does not spoil the model.

    Returns:
        float32 array of shape (h, w): small on the wall, large elsewhere.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)[..., WALL_CHANNELS]
    h, w = lab.shape[:2]
    px = lab[border_mask(h, w, max(2, int(WALL_BORDER_FRAC * min(h, w))))]
    dev = np.abs(px - np.median(px, axis=0)).sum(axis=1)
    px = px[dev <= np.quantile(dev, 1 - WALL_TRIM)]

    cov = np.atleast_2d(np.cov(px, rowvar=False)) + np.eye(len(WALL_CHANNELS)) * 1e-3
    delta = lab - px.mean(axis=0)
    d2 = np.einsum("...i,ij,...j->...", delta, np.linalg.pinv(cov), delta)
    return np.sqrt(np.maximum(d2, 0)).astype(np.float32)


def content_map(edges: np.ndarray, wall_dist: np.ndarray) -> np.ndarray:
    """Pixels that are painting evidence: not wall-coloured, or on an edge (uint8, 1 = content)."""
    return ((wall_dist > WALL_THR) | (edges > 0)).astype(np.uint8)


# =============================================================================
# 5. SPLIT OF MERGED PAINTINGS (new)
# =============================================================================

def cut_range(n: int, other: int) -> tuple[int, int]:
    """
    Positions [lo, hi) where a region of length n (and length `other` along the
    other axis) may be cut: both parts at least SPLIT_MARGIN of n and not more
    elongated than SPLIT_MAX_ASPECT.
    """
    min_part = max(1, int(SPLIT_MARGIN * n), int(np.ceil(other / SPLIT_MAX_ASPECT)))
    return min_part, n - min_part


def find_cut(region: np.ndarray, content: np.ndarray):
    """
    Find the best row or column where a region should be cut in two.

    Inside the bounding box of the region, the fraction of content pixels is
    computed per row and per column. A cut is accepted at the minimum of a
    profile when it is below SPLIT_THR times the median content of both sides.
    The minimum is searched away from the borders: each part must be at least
    SPLIT_MARGIN of the region and not more elongated than SPLIT_MAX_ASPECT
    (so e.g. the plain top band of a frame is not cut off as a painting).

    Returns:
        (axis, index) with axis 0 for a row and 1 for a column, in image
        coordinates, or None if the region is a single painting.
    """
    ys, xs = np.nonzero(region)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    inside = content[y0:y1, x0:x1] * region[y0:y1, x0:x1]

    best = None
    for axis, profile, offset in ((0, inside.mean(axis=1), y0), (1, inside.mean(axis=0), x0)):
        n = len(profile)
        lo, hi = cut_range(n, inside.shape[1 - axis])
        if hi <= lo:
            continue
        smooth = np.convolve(profile, np.ones(3) / 3, mode="same")
        i = lo + int(np.argmin(smooth[lo:hi]))
        ref = min(np.median(profile[:i]), np.median(profile[i:]))
        if ref > 0 and smooth[i] < SPLIT_THR * ref and (best is None or smooth[i] / ref < best[0]):
            best = (smooth[i] / ref, axis, offset + i)
    return None if best is None else best[1:]


def split_region(region: np.ndarray, content: np.ndarray, depth: int = SPLIT_DEPTH) -> list[np.ndarray]:
    """
    Recursively cut a region into the paintings it contains (XY-cut).

    Returns:
        List of uint8 masks; [region] if no cut is found.
    """
    if depth == 0 or not region.any():
        return [region]
    cut = find_cut(region, content)
    if cut is None:
        return [region]
    axis, i = cut
    a, b = region.copy(), region.copy()
    if axis == 0:
        a[i:], b[:i] = 0, 0
    else:
        a[:, i:], b[:, :i] = 0, 0
    return split_region(a, content, depth - 1) + split_region(b, content, depth - 1)


def split_regions(regions: np.ndarray, content: np.ndarray) -> list[np.ndarray]:
    """Split every large enough connected region; returns the list of parts."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(regions, connectivity=8)
    parts = []
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= MIN_AREA_FRAC * regions.size:
            parts += split_region((labels == label).astype(np.uint8), content)
    return parts


# =============================================================================
# 6-9. SELECTION, RECTANGLE, FALLBACK, ORDER
# =============================================================================

def select_paintings(parts: list[np.ndarray]) -> list[np.ndarray]:
    """
    Keep up to MAX_PAINTINGS parts as paintings.

    Every part is reduced to its largest blob with the holes filled. A part is
    kept when it is at least MIN_AREA_FRAC of the image, at least MIN_REL_AREA
    of the largest part and fills at least MIN_EXTENT of its bounding box.

    Returns:
        List of uint8 masks, largest first.
    """
    blobs = []
    for part in parts:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(part, connectivity=8)
        if n < 2:
            continue
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        _, _, w, h, area = stats[label]
        blobs.append((area, area / (w * h), fill_holes(labels == label)))
    if not blobs:
        return []

    size = parts[0].size
    largest = max(area for area, _, _ in blobs)
    kept = [(area, blob) for area, extent, blob in blobs
            if area >= MIN_AREA_FRAC * size and area >= MIN_REL_AREA * largest and extent >= MIN_EXTENT]
    return [blob for _, blob in sorted(kept, key=lambda t: -t[0])[:MAX_PAINTINGS]]


def mask_to_rect(mask: np.ndarray) -> np.ndarray:
    """
    Projection rectangle of one painting (Week 2).

    A column (row) belongs to the painting if its fraction of foreground pixels
    exceeds PROJ_FRAC times the maximum over all columns (rows), which ignores
    thin protrusions that a plain bounding box would include.
    """
    rect = np.zeros(mask.shape, np.uint8)
    if not mask.any():
        return rect
    cols, rows = mask.mean(axis=0), mask.mean(axis=1)
    x0, x1 = np.where(cols > PROJ_FRAC * cols.max())[0][[0, -1]]
    y0, y1 = np.where(rows > PROJ_FRAC * rows.max())[0][[0, -1]]
    rect[y0:y1 + 1, x0:x1 + 1] = 1
    return rect


def stacked(a: tuple, b: tuple) -> bool:
    """Week 3 rule: two boxes (x, y, w, h) are stacked if they overlap horizontally > 50% of the narrower."""
    return min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]) > 0.5 * min(a[2], b[2])


def side_by_side(a: tuple, b: tuple) -> bool:
    """Two boxes (x, y, w, h) are side by side if they overlap vertically > 50% of the shorter."""
    return min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]) > 0.5 * min(a[3], b[3])


def groups(items: list[int], related) -> list[list[int]]:
    """Connected components of a relation: items related directly or through a chain are grouped."""
    left, out = list(items), []
    while left:
        group, todo = [], [left.pop(0)]
        while todo:
            i = todo.pop()
            group.append(i)
            linked = [j for j in left if related(i, j)]
            left = [j for j in left if j not in linked]
            todo += linked
        out.append(group)
    return out


def order_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    """
    Order masks by columns: left-to-right, and top-to-bottom inside a column.

    Stacked paintings (Week 3 rule), also through a chain, form a column.
    Columns go from left to right by their left edge. Inside a column the
    paintings go from top to bottom, and side-by-side ones (one painting above
    two) from left to right. With 2 masks this is the Week 3 order: the upper
    one first if they are stacked, the left one first otherwise.
    """
    boxes = [cv2.boundingRect(m) for m in masks]
    order = []
    columns = groups(list(range(len(masks))), lambda i, j: stacked(boxes[i], boxes[j]))
    for col in sorted(columns, key=lambda g: min(boxes[i][0] for i in g)):
        rows = groups(col, lambda i, j: side_by_side(boxes[i], boxes[j]))
        for row in sorted(rows, key=lambda g: min(boxes[i][1] for i in g)):
            order += sorted(row, key=lambda i: boxes[i][0])
    return [masks[i] for i in order]


# =============================================================================
# PIPELINE
# =============================================================================

def segment_steps(img_bgr: np.ndarray) -> dict:
    """
    Run the whole detection and keep every intermediate result (for plots).

    Args:
        img_bgr: Input BGR image (already denoised).

    Returns:
        Dict with the work-size results "work", "grad", "edges", "regions",
        "wall_dist", "content", "parts", "paintings", "rects", the flag
        "fallback", and "masks": the final full-size masks.
    """
    h, w = img_bgr.shape[:2]
    s = {"work": resize_max_side(img_bgr)}
    s["grad"] = gradient_magnitude(s["work"])
    s["edges"] = edge_segmentation(s["grad"])
    s["regions"] = edges_to_regions(s["edges"])
    s["wall_dist"] = wall_distance(s["work"])
    s["content"] = content_map(s["edges"], s["wall_dist"])
    s["parts"] = split_regions(s["regions"], s["content"])
    s["paintings"] = select_paintings(s["parts"])
    s["rects"] = order_masks([mask_to_rect(p) for p in s["paintings"]])
    s["fallback"] = not s["rects"]
    if s["fallback"]:
        s["rects"] = [np.ones(s["work"].shape[:2], np.uint8)]
    s["masks"] = [cv2.resize(r, (w, h), interpolation=cv2.INTER_NEAREST) for r in s["rects"]]
    return s


def detect_paintings(img_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Detect up to two paintings in an image.

    Args:
        img_bgr: Input BGR image (already denoised).

    Returns:
        List with 1 to MAX_PAINTINGS binary uint8 masks (one per painting, 1 inside),
        ordered left-to-right / top-to-bottom, each the size of the image.
    """
    return segment_steps(img_bgr)["masks"]
