import cv2
import numpy as np


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


def best_descriptor_3d(
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the best 3D descriptor found on the development set.

    Args:
        image: Non-empty uint8 OpenCV Lab image of shape (H, W, 3).
        mask: Optional foreground mask of shape (H, W).

    Returns:
        A normalized float32 descriptor containing 2048 values.
        Use histogram intersection when comparing descriptors.
    """
    return spatial_descriptor_3d(
        image=image,
        bins=(2, 8, 8),
        grids=(4,),
        mask=mask,
    )