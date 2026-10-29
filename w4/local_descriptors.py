"""Local descriptors (3 options): RootSIFT, Opponent-ColorSIFT, ORB.
Each takes (BGR image, keypoints) and returns (keypoints, descriptors)."""
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from keypoints import KPs, detect, to_gray

Out = Tuple[KPs, np.ndarray]


def _root(d: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """RootSIFT: L1-normalise + sqrt. Euclidean distance then equals the Hellinger kernel (better than plain SIFT)."""
    return np.sqrt(d / (np.abs(d).sum(1, keepdims=True) + eps)).astype(np.float32)


def sift(img: np.ndarray, kps: KPs, root: bool = True, clahe: bool = True) -> Out:
    """SIFT (128-D, gradient histograms; rotation from kp.angle, scale from kp.size). root=True -> RootSIFT."""
    kps, d = cv2.SIFT_create().compute(to_gray(img, clahe), kps)
    return kps, (_root(d) if root else d)


# Opponent colour channels with (lo, hi) value range. O1, O2 are invariant to light intensity shifts, O3 = intensity.
# Fixed ranges (not per-image min-max) keep channels comparable across images.
_OPP = [(lambda r, g, b: (r - g) / 2 ** .5, (-180.3, 180.3)),
        (lambda r, g, b: (r + g - 2 * b) / 6 ** .5, (-208.2, 208.2)),
        (lambda r, g, b: (r + g + b) / 3 ** .5, (0.0, 441.7))]


def color_sift(img: np.ndarray, kps: KPs, root: bool = True) -> Out:
    """Opponent-ColorSIFT: RootSIFT on each opponent channel, concatenated (384-D, unit L2 norm).
    Adds colour information (very discriminative for paintings) while staying robust to illumination."""
    b, g, r = [c.astype(np.float32) for c in cv2.split(img)]
    sf, ds = cv2.SIFT_create(), []
    for f, (lo, hi) in _OPP:
        ch = ((f(r, g, b) - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
        kps, d = sf.compute(ch, kps)
        ds.append(_root(d) if root else d)
    return kps, (np.hstack(ds) / np.sqrt(3)).astype(np.float32)


def orb(img: np.ndarray, kps: KPs, clahe: bool = True, **kw) -> Out:
    """ORB / rotated BRIEF (32 bytes, binary -> use Hamming). Uses kp.angle; patch size is fixed (not scale adaptive).
    May drop keypoints too close to the border. **kw go to cv2.ORB_create."""
    kps = [cv2.KeyPoint(k.pt[0], k.pt[1], 31, k.angle, k.response, 0) for k in kps]
    return cv2.ORB_create(**kw).compute(to_gray(img, clahe), kps)


DESCRIPTORS: Dict[str, Callable[..., Out]] = {"sift": sift, "color_sift": color_sift, "orb": orb}


def describe(img: np.ndarray, kps: KPs, method: str = "sift", **kw) -> Out:
    """Compute descriptors for given keypoints. Returns an empty array if there are no keypoints."""
    if not kps:
        return [], np.zeros((0, 1), np.float32)
    kps, des = DESCRIPTORS[method](img, kps, **kw)
    return kps, (des if des is not None else np.zeros((0, 1), np.float32))


def extract(img: np.ndarray, det: str = "dog", desc: str = "sift", det_kw: Optional[dict] = None,
            desc_kw: Optional[dict] = None, mask: Optional[np.ndarray] = None) -> Out:
    """Full pipeline: image -> (keypoints, descriptors). This is the function to call inside the grid search."""
    return describe(img, detect(img, det, mask=mask, **(det_kw or {})), desc, **(desc_kw or {}))
