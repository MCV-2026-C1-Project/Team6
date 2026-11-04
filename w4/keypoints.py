"""Keypoint detectors. Each detector returns a list of cv2.KeyPoint with size = 2*sigma and
response (strength). Orientation is added once, for every detector, inside detect().

Detectors only decide WHERE the points are (and at which scale); descriptors.py decides HOW each
point is described. They are independent, so any detector can be combined with any descriptor
(e.g. akaze_det here has nothing to do with the akaze descriptor there).

    dog              blobs   (SIFT detector)
    harris_laplace   corners (scale-normalised Harris + Laplacian scale selection)
    hessian_laplace  blobs / stroke crossings (det Hessian + Laplacian scale selection)
    akaze_det        Hessian extrema in a nonlinear (edge-preserving) scale space
    fast             many fast corners on an image pyramid (pairs well with binary descriptors)
    mser             stable flat regions (colour areas the other detectors ignore)
    dense            regular grid, no detection: baseline that guarantees coverage
    hybrid           union of dog + harris_laplace (blobs + corners)
"""
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

KPs = List[cv2.KeyPoint]


def resize_max_side(img: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Downscale an image so its longest side is at most max_side (never upscales)."""
    scale = max_side / max(img.shape[:2])
    if scale >= 1:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def to_gray(img: np.ndarray, clahe: bool = True) -> np.ndarray:
    """BGR -> gray, optionally with CLAHE (local contrast equalisation)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.createCLAHE(2.0, (8, 8)).apply(g) if clahe else g


# ---------------------------------------------------------------- DoG
def dog(gray: np.ndarray, contrast: float = 0.04, edge: float = 10, sigma: float = 1.6) -> KPs:
    """DoG extrema in space+scale. `contrast` lower -> more keypoints; `edge` rejects edge-like points."""
    kps = cv2.SIFT_create(contrastThreshold=contrast, edgeThreshold=edge, sigma=sigma).detect(gray)
    # SIFT returns one keypoint per orientation: keep one per location/scale (orientation is added in detect())
    return list({(k.pt, k.size): cv2.KeyPoint(*k.pt, k.size, -1, k.response) for k in kps}.values())


# ---------------------------------------------------------------- Harris-Laplace / Hessian-Laplace
def _laplace_detector(gray, response, thr, n_scales, sigma0, k, log_thr) -> KPs:
    """Shared scheme: at each scale find spatial maxima of `response`, keep those where the
    scale-normalised |LoG| is also maximal across neighbouring scales (automatic scale selection)."""
    g = gray.astype(np.float32) / 255
    sigmas = [sigma0 * k ** i for i in range(n_scales + 2)]  # +2 = neighbours needed for scale selection
    log = [s ** 2 * np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), s), cv2.CV_32F)) for s in sigmas]
    ones, kps = np.ones((3, 3), np.uint8), []
    for i in range(1, n_scales + 1):
        R, valid = response(g, sigmas[i])
        ok = ((R > thr) & valid & (R == cv2.dilate(R, ones))                       # spatial 3x3 maximum
              & (log[i] >= log[i - 1]) & (log[i] >= log[i + 1]) & (log[i] > log_thr))  # LoG maximum in scale
        ys, xs = np.nonzero(ok)
        kps += [cv2.KeyPoint(float(x), float(y), 2 * sigmas[i], -1, float(R[y, x])) for y, x in zip(ys, xs)]
    return kps


def harris_laplace(gray: np.ndarray, n_scales: int = 8, sigma0: float = 1.6, k: float = 1.4,
                   harris_thr: float = 1e-6, log_thr: float = 0.01, alpha: float = 0.04) -> KPs:
    """Harris corners (scale-normalised) + Laplacian scale selection.
    harris_thr: min Harris response -> main knob for the keypoint count. log_thr: rejects flat regions."""
    def response(g, s):
        L = cv2.GaussianBlur(g, (0, 0), 0.7 * s)                            # differentiation scale
        Ix, Iy = (cv2.Sobel(L, cv2.CV_32F, *d) * 0.7 * s for d in ((1, 0), (0, 1)))
        A, B, C = (cv2.GaussianBlur(m, (0, 0), s) for m in (Ix * Ix, Iy * Iy, Ix * Iy))  # 2nd-moment matrix
        return A * B - C * C - alpha * (A + B) ** 2, True
    return _laplace_detector(gray, response, harris_thr, n_scales, sigma0, k, log_thr)


def hessian_laplace(gray: np.ndarray, n_scales: int = 8, sigma0: float = 1.6, k: float = 1.4,
                    det_thr: float = 2e-4, log_thr: float = 0.01, edge: float = 10.0) -> KPs:
    """det(Hessian) maxima (blobs, stroke crossings) + Laplacian scale selection.
    det_thr: min normalised det(H) -> main knob for the keypoint count. edge: rejects edge-like points."""
    kxx = np.array([[1, -2, 1]], np.float32)
    kxy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], np.float32) / 4
    def response(g, s):
        L = cv2.GaussianBlur(g, (0, 0), s)
        Lxx, Lyy, Lxy = (cv2.filter2D(L, -1, kern) for kern in (kxx, kxx.T, kxy))
        det = Lxx * Lyy - Lxy ** 2
        return s ** 4 * det, (Lxx + Lyy) ** 2 < (edge + 1) ** 2 / edge * det
    return _laplace_detector(gray, response, det_thr, n_scales, sigma0, k, log_thr)


def fast(gray: np.ndarray, thr: int = 20, n_levels: int = 4, ratio: float = 1.5, sigma0: float = 2.0) -> KPs:
    """FAST-9 corners on an image pyramid. thr: intensity difference (lower -> more points).
    Level l is downscaled by ratio**l and gets size = 2 * sigma0 * ratio**l (sigma0 sets the descriptor support)."""
    det, kps = cv2.FastFeatureDetector_create(int(thr)), []
    for l in range(n_levels):
        f = ratio ** l
        img = gray if l == 0 else cv2.resize(gray, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
        kps += [cv2.KeyPoint((k.pt[0] + .5) * f - .5, (k.pt[1] + .5) * f - .5, 2 * sigma0 * f, -1, k.response)
                for k in det.detect(img)]
    return kps

def add_orientation(gray: np.ndarray, kps: KPs, bins: int = 36, peak: float = 0.8) -> KPs:
    """Rotation invariance: dominant gradient orientation(s) in a Gaussian window (as in SIFT).
    A keypoint is duplicated for each histogram peak >= `peak` * max."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    mag, ang = cv2.cartToPolar(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1), angleInDegrees=True)
    H, W = g.shape
    out: KPs = []
    for kp in kps:
        x, y, s = min(int(kp.pt[0]), W - 1), min(int(kp.pt[1]), H - 1), kp.size / 2
        r = int(np.clip(4.5 * s, 2, 64))  # window radius = 3 * (1.5 sigma)
        y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, H), max(x - r, 0), min(x + r + 1, W)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        w = mag[y0:y1, x0:x1] * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (1.5 * s) ** 2))
        h = np.bincount((ang[y0:y1, x0:x1] / 360 * bins).astype(int).ravel() % bins, w.ravel(), bins)
        for _ in range(2):  # circular smoothing
            h = (np.roll(h, 1) + h + np.roll(h, -1)) / 3
        peaks = np.nonzero((h >= peak * h.max()) & (h >= np.roll(h, 1)) & (h >= np.roll(h, -1)))[0] if h.max() > 0 else [0]
        out += [cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, (b + 0.5) * 360 / bins, kp.response) for b in peaks]
    return out


# ---------------------------------------------------------------- Selection
def anms(kps: KPs, n: Optional[int], pool: int = 3000, c: float = 0.9) -> KPs:
    """Adaptive non-maximal suppression: keep the n keypoints that are strong AND spatially spread
    (plain top-n by response clusters on textured areas). `pool` bounds the O(pool^2) cost."""
    if n is None or len(kps) <= n:
        return list(kps)
    kps = sorted(kps, key=lambda k: -k.response)[:pool]
    p, r = np.float32([k.pt for k in kps]), np.float32([k.response for k in kps])
    d2 = ((p[:, None] - p[None]) ** 2).sum(-1)
    d2[~(r[:, None] < c * r[None])] = np.inf      # only distances to significantly stronger keypoints
    return [kps[i] for i in np.argsort(-d2.min(1), kind="stable")[:n]]  # strongest ones have radius inf


DETECTORS: Dict[str, Callable[..., KPs]] = {
    "dog": dog, "harris_laplace": harris_laplace, "hessian_laplace": hessian_laplace, "fast": fast}


def detect(img: np.ndarray, method: str = "dog", max_kps: Optional[int] = None, clahe: bool = True,
           mask: Optional[np.ndarray] = None, **kw) -> KPs:
    """Detect keypoints on a BGR/gray image: gray(+CLAHE) -> detector -> mask -> ANMS -> orientation.

    max_kps limits the number of locations; orientation can duplicate a point for a second strong
    orientation, so the returned list can be ~10-20% longer.

    Args:
        method: key of DETECTORS.
        max_kps: keep at most this many keypoints (ANMS). None = keep all.
        mask: optional uint8 mask (>0 = keep), e.g. the painting mask from W1: drops background keypoints.
        **kw: detector parameters (the grid-search knobs).
    """
    gray = to_gray(img, clahe)
    kps = DETECTORS[method](gray, **kw)
    if mask is not None:
        kps = [k for k in kps if mask[min(int(k.pt[1]), mask.shape[0] - 1), min(int(k.pt[0]), mask.shape[1] - 1)] > 0]
    return add_orientation(gray, anms(kps, max_kps))
