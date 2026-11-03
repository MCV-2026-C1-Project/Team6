"""Keypoint detectors. Every detector returns a list of cv2.KeyPoint with
size = 2*sigma, angle (rotation invariance) and response (strength) set."""
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

KPs = List[cv2.KeyPoint]


# ---------------------------------------------------------------- preprocessing
def resize_max_side(img: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Downscale so the longest side is <= max_side: makes thresholds/scales comparable across images and cuts runtime."""
    s = max_side / max(img.shape[:2])
    return img if s >= 1 else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def to_gray(img: np.ndarray, clahe: bool = True) -> np.ndarray:
    """BGR -> gray, optionally with CLAHE (local contrast equalisation -> illumination robustness)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.createCLAHE(2.0, (8, 8)).apply(g) if clahe else g


# ---------------------------------------------------------------- DoG
def dog(gray: np.ndarray, contrast: float = 0.04, edge: float = 10, sigma: float = 1.6) -> KPs:
    """DoG extrema in space+scale (SIFT detector). `contrast` lower -> more keypoints; `edge` rejects edge-like points."""
    return list(cv2.SIFT_create(contrastThreshold=contrast, edgeThreshold=edge, sigma=sigma).detect(gray))


# ---------------------------------------------------------------- Harris-Laplace
def harris_laplace(gray: np.ndarray, n_scales: int = 8, sigma0: float = 1.6, k: float = 1.4,
                   harris_thr: float = 1e-6, log_thr: float = 0.01, alpha: float = 0.04) -> KPs:
    """Harris corners in space (scale-normalised), kept only if the normalised LoG is maximal across neighbouring scales.

    Args:
        n_scales: number of scales evaluated; sigma_i = sigma0 * k**i.
        harris_thr: min (scale-normalised) Harris response -> main knob for the keypoint count.
        log_thr: min normalised |LoG| response (rejects flat regions).
    """
    g = gray.astype(np.float32) / 255
    sigmas = [sigma0 * k ** i for i in range(n_scales + 2)]  # +2 = neighbours needed for scale selection
    # Scale-normalised |LoG| per scale (sigma^2 compensates the response decay with scale)
    log = [s ** 2 * np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), s), cv2.CV_32F)) for s in sigmas]
    ones, kps = np.ones((3, 3), np.uint8), []
    for i in range(1, n_scales + 1):
        s = sigmas[i]
        L = cv2.GaussianBlur(g, (0, 0), 0.7 * s)  # differentiation scale
        Ix = cv2.Sobel(L, cv2.CV_32F, 1, 0) * 0.7 * s  # scale-normalised gradients
        Iy = cv2.Sobel(L, cv2.CV_32F, 0, 1) * 0.7 * s
        A, B, C = (cv2.GaussianBlur(m, (0, 0), s) for m in (Ix * Ix, Iy * Iy, Ix * Iy))  # 2nd-moment matrix (integration scale s)
        R = A * B - C * C - alpha * (A + B) ** 2  # Harris response
        ok = ((R > harris_thr) & (R == cv2.dilate(R, ones))  # spatial 3x3 local maximum
              & (log[i] >= log[i - 1]) & (log[i] >= log[i + 1]) & (log[i] > log_thr))  # LoG maximum in scale
        ys, xs = np.nonzero(ok)
        kps += [cv2.KeyPoint(float(x), float(y), 2 * s, -1, float(R[y, x])) for y, x in zip(ys, xs)]
    return add_orientation(gray, kps)


def add_orientation(gray: np.ndarray, kps: KPs, bins: int = 36, peak: float = 0.8) -> KPs:
    """Rotation invariance: dominant gradient orientation(s) in a Gaussian window (as in SIFT).
    A keypoint is duplicated for each histogram peak >= `peak` * max."""
    g = gray.astype(np.float32)
    mag, ang = cv2.cartToPolar(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1), angleInDegrees=True)
    H, W = g.shape
    out: KPs = []
    for kp in kps:
        x, y, s = int(kp.pt[0]), int(kp.pt[1]), kp.size / 2
        r = max(int(4.5 * s), 2)  # window radius = 3 * (1.5 sigma)
        y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, H), max(x - r, 0), min(x + r + 1, W)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        w = mag[y0:y1, x0:x1] * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (1.5 * s) ** 2))  # gaussian-weighted magnitude
        h = np.bincount((ang[y0:y1, x0:x1] / 360 * bins).astype(int).ravel() % bins, w.ravel(), bins)
        for _ in range(2):  # circular smoothing
            h = (np.roll(h, 1) + h + np.roll(h, -1)) / 3
        for b in np.nonzero((h > peak * h.max()) & (h > np.roll(h, 1)) & (h > np.roll(h, -1)))[0]:
            out.append(cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, (b + 0.5) * 360 / bins, kp.response))
    return out


# ---------------------------------------------------------------- Combination
def hybrid(gray: np.ndarray, dog_kw: Optional[dict] = None, hl_kw: Optional[dict] = None) -> KPs:
    """Union of DoG (blobs) and Harris-Laplace (corners): complementary structures -> more repeatable matches.
    Responses are max-normalised per detector so they are comparable (needed by ANMS); near-duplicates are dropped."""
    seen, out = set(), []
    for det, kw in ((dog, dog_kw), (harris_laplace, hl_kw)):
        kps = det(gray, **(kw or {}))
        top = max((k.response for k in kps), default=1) or 1
        for k in kps:
            key = (int(k.pt[0] // 2), int(k.pt[1] // 2), round(float(np.log2(k.size))), int(k.angle // 30))
            if key not in seen:
                seen.add(key)
                k.response /= top
                out.append(k)
    return out


# ---------------------------------------------------------------- Selection
def anms(kps: KPs, n: Optional[int], pool: int = 3000, c: float = 0.9) -> KPs:
    """Adaptive non-maximal suppression: keep the n keypoints that are strong AND spatially spread
    (plain top-n by response clusters on textured areas). `pool` bounds the O(n^2) cost."""
    if n is None or len(kps) <= n:
        return list(kps)
    kps = sorted(kps, key=lambda k: -k.response)[:pool]
    p, r = np.array([k.pt for k in kps]), np.array([k.response for k in kps])
    rad = np.full(len(kps), np.inf)  # strongest keypoint keeps an infinite suppression radius
    for i in range(1, len(kps)):
        m = r[i] < c * r[:i]  # significantly stronger keypoints
        if m.any():
            rad[i] = np.min(np.sum((p[:i][m] - p[i]) ** 2, 1))  # squared distance to the nearest one
    return [kps[i] for i in np.argsort(-rad)[:n]]


DETECTORS: Dict[str, Callable[..., KPs]] = {"dog": dog, "harris_laplace": harris_laplace, "hybrid": hybrid}


def detect(img: np.ndarray, method: str = "dog", max_kps: Optional[int] = None, clahe: bool = True,
           mask: Optional[np.ndarray] = None, **kw) -> KPs:
    """Detect keypoints on a BGR/gray image.

    Args:
        method: key of DETECTORS.
        max_kps: keep at most this many keypoints (ANMS). None = keep all.
        mask: optional uint8 mask (>0 = keep), e.g. the painting mask from W1: drops background keypoints.
        **kw: detector parameters (the grid-search knobs).
    """
    kps = DETECTORS[method](to_gray(img, clahe), **kw)
    if mask is not None:
        kps = [k for k in kps if mask[int(k.pt[1]), int(k.pt[0])] > 0]
    return anms(kps, max_kps)
