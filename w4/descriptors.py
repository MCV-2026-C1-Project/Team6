"""OpenCV local descriptors: (img, max_kps, color, clahe) -> (points Nx2, descriptors NxD)."""
import cv2
import numpy as np
from keypoints import detect
from utils import to_gray

try:
    AKAZE_create = cv2.AKAZE_create
except AttributeError:
    AKAZE_create = cv2.xfeatures2d.AKAZE_create


COLORS = ("gray", "gray_saturation", "opponent")


def color_channels(img, color="gray", clahe=True):
    if color == "gray":
        return [to_gray(img, clahe)]
    if color == "gray_saturation":
        return [to_gray(img, clahe), cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]]
    if color == "opponent":
        b, g, r = cv2.split(img.astype(np.float32))
        return [np.uint8((r - g + 255) / 2), np.uint8((r + g - 2 * b + 510) / 4), np.uint8((r + g + b) / 3)]
    raise ValueError(color)


def _describe(img, f, max_kps, color, clahe, root=False, kps=None):
    if kps is None:
        kps = sorted(f.detect(to_gray(img, clahe)), key=lambda k: -k.response)[:max_kps]
    chans = color_channels(img, color, clahe)
    kps, d = f.compute(chans[0], kps) if kps else ([], None)
    if d is None or not len(kps):
        return np.zeros((0, 2), np.float32), None
    ds = [d] + [f.compute(c, kps)[1] for c in chans[1:]]
    if root:
        ds = [np.sqrt(d / (d.sum(1, keepdims=True) + 1e-7)) for d in ds]
        return np.float32([k.pt for k in kps]), (np.hstack(ds) / np.sqrt(len(ds))).astype(np.float32)
    return np.float32([k.pt for k in kps]), np.hstack(ds)


def rootsift(img, max_kps=1000, color="gray", clahe=True, contrast=0.04, kps=None):
    return _describe(img, cv2.SIFT_create(contrastThreshold=contrast), max_kps, color, clahe, root=True, kps=kps)


def akaze(img, max_kps=1000, color="gray", clahe=True, threshold=5e-4):
    return _describe(img, AKAZE_create(threshold=threshold), max_kps, color, clahe)


def orb(img, max_kps=1000, color="gray", clahe=True):
    return _describe(img, cv2.ORB_create(nfeatures=max_kps), max_kps, color, clahe)


DESCRIPTORS = {"rootsift": rootsift, "akaze": akaze, "orb": orb}
NORMS = {"rootsift": cv2.NORM_L2, "akaze": cv2.NORM_HAMMING, "orb": cv2.NORM_HAMMING}


def describe(image: np.ndarray, descriptor: str = "rootsift", detector: str | None = None,
             color: str = "gray", clahe: bool = True, max_keypoints: int = 1000):
    """
    Local features of an image, described as given (no resizing).

    Args:
        image: BGR image.
        descriptor: Key of DESCRIPTORS.
        detector: Key of keypoints.DETECTORS, or None for the descriptor's own detector.
        color: One of COLORS.
        clahe: Equalise the local contrast before detecting and describing.
        max_keypoints: Maximum number of keypoints.

    Returns:
        Tuple (points, descriptors): keypoint coordinates (N, 2) and their
        descriptors (N, D), or None descriptors if no keypoint was found.
    """
    if detector is None:
        return DESCRIPTORS[descriptor](image, max_keypoints, color, clahe)
    if descriptor != "rootsift":
        raise ValueError("A custom keypoint detector is only supported with rootsift")
    keypoints = detect(image, detector, max_kps=max_keypoints, clahe=clahe)
    return rootsift(image, max_keypoints, color, clahe, kps=keypoints)
