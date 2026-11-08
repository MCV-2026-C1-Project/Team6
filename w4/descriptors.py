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
    """
    Split an image into the uint8 channels that are described.

    "gray" gives the gray image, "gray_saturation" adds the HSV saturation
    and "opponent" gives the opponent colour channels (r - g, r + g - 2b,
    r + g + b), rescaled to [0, 255].

    Args:
        img: BGR image.
        color: One of COLORS.
        clahe: Apply CLAHE to the gray channel.

    Returns:
        List of uint8 channels of shape (h, w); the first one is used to
        compute the keypoints.

    Raises:
        ValueError: If color is not one of COLORS.
    """
    if color == "gray":
        return [to_gray(img, clahe)]
    if color == "gray_saturation":
        return [to_gray(img, clahe), cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]]
    if color == "opponent":
        b, g, r = cv2.split(img.astype(np.float32))
        return [np.uint8((r - g + 255) / 2), np.uint8((r + g - 2 * b + 510) / 4), np.uint8((r + g + b) / 3)]
    raise ValueError(color)


def _describe(img, f, max_kps, color, clahe, root=False, kps=None):
    """
    Detect and describe keypoints with an OpenCV feature object.

    The descriptors of every colour channel are concatenated. With root=True,
    each channel's descriptors are L1-normalised and square-rooted (RootSIFT)
    and the result is divided by sqrt(n_channels).

    Args:
        img: BGR image.
        f: OpenCV Feature2D (SIFT, AKAZE, ORB...).
        max_kps: Maximum number of keypoints, keeping the strongest responses.
        color: One of COLORS.
        clahe: Equalise the local contrast before detecting and describing.
        root: Apply the RootSIFT transform.
        kps: Precomputed keypoints. If None, f detects them.

    Returns:
        Tuple (points, descriptors): keypoint coordinates (N, 2) float32 and
        descriptors (N, D * n_channels), or an empty (0, 2) array and None if
        no keypoint was found.
    """
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
    """
    Compute RootSIFT features of an image.

    Args:
        img: BGR image.
        max_kps: Maximum number of keypoints.
        color: One of COLORS.
        clahe: Equalise the local contrast before detecting and describing.
        contrast: SIFT contrast threshold (lower -> more keypoints).
        kps: Precomputed keypoints (e.g. from keypoints.detect). If None, the
            SIFT detector is used.

    Returns:
        Tuple (points (N, 2), descriptors (N, 128 * n_channels) float32), or
        None descriptors if no keypoint was found.
    """
    return _describe(img, cv2.SIFT_create(contrastThreshold=contrast), max_kps, color, clahe, root=True, kps=kps)


def akaze(img, max_kps=1000, color="gray", clahe=True, threshold=5e-4):
    """
    Compute AKAZE features (binary descriptors, Hamming norm) of an image.

    Args:
        img: BGR image.
        max_kps: Maximum number of keypoints.
        color: One of COLORS.
        clahe: Equalise the local contrast before detecting and describing.
        threshold: AKAZE detector response threshold (lower -> more keypoints).

    Returns:
        Tuple (points (N, 2), descriptors (N, D) uint8), or None descriptors
        if no keypoint was found.
    """
    return _describe(img, AKAZE_create(threshold=threshold), max_kps, color, clahe)


def orb(img, max_kps=1000, color="gray", clahe=True):
    """
    Compute ORB features (binary descriptors, Hamming norm) of an image.

    Args:
        img: BGR image.
        max_kps: Maximum number of keypoints (also ORB's nfeatures).
        color: One of COLORS.
        clahe: Equalise the local contrast before detecting and describing.

    Returns:
        Tuple (points (N, 2), descriptors (N, 32 * n_channels) uint8), or None
        descriptors if no keypoint was found.
    """
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
