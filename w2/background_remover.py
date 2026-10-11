"""
Color-only background removal for painting images (QS2).
No contour/edge/object detectors: only color statistics + projections.

Pipeline
1. Preprocess: Gaussian blur (kills texture/noise) -> Lab color space.
2. Model the background from a 10 px strip around the image border
   (robust median + covariance, so the wall's soft shadows are included).
3. Mahalanobis distance of every pixel to that background model.
4. Threshold (Otsu on the distance map) -> raw foreground mask.
5. Morphological clean-up.
6. Force a rectangle: row/column projections of the mask -> bounding box
   (robust to the museum label, shadows, drips, etc.).
7. Evaluate against ground truth (precision / recall / F1).
"""
import glob
import os

import cv2
import numpy as np

BORDER = 10          # px strip used to model the background
BLUR_KSIZE = 7       # preprocessing blur
PROJ_FRAC = 0.5      # a row/col belongs to the painting if >50% of it is foreground


def border_pixels(lab, b=BORDER):
    h, w = lab.shape[:2]
    strips = [lab[:b].reshape(-1, 3), lab[h - b:].reshape(-1, 3),
              lab[b:h - b, :b].reshape(-1, 3), lab[b:h - b, w - b:].reshape(-1, 3)]
    return np.vstack(strips).astype(np.float32)


def background_distance(lab):
    """Mahalanobis distance to the background color distribution."""
    px = border_pixels(lab)
    med = np.median(px, axis=0)
    # robust covariance: ignore border outliers (labels, frame shadows)
    d = np.linalg.norm(px - med, axis=1)
    inl = px[d < np.percentile(d, 90)]
    cov = np.cov(inl.T) + np.eye(3) * 1.0          # regularize
    icov = np.linalg.inv(cov)
    diff = lab.reshape(-1, 3).astype(np.float32) - med
    m2 = np.einsum("ij,jk,ik->i", diff, icov, diff)
    return np.sqrt(np.maximum(m2, 0)).reshape(lab.shape[:2])


def rectangle_from_mask(mask):
    """Bounding rectangle from row/column projections (no contours)."""
    cols = mask.mean(axis=0) > PROJ_FRAC
    rows = mask.mean(axis=1) > PROJ_FRAC
    if not cols.any() or not rows.any():
        return np.ones_like(mask)
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0, y1 = np.where(rows)[0][[0, -1]]
    rect = np.zeros_like(mask)
    rect[y0:y1 + 1, x0:x1 + 1] = 1
    return rect


def segment(img_bgr, return_debug=False):
    blur = cv2.GaussianBlur(img_bgr, (BLUR_KSIZE, BLUR_KSIZE), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)

    dist = background_distance(lab)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, fg = cv2.threshold(dist_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)   # fill holes inside the painting
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)    # remove label / specks

    mask = rectangle_from_mask(fg)
    return (mask, dist_u8, fg) if return_debug else mask


def evaluate(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return p, r, 2 * p * r / (p + r + 1e-9)


if __name__ == "__main__":
    IMG_DIR, GT_DIR, OUT_DIR = "data/qsd2_w2", "data/qsd2_w2", "qs2_masks"   # adjust
    os.makedirs(OUT_DIR, exist_ok=True)
    scores = []
    for path in sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg"))):
        name = os.path.splitext(os.path.basename(path))[0]
        mask = segment(cv2.imread(path))
        cv2.imwrite(os.path.join(OUT_DIR, name + ".png"), mask * 255)
        gt_path = os.path.join(GT_DIR, name + ".png")
        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path, 0) > 127
            scores.append(evaluate(mask, gt))
    if scores:
        p, r, f = np.mean(scores, axis=0)
        print(f"Precision {p:.3f}  Recall {r:.3f}  F1 {f:.3f}")