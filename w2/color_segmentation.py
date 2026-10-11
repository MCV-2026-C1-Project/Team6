from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

L_WEIGHT = 0.4  # down-weight lightness: walls change in L (shadows), not in a/b


def load_images(folder):
    """Load JPGs as 8-bit Lab. Returns (ids, images)."""
    ids, images = [], []
    for path in sorted(Path(folder).glob("*.jpg")):
        img = cv2.imread(str(path))
        if img is None:
            print(f"Could not read {path}")
            continue
        ids.append(int(path.stem.split("_")[-1]))
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    return ids, images


def _corners(img, s=20):
    h, w = img.shape[:2]
    s = max(1, min(s, h // 2, w // 2))
    return np.concatenate([p.reshape(-1, 3) for p in
                           (img[:s, :s], img[:s, w-s:], img[h-s:, :s], img[h-s:, w-s:])])


def _clean(mask, k=5):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)


def _largest_filled(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return out


def segment_painting(img, wall_thr=15.0, frame_thr=3.0, strip_frac=0.03,
                     proj_frac=0.5, min_ratio=0.25, max_ratio=0.98):
    """
    Returns a mask (255) of the painting WITHOUT its frame.
    1) Remove the wall (distance to median corner color) -> painting + frame.
    2) Model the frame from thin strips along that region's bounding-box edges;
       pixels that don't fit the frame are painting. Painting rectangle comes
       from row/column projections.
    Falls back to painting + frame if no sensible inner rectangle is found.
    """
    h, w = img.shape[:2]

    # Stage 1: painting + frame
    bg = np.median(_corners(img).astype(np.float32), axis=0)
    diff = (img.astype(np.float32) - bg) * np.array([L_WEIGHT, 1, 1], np.float32)
    outer = _largest_filled(_clean((np.linalg.norm(diff, axis=2) > wall_thr).astype(np.uint8) * 255))
    if not outer.any():
        return outer

    x, y, bw, bh = cv2.boundingRect(outer)
    crop, in_outer = img[y:y+bh, x:x+bw], outer[y:y+bh, x:x+bw] > 0

    # Stage 2: frame color model from the edge strips
    s = max(2, int(round(strip_frac * min(bw, bh))))
    strips = np.zeros((bh, bw), bool)
    strips[:s] = strips[-s:] = True
    strips[:, :s] = strips[:, -s:] = True
    samples = crop[strips & in_outer].astype(np.float64)
    if len(samples) < 50:
        return outer

    cov_inv = np.linalg.pinv(np.cov(samples, rowvar=False) + np.eye(3))
    d = crop.astype(np.float64) - samples.mean(axis=0)
    dist = np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", d, cov_inv, d), 0))
    inner = _clean(((dist > frame_thr) & in_outer).astype(np.uint8) * 255) > 0

    cols = np.flatnonzero(inner.mean(axis=0) > proj_frac)
    rows = np.flatnonzero(inner.mean(axis=1) > proj_frac)
    if not cols.size or not rows.size:
        return outer
    x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    if not (min_ratio <= (x1 - x0) * (y1 - y0) / (bw * bh) <= max_ratio):
        return outer

    mask = np.zeros((h, w), np.uint8)
    mask[y+y0:y+y1, x+x0:x+x1] = 255
    return mask


def show(img, mask, title=""):
    rgb = lambda a: cv2.cvtColor(cv2.cvtColor(a, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    fg = img.copy()
    fg[mask == 0] = 0
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, im, t in zip(ax, (rgb(img), mask, rgb(fg)), ("Original", "Mask", "Painting only")):
        a.imshow(im, cmap="gray" if im.ndim == 2 else None)
        a.set_title(t)
        a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = Path(__file__).resolve().parent.parent / "data" / "qsd2_w2"
    out = Path("qs2_masks")
    out.mkdir(exist_ok=True)

    ids, images = load_images(data)
    for i, img in zip(ids, images):
        cv2.imwrite(str(out / f"{i}_mask.png"), segment_painting(img))

    if images:
        show(images[0], segment_painting(images[0]), f"Image {ids[0]}")