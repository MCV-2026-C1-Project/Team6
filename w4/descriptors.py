"""
descriptors.py - Local feature descriptors (NumPy + OpenCV).

    from descriptors import sift, orb, akaze
    desc = sift(img, kps)

All functions: (img, kps, **options) -> np.ndarray of shape (len(kps), D)
    img : grayscale or BGR image (uint8 or float)
    kps : list of cv2.KeyPoint; uses kp.pt, kp.size and kp.angle (OpenCV convention)
    float32 -> compare with L2 | uint8 (orb, akaze, latch) -> compare with Hamming
"""

import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ["sift", "rootsift", "surf", "orb", "lbp", "pca_sift", "color_sift",
           "gloh", "daisy", "akaze", "latch"]


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------

def _to_float(img):
    img = img.astype(np.float32)
    return img / 255.0 if img.max() > 1.5 else img


def _gray(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img if img.dtype == np.uint8 else img.astype(np.float32),
                           cv2.COLOR_BGR2GRAY)
    return _to_float(img)


def _patch(img, kp, size, support, angle=None):
    """size x size patch covering a square of side `support` px centered on kp and
    rotated by kp.angle (or `angle` in radians). Anti-aliased when downsampling."""
    k = int(np.clip(np.ceil(support / size), 1, 8))
    P = size * k
    if angle is None:
        angle = np.deg2rad(kp.angle) if kp.angle >= 0 else 0.0
    c, s = np.cos(angle), np.sin(angle)
    A = support / P * np.array([[c, -s], [s, c]])
    t = np.array(kp.pt) - A @ np.full(2, (P - 1) / 2.0)
    M = np.hstack([A, t[:, None]]).astype(np.float32)
    p = cv2.warpAffine(img, M, (P, P), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_REFLECT)
    return cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA) if k > 1 else p


def _patch_b(img, kp, P, support, angle=None):
    """Same as _patch but with a 1-px border (to compute gradients)."""
    return _patch(img, kp, P + 2, support * (P + 2) / P, angle)


def _grad(p):
    gy, gx = np.gradient(p)
    return gx[1:-1, 1:-1], gy[1:-1, 1:-1]


def _norm(v, clip=None, eps=1e-7):
    v = v / (np.linalg.norm(v) + eps)
    if clip is not None:
        v = np.minimum(v, clip)
        v = v / (np.linalg.norm(v) + eps)
    return v


def _gauss(P, sigma):
    r = np.arange(P) - (P - 1) / 2.0
    xx, yy = np.meshgrid(r, r)
    return np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))


# --------------------------------------------------------------------------------------
# SIFT family
# --------------------------------------------------------------------------------------

def _sift_channel(g, kps, n_cells=4, n_bins=8, magnif=3.0, cell_px=4):
    P = n_cells * cell_px
    coords = (np.arange(P) + 0.5) / cell_px - 0.5
    rx, ry = np.meshgrid(coords, coords)
    x0, y0 = np.floor(rx).astype(int), np.floor(ry).astype(int)
    fx, fy = rx - x0, ry - y0
    w = _gauss(P, P / 2.0)
    out = np.zeros((len(kps), n_cells * n_cells * n_bins), np.float32)
    for i, kp in enumerate(kps):
        gx, gy = _grad(_patch_b(g, kp, P, n_cells * magnif * kp.size / 2.0))
        mag = np.hypot(gx, gy) * w
        ro = (np.arctan2(gy, gx) % (2 * np.pi)) * n_bins / (2 * np.pi)
        o0 = np.floor(ro).astype(int)
        fo = ro - o0
        h = np.zeros((n_cells + 2, n_cells + 2, n_bins))
        for dx in (0, 1):                       # trilinear interpolation
            for dy in (0, 1):
                for do in (0, 1):
                    wt = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fo if do else 1 - fo)
                    np.add.at(h, (y0 + dy + 1, x0 + dx + 1, (o0 + do) % n_bins), mag * wt)
        out[i] = _norm(h[1:-1, 1:-1].ravel(), clip=0.2)
    return out


def sift(img, kps, n_cells=4, n_bins=8, magnif=3.0):
    """SIFT: 4x4 cells x 8 gradient-orientation bins = 128.
    Gaussian weighting, trilinear interpolation, normalize -> clip 0.2 -> renormalize."""
    return _sift_channel(_gray(img), kps, n_cells, n_bins, magnif)


def rootsift(img, kps, **kw):
    """RootSIFT: SIFT with L1 normalization + square root (L2 = Hellinger kernel). 128."""
    d = sift(img, kps, **kw)
    return np.sqrt(d / (d.sum(axis=1, keepdims=True) + 1e-7)).astype(np.float32)


def color_sift(img, kps, space="opponent", **kw):
    """Color-SIFT: SIFT on 3 color channels, concatenated (384).
    space: 'opponent' (OpponentSIFT), 'c' (C-SIFT), 'rgb', 'hsv', 'rg'."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bgr = _to_float(img)
    B, G, R = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    O1, O2, O3 = (R - G) / np.sqrt(2), (R + G - 2 * B) / np.sqrt(6), (R + G + B) / np.sqrt(3)
    if space == "opponent":
        chans = [O1, O2, O3]
    elif space == "c":
        chans = [O1 / (O3 + 1e-3), O2 / (O3 + 1e-3), O3]
    elif space == "rgb":
        chans = [R, G, B]
    elif space == "hsv":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        chans = [hsv[..., 0] / 360.0, hsv[..., 1], hsv[..., 2]]
    elif space == "rg":
        S = R + G + B + 1e-3
        chans = [R / S, G / S, S / 3]
    else:
        raise ValueError(space)
    return np.hstack([_sift_channel(c.astype(np.float32), kps, **kw) for c in chans])


def pca_sift(img, kps, pca=None, n_components=36, return_pca=False, magnif=3.0):
    """PCA-SIFT: x/y gradients of a 41x41 patch (3042 dims) projected with PCA (36).
    The same projection must be used for every image:
        d1, model = pca_sift(img1, kps1, return_pca=True)   # fit
        d2 = pca_sift(img2, kps2, pca=model)                # reuse"""
    g = _gray(img)
    raw = np.zeros((len(kps), 2 * 39 * 39), np.float32)
    for i, kp in enumerate(kps):
        p = _patch(g, kp, 41, 4 * magnif * kp.size / 2.0)
        gx, gy = p[1:-1, 2:] - p[1:-1, :-2], p[2:, 1:-1] - p[:-2, 1:-1]
        raw[i] = _norm(np.concatenate([gx.ravel(), gy.ravel()]))
    if pca is None:
        mean = raw.mean(axis=0)
        _, _, Vt = np.linalg.svd(raw - mean, full_matrices=False)
        pca = (mean, Vt[:n_components])
    d = ((raw - pca[0]) @ pca[1].T).astype(np.float32)
    return (d, pca) if return_pca else d


def gloh(img, kps, n_ori=16, radii=(6, 11, 15), n_sectors=8, magnif=3.0, P=32):
    """GLOH: log-polar grid (center + 2 rings x 8 sectors = 17) x 16 orientations = 272."""
    g = _gray(img)
    r = np.arange(P) - (P - 1) / 2.0
    xx, yy = np.meshgrid(r, r)
    rad = np.hypot(xx, yy) / (P / 2.0) * radii[-1]
    sector = np.minimum((np.arctan2(yy, xx) % (2 * np.pi) / (2 * np.pi) * n_sectors).astype(int),
                        n_sectors - 1)
    ring = np.where(rad < radii[0], 0, np.where(rad < radii[1], 1, 2))
    sbin = np.where(ring == 0, 0, 1 + (ring - 1) * n_sectors + sector)
    valid = rad < radii[-1]
    n_sp = 1 + 2 * n_sectors
    w = _gauss(P, P / 2.0)
    out = np.zeros((len(kps), n_sp * n_ori), np.float32)
    for i, kp in enumerate(kps):
        gx, gy = _grad(_patch_b(g, kp, P, 2 * magnif * kp.size))
        mag = np.hypot(gx, gy) * w
        ob = np.minimum((np.arctan2(gy, gx) % (2 * np.pi) / (2 * np.pi) * n_ori).astype(int), n_ori - 1)
        h = np.bincount((sbin * n_ori + ob)[valid], mag[valid], minlength=n_sp * n_ori)
        out[i] = _norm(h, clip=0.2)
    return out


# --------------------------------------------------------------------------------------
# SURF, DAISY
# --------------------------------------------------------------------------------------

def surf(img, kps, extended=False):
    """SURF: 20s window (s = 1.2*size/9), 4x4 subregions of 5x5 samples,
    [sum dx, sum dy, sum|dx|, sum|dy|] per subregion = 64 (128 if extended).
    Haar responses approximated by gradients of the resampled patch."""
    g = _gray(img)
    w = _gauss(20, 3.3)
    blk = lambda a: a.reshape(4, 5, 4, 5).sum(axis=(1, 3))
    out = np.zeros((len(kps), 128 if extended else 64), np.float32)
    for i, kp in enumerate(kps):
        dx, dy = _grad(_patch_b(g, kp, 20, 20 * 1.2 * kp.size / 9.0))
        dx, dy = dx * w, dy * w
        if not extended:
            f = [blk(dx), blk(dy), blk(np.abs(dx)), blk(np.abs(dy))]
        else:
            py, px = dy >= 0, dx >= 0
            f = [blk(dx * py), blk(dx * ~py), blk(np.abs(dx) * py), blk(np.abs(dx) * ~py),
                 blk(dy * px), blk(dy * ~px), blk(np.abs(dy) * px), blk(np.abs(dy) * ~px)]
        out[i] = _norm(np.stack(f, -1).ravel())
    return out


def daisy(img, kps, rings=3, ring_pts=8, n_ori=8, P=33, support_factor=6.0):
    """DAISY: oriented gradient maps max(<grad, u_o>, 0) smoothed with Gaussians of
    increasing sigma per ring, sampled at the center + rings x ring_pts points.
    Each histogram normalized separately: (1 + 3*8) x 8 = 200."""
    g = _gray(img)
    R = (P - 1) / 2.0
    oris = np.arange(n_ori) * 2 * np.pi / n_ori
    ring_r = [R * (k + 1) / rings for k in range(rings)]
    sigmas = [R / (2 * rings)] + [r / 2.0 for r in ring_r]
    samples = [(R, R, 0)] + [(R + r * np.cos(a), R + r * np.sin(a), k + 1)
                             for k, r in enumerate(ring_r)
                             for a in 2 * np.pi * np.arange(ring_pts) / ring_pts]
    idx = [(int(np.clip(round(y), 0, P - 1)), int(np.clip(round(x), 0, P - 1)), l)
           for x, y, l in samples]
    out = np.zeros((len(kps), len(samples) * n_ori), np.float32)
    for i, kp in enumerate(kps):
        gx, gy = _grad(_patch_b(g, kp, P, support_factor * kp.size))
        G = [np.maximum(gx * np.cos(o) + gy * np.sin(o), 0) for o in oris]
        conv = [np.stack([cv2.GaussianBlur(m, (0, 0), s, borderType=cv2.BORDER_REFLECT) for m in G], -1)
                for s in sigmas]
        out[i] = np.concatenate([_norm(conv[l][y, x]) for y, x, l in idx])
    return out


# --------------------------------------------------------------------------------------
# LBP
# --------------------------------------------------------------------------------------

def _uniform_table():
    t, nxt = np.full(256, 58, np.int32), 0
    for c in range(256):
        b = [(c >> k) & 1 for k in range(8)]
        if sum(b[k] != b[(k + 1) % 8] for k in range(8)) <= 2:
            t[c], nxt = nxt, nxt + 1
    return t


_ULBP = _uniform_table()
_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def lbp(img, kps, P=32, grid=4, uniform=True, support_factor=6.0):
    """LBP: 8-neighbor binary code per pixel, one histogram per cell of a grid x grid layout.
    uniform=True -> 59 bins/cell (4x4 x 59 = 944); otherwise 256 bins/cell."""
    g = _gray(img)
    nb = 59 if uniform else 256
    cell = np.arange(P) // (P // grid)
    cid = cell[:, None] * grid + cell[None, :]
    out = np.zeros((len(kps), grid * grid * nb), np.float32)
    for i, kp in enumerate(kps):
        p = _patch_b(g, kp, P, support_factor * kp.size)
        c = p[1:-1, 1:-1]
        code = np.zeros((P, P), np.int32)
        for k, (dy, dx) in enumerate(_N8):
            code |= (p[1 + dy:P + 1 + dy, 1 + dx:P + 1 + dx] >= c).astype(np.int32) << k
        if uniform:
            code = _ULBP[code]
        out[i] = _norm(np.bincount((cid * nb + code).ravel(), minlength=grid * grid * nb).astype(float))
    return out


# --------------------------------------------------------------------------------------
# Binary: ORB, LATCH, AKAZE
# --------------------------------------------------------------------------------------

def orb(img, kps, n_bits=256, patch_size=31, size_factor=1.0, seed=0):
    """ORB: orientation from the intensity centroid + rotated BRIEF
    (n_bits comparisons I(p) < I(q) on a smoothed patch). 32 bytes.
    Region = size_factor * kp.size (with ORB keypoints, size is already 31).
    Simplification: random Gaussian pairs instead of the learned ones."""
    g = _gray(img)
    S = patch_size
    pr = np.clip(np.round(np.random.default_rng(seed).normal(0, S / 5.0, (n_bits, 4))),
                 -(S // 2), S // 2).astype(int) + S // 2
    r = np.arange(S) - (S - 1) / 2.0
    xx, yy = np.meshgrid(r, r)
    disk = xx ** 2 + yy ** 2 <= (S / 2.0) ** 2
    bits = np.zeros((len(kps), n_bits), bool)
    for i, kp in enumerate(kps):
        support = max(size_factor * kp.size, 1.0)
        p0 = _patch(g, kp, S, support, angle=0.0)
        theta = np.arctan2((yy * p0 * disk).sum(), (xx * p0 * disk).sum())
        p = cv2.GaussianBlur(_patch(g, kp, S, support, angle=theta), (9, 9), 2,
                             borderType=cv2.BORDER_REFLECT)
        bits[i] = p[pr[:, 1], pr[:, 0]] < p[pr[:, 3], pr[:, 2]]
    return np.packbits(bits, axis=1)


def latch(img, kps, n_bits=256, window=48, half_ssd=3, support_factor=6.0, sigma=1.0, seed=0):
    """LATCH: compares triplets of 7x7 patches (anchor a, p1, p2) inside a 48x48 window:
    bit = SSD(a, p1) > SSD(a, p2). 32 bytes.
    Simplification: random triplets instead of the learned ones."""
    g = _gray(img)
    k = 2 * half_ssd + 1
    tr = np.random.default_rng(seed).integers(0, window - k + 1, size=(n_bits, 3, 2))
    bits = np.zeros((len(kps), n_bits), bool)
    for i, kp in enumerate(kps):
        p = _patch(g, kp, window, max(support_factor * kp.size, 1.0))
        if sigma > 0:
            p = cv2.GaussianBlur(p, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        win = sliding_window_view(p, (k, k))
        A, P1, P2 = (win[tr[:, j, 1], tr[:, j, 0]] for j in range(3))
        bits[i] = ((A - P1) ** 2).sum(axis=(1, 2)) > ((A - P2) ** 2).sum(axis=(1, 2))
    return np.packbits(bits, axis=1)


def _diffuse(L, t, dt=0.25, k_percentile=70):
    """Perona-Malik nonlinear diffusion (c = 1 / (1 + |grad|^2/k^2)), explicit scheme."""
    L = L.astype(np.float32).copy()
    gy, gx = np.gradient(cv2.GaussianBlur(L, (0, 0), 1.0))
    mag = np.hypot(gx, gy)
    k = float(np.percentile(mag[mag > 0], k_percentile)) if np.any(mag > 0) else 1.0
    n = max(1, int(np.ceil(t / dt)))
    for _ in range(n):
        gy, gx = np.gradient(cv2.GaussianBlur(L, (0, 0), 1.0))
        c = 1.0 / (1.0 + (gx ** 2 + gy ** 2) / k ** 2)
        Lp, cp = np.pad(L, 1, mode="edge"), np.pad(c, 1, mode="edge")
        flux = sum(0.5 * (cp[a] + c) * (Lp[a] - L) for a in
                   [(slice(None, -2), slice(1, -1)), (slice(2, None), slice(1, -1)),
                    (slice(1, -1), slice(None, -2)), (slice(1, -1), slice(2, None))])
        L += (t / n) * flux
    return L


def akaze(img, kps, grids=(2, 3, 4), P=24, window=20.0):
    """AKAZE (M-LDB) over a patch filtered with nonlinear diffusion.
    Grids of 2x2, 3x3 and 4x4 cells; per cell (mean intensity, mean dx, mean dy);
    compares every pair of cells in the same grid: 3*(6+36+120) = 486 bits (61 bytes).
    s = size/2, window = 20s."""
    g = _gray(img)
    pairs = [(n, *np.triu_indices(n * n, 1)) for n in grids]
    nbits = sum(3 * len(I) for _, I, _ in pairs)
    out = np.zeros((len(kps), nbits), bool)
    for i, kp in enumerate(kps):
        support = window * kp.size / 2.0
        p = _patch(g, kp, 2 * (P + 2), support * (P + 2) / P)    # 2x oversampled
        p = cv2.resize(_diffuse(p, 2.0), (P + 2, P + 2), interpolation=cv2.INTER_AREA)
        dx, dy = _grad(p)
        L = p[1:-1, 1:-1]
        bits = []
        for n, I, J in pairs:
            c = P // n
            f = np.stack([a[:c * n, :c * n].reshape(n, c, n, c).mean(axis=(1, 3)).ravel()
                          for a in (L, dx, dy)], -1)
            bits.append((f[I] > f[J]).ravel())
        out[i] = np.concatenate(bits)
    return np.packbits(out, axis=1)