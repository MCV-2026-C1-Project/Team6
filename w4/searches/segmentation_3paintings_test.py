"""
Synthetic test of the detection of up to 3 paintings per image.

None of our query sets has an image with 3 paintings, so this script builds
them from real data and evaluates segmentation.detect_paintings on them:

    wall       a real query (QSD1-W4 or QSD2-W3) with its paintings removed: the
               ground-truth mask, dilated to also remove frame shadows, is filled
               with a smooth surface fitted to the wall (2nd-order polynomial per
               channel) plus noise like the wall's
    paintings  museum images (BBDD) with a frame (70%), sometimes a white mat,
               a small rotation (+-2 deg) and a drop shadow
    layouts    1 painting: single
               2 paintings: row (side by side), column (stacked)
               3 paintings: row, column, L (2 stacked next to 1), T (1 above/below 2)
    gaps       2-8% of the smallest side between paintings (small gaps test the split)
    JPEG       quality 80-95

The 1 and 2 painting images are a control: comparing their scores with the real
query sets shows how hard the synthetic images are.

The images are written to data/synthetic_w4/ like a query set (00000.jpg and the
00000.png ground-truth mask, plus gt.pkl with the layout and the painting masks),
so visualizations/segmentation_steps.ipynb can show them too.

    python searches/segmentation_3paintings_test.py          (from w4/)

Reported per number of paintings and per layout: images with the right count,
pixel F1 of the union mask, paintings found (IoU > 0.5 with a detection) and
images whose paintings are found in the expected order (segmentation.order_masks
rule applied to the ground truth). Per-image results go to
searches/results/segmentation_3paintings.csv and examples to
searches/results/segmentation_3paintings_examples.png.
"""

import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

W4 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W4))
import segmentation as S
from denoising import denoise_image

DATA = W4.parent / "data"
SOURCES = [DATA / "qsd1_w4", DATA / "qsd2_w3"]
OUT_DIR = DATA / "synthetic_w4"
RESULTS = W4 / "searches" / "results"
SEED = 0

LAYOUTS = {  # name: (number of paintings, number of images)
    "single": (1, 10),
    "row2": (2, 10),
    "column2": (2, 10),
    "row3": (3, 15),
    "column3": (3, 15),
    "L3": (3, 15),
    "T3": (3, 15),
}
CANVAS = {"row3": (640, 1024), "column3": (1024, 700)}   # (h, w); others: random landscape/portrait
FRAME_COLORS = [(30, 60, 100), (90, 150, 200), (40, 160, 200), (25, 25, 25), (235, 235, 235), (175, 175, 175)]


# =============================================================================
# WALLS
# =============================================================================

def clean_wall(img: np.ndarray, gt: np.ndarray, rng) -> np.ndarray:
    """Remove the paintings of a real query: fill them with a smooth fit of the wall plus noise."""
    h, w = gt.shape
    k = max(3, int(0.04 * min(h, w)))
    hole = cv2.dilate(gt.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))) > 0

    ys, xs = np.nonzero(~hole)
    pick = rng.choice(len(ys), min(20000, len(ys)), replace=False)
    ys, xs = ys[pick], xs[pick]
    design = lambda x, y: np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)
    A = design(xs / w, ys / h)
    gy, gx = np.mgrid[0:h, 0:w]
    G = design(gx / w, gy / h)

    fill = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        v = img[ys, xs, c].astype(np.float64)
        coef = np.linalg.lstsq(A, v, rcond=None)[0]
        res = v - A @ coef
        sigma = min(6.0, 1.4826 * np.median(np.abs(res - np.median(res))))
        noise = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.0) * sigma * 2
        fill[..., c] = G @ coef + noise

    alpha = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 5)[..., None]
    out = alpha * fill + (1 - alpha) * img.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_walls(rng) -> list[np.ndarray]:
    """Real walls of the queries with at least half of the image being wall."""
    walls = []
    for folder in SOURCES:
        for path in sorted(folder.glob("*.jpg")):
            img = S.resize_max_side(denoise_image(cv2.imread(str(path))))
            gt = cv2.imread(str(path.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            if gt.mean() <= 0.5:
                walls.append(clean_wall(img, gt, rng))
    return walls


# =============================================================================
# PAINTINGS
# =============================================================================

def framed(painting: np.ndarray, rng) -> np.ndarray:
    """Add a mat (30%) and a frame (70%) around a museum image. The frame is part of the painting."""
    out = painting
    s = min(painting.shape[:2])
    if rng.random() < 0.3:
        t = int(rng.uniform(0.05, 0.12) * s)
        out = cv2.copyMakeBorder(out, t, t, t, t, cv2.BORDER_CONSTANT, value=(240, 242, 245))
    if rng.random() < 0.7:
        t = max(2, int(rng.uniform(0.03, 0.08) * s))
        color = np.array(FRAME_COLORS[rng.integers(len(FRAME_COLORS))], np.float64)
        out = cv2.copyMakeBorder(out, t, t, t, t, cv2.BORDER_CONSTANT, value=color.tolist())
        half = max(1, t // 2)  # outer half of the frame a bit darker (bevel)
        ring = np.zeros(out.shape[:2], bool)
        ring[:half], ring[-half:], ring[:, :half], ring[:, -half:] = True, True, True, True
        out[ring] = (out[ring] * 0.8).astype(np.uint8)
    return out


def rotated(tile: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a tile, enlarging the canvas. Returns (tile, alpha in [0, 1])."""
    h, w = tile.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(tile, M, (nw, nh), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpAffine(np.ones((h, w), np.float32), M, (nw, nh), flags=cv2.INTER_LINEAR)
    return out, alpha


# =============================================================================
# LAYOUTS
# =============================================================================

def cells(layout: str, h: int, w: int, rng) -> list[tuple[int, int, int, int]]:
    """Cells (x0, y0, x1, y1) where the paintings go, with a margin to the border and gaps between them."""
    m = rng.uniform(0.06, 0.12) * min(h, w)
    g = rng.uniform(0.02, 0.08) * min(h, w)
    X0, Y0, X1, Y1 = m, m, w - m, h - m

    def split(a, b, n, fracs=None):
        fracs = np.full(n, 1 / n) if fracs is None else np.asarray(fracs)
        size = (b - a - g * (n - 1))
        edges, pos = [], a
        for f in fracs:
            edges.append((pos, pos + f * size))
            pos += f * size + g
        return edges

    if layout == "single":
        boxes = [(X0, Y0, X1, Y1)]
    elif layout in ("row2", "row3"):
        boxes = [(a, Y0, b, Y1) for a, b in split(X0, X1, int(layout[-1]))]
    elif layout in ("column2", "column3"):
        boxes = [(X0, a, X1, b) for a, b in split(Y0, Y1, int(layout[-1]))]
    elif layout == "L3":  # two stacked paintings next to a third one (stack on the left or right)
        r = rng.uniform(0.4, 0.6)
        (la, lb), (ra, rb) = split(X0, X1, 2, [r, 1 - r])
        stack, single = ((la, lb), (ra, rb)) if rng.random() < 0.5 else ((ra, rb), (la, lb))
        boxes = [(stack[0], a, stack[1], b) for a, b in split(Y0, Y1, 2)] + [(single[0], Y0, single[1], Y1)]
    elif layout == "T3":  # one painting above two (or below)
        r = rng.uniform(0.4, 0.6)
        (ta, tb), (ba, bb) = split(Y0, Y1, 2, [r, 1 - r])
        one, two = ((ta, tb), (ba, bb)) if rng.random() < 0.5 else ((ba, bb), (ta, tb))
        boxes = [(X0, one[0], X1, one[1])] + [(a, two[0], b, two[1]) for a, b in split(X0, X1, 2)]
    else:
        raise ValueError(layout)
    return [tuple(int(round(v)) for v in box) for box in boxes]


def compose(wall: np.ndarray, museum: list[np.ndarray], layout: str, rng):
    """Paste framed museum paintings on a wall. Returns (image, list of painting masks)."""
    img = wall.astype(np.float32)
    h, w = wall.shape[:2]
    masks = []
    for (x0, y0, x1, y1), painting in zip(cells(layout, h, w, rng), museum):
        tile, alpha = rotated(framed(painting, rng), rng.uniform(-2, 2))
        cw, ch = x1 - x0, y1 - y0
        s = min(cw / tile.shape[1], ch / tile.shape[0]) * rng.uniform(0.8, 1.0)
        tile = cv2.resize(tile, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_LINEAR)
        th, tw = alpha.shape
        px = x0 + int(rng.uniform(0, max(cw - tw, 0) + 1))
        py = y0 + int(rng.uniform(0, max(ch - th, 0) + 1))

        canvas_alpha = np.zeros((h, w), np.float32)
        canvas_alpha[py:py + th, px:px + tw] = alpha[:h - py, :w - px]

        # drop shadow: the painting's silhouette shifted down (and a bit sideways), blurred, darkens the wall
        d = min(h, w)
        dx, dy = int(rng.uniform(-0.01, 0.015) * d), int(rng.uniform(0.005, 0.02) * d)
        shadow = np.roll(np.roll(canvas_alpha, dy, axis=0), dx, axis=1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), max(1.0, 0.01 * d))
        img *= (1 - rng.uniform(0.15, 0.35) * shadow)[..., None]

        tile_full = np.zeros((h, w, 3), np.float32)
        tile_full[py:py + th, px:px + tw] = tile[:h - py, :w - px]
        img = canvas_alpha[..., None] * tile_full + (1 - canvas_alpha[..., None]) * img
        masks.append((canvas_alpha > 0.5).astype(np.uint8))
    return np.clip(img, 0, 255).astype(np.uint8), masks


def build(rng) -> list[dict]:
    """Create and save the synthetic query set."""
    walls = load_walls(rng)
    museum_paths = sorted((DATA / "BBDD").glob("*.jpg"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items, idx = [], 0
    for layout, (n, count) in LAYOUTS.items():
        for _ in range(count):
            wall = walls[rng.integers(len(walls))]
            if layout in CANVAS:
                size = CANVAS[layout]
            else:
                size = (700, 1024) if rng.random() < 0.6 else (1024, 760)
            wall = cv2.resize(wall, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
            museum = [S.resize_max_side(cv2.imread(str(museum_paths[i])), 700)
                      for i in rng.choice(len(museum_paths), n, replace=False)]
            img, masks = compose(wall, museum, layout, rng)
            masks = S.order_masks(masks)   # expected order: the rule of segmentation.py on the ground truth

            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(80, 96))])
            (OUT_DIR / f"{idx:05d}.jpg").write_bytes(jpg.tobytes())
            union = np.clip(sum(masks), 0, 1).astype(np.uint8) * 255
            cv2.imwrite(str(OUT_DIR / f"{idx:05d}.png"), union)
            items.append({"id": idx, "layout": layout, "n": n, "masks": masks})
            idx += 1
    with open(OUT_DIR / "gt.pkl", "wb") as f:
        pickle.dump([{"layout": it["layout"], "n": it["n"], "masks": it["masks"]} for it in items], f)
    return items


# =============================================================================
# EVALUATION
# =============================================================================

def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 0.0


def evaluate(item: dict) -> dict:
    img = denoise_image(cv2.imread(str(OUT_DIR / f"{item['id']:05d}.jpg")))
    pred = S.detect_paintings(img)
    gt = item["masks"]

    union_gt = np.clip(sum(gt), 0, 1) > 0
    union_pred = np.zeros(union_gt.shape, bool)
    for m in pred:
        union_pred |= m > 0
    tp = np.count_nonzero(union_pred & union_gt)
    p = tp / max(np.count_nonzero(union_pred), 1)
    r = tp / max(np.count_nonzero(union_gt), 1)

    found = sum(any(iou(g, m) > 0.5 for m in pred) for g in gt)
    in_order = len(pred) == len(gt) and all(iou(g, m) > 0.5 for g, m in zip(gt, pred))
    return {"id": item["id"], "layout": item["layout"], "n_gt": len(gt), "n_pred": len(pred),
            "count_ok": len(pred) == len(gt), "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "precision": p, "recall": r, "found": found, "in_order": in_order}


def summary(rows: list[dict], key: str) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    print(f"{key:>8} {'images':>6} {'count ok':>9} {'F1':>7} {'P':>7} {'R':>7} {'found':>11} {'order ok':>9}")
    for g, rs in groups.items():
        n_gt = sum(r["n_gt"] for r in rs)
        print(f"{g!s:>8} {len(rs):>6} {sum(r['count_ok'] for r in rs):>5}/{len(rs):<3} "
              f"{np.mean([r['f1'] for r in rs]):>7.4f} {np.mean([r['precision'] for r in rs]):>7.3f} "
              f"{np.mean([r['recall'] for r in rs]):>7.3f} {sum(r['found'] for r in rs):>5}/{n_gt:<5} "
              f"{sum(r['in_order'] for r in rs):>5}/{len(rs):<3}")


def examples(items: list[dict], rows: list[dict], path: Path, per_layout: int = 2) -> None:
    """Figure: ground truth (green) and detections (numbered, coloured) for some 3-painting images."""
    colors = [(0, 0, 255), (255, 0, 0), (0, 200, 255)]
    tiles = []
    by_id = {r["id"]: r for r in rows}
    for layout in ("row3", "column3", "L3", "T3"):
        chosen = [it for it in items if it["layout"] == layout]
        failed = [it for it in chosen if not by_id[it["id"]]["in_order"]]
        for it in (failed + [c for c in chosen if c not in failed])[:per_layout]:
            img = cv2.imread(str(OUT_DIR / f"{it['id']:05d}.jpg"))
            pred = S.detect_paintings(denoise_image(img))
            for g in it["masks"]:
                cs, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, (0, 255, 0), 6)
            for k, m in enumerate(pred):
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, colors[k % 3], 3)
                x, y, w, h = cv2.boundingRect(m.astype(np.uint8))
                cv2.putText(img, str(k + 1), (x + w // 2 - 15, y + h // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
            r = by_id[it["id"]]
            tile = cv2.resize(img, (512, 400))
            label = f"#{it['id']} {layout}: {r['n_pred']}/{r['n_gt']} F1 {r['f1']:.2f} {'ok' if r['in_order'] else 'FAIL'}"
            cv2.rectangle(tile, (0, 0), (512, 34), (0, 0, 0), -1)
            cv2.putText(tile, label, (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(tile)
    rows_img = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), np.vstack(rows_img))


def main() -> None:
    rng = np.random.default_rng(SEED)
    items = build(rng)
    print(f"{len(items)} synthetic images written to {OUT_DIR}")

    rows = [evaluate(it) for it in items]
    print("\nPer number of paintings:")
    summary(rows, "n_gt")
    print("\nPer layout:")
    summary(rows, "layout")
    bad = [r for r in rows if not r["in_order"]]
    print("\nFailures:", ", ".join(f"#{r['id']} {r['layout']} ({r['n_pred']}/{r['n_gt']} found {r['found']})" for r in bad) or "none")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "segmentation_3paintings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    examples(items, rows, RESULTS / "segmentation_3paintings_examples.png")
    print(f"Results saved under {RESULTS}")


if __name__ == "__main__":
    main()
