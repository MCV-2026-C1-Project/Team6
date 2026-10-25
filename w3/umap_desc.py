from utils import *
from pathlib import Path
import umap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"

PARAMS = {
    "descriptor": dct_pyramid_descriptor,
    "color_code": cv2.COLOR_BGR2YCrCb,
    "channels": [0, 1, 2],
    "size": 672,
    "metric": "canberra",
    "kwargs": {
        "levels": [1, 2, 3],
        "n_coef": 32,
        "skip_dc": False,
    },
}

UMAP_METRIC = "cosine"
SEED = 42
THUMB_SIZE = 40      # costat més llarg de cada miniatura, en píxels
SHOW_IDS = False     # True per escriure també l'ID sota cada quadre


def thumbnail(img_bgr, size):
    """Miniatura RGB amb el costat més llarg igual a size."""
    h, w = img_bgr.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    db_ids, db_images = load_images(BBDD_DIR)

    print("Computing BBDD descriptors...")
    db_descs = np.vstack([describe(img, PARAMS) for img in tqdm(db_images, desc="BBDD")])

    print("Running UMAP...")
    emb = umap.UMAP(n_neighbors=100, min_dist=0.01, metric=UMAP_METRIC, random_state=SEED).fit_transform(db_descs)

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.scatter(emb[:, 0], emb[:, 1], s=1, alpha=0)  # fixa els límits dels eixos

    for (x, y), img, i in zip(emb, db_images, db_ids):
        box = AnnotationBbox(OffsetImage(thumbnail(img, THUMB_SIZE), zoom=1), (x, y), frameon=False)
        ax.add_artist(box)
        if SHOW_IDS:
            ax.annotate(str(i), (x, y), xytext=(0, -THUMB_SIZE / 2 - 4), textcoords="offset points",
                        ha="center", fontsize=6)

    ax.set_title(f"UMAP of BBDD ({len(db_ids)} paintings) - DCT pyramid {PARAMS['kwargs']['levels']}, "
                 f"YCrCb, {db_descs.shape[1]} dims, {UMAP_METRIC}")
    ax.margins(0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()