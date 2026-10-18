from utils import *
import pickle


# Dataset and output paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QST1 = DATA_DIR / "qst1_w2"
QST2 = DATA_DIR / "qst2_w2"
OUT_DIR = Path(__file__).resolve().parent.parent / "w1" / "results"

METRIC = "histogram_intersection"
K = 10

if __name__ == "__main__":
    print("Loading images...")
    db_ids, db_images = load_images(BBDD_DIR)

    descriptor_params = BEST_PARAMS["2d_descriptor"]
    segmentation_params = BEST_PARAMS["segmentation"]

    # Segment database images once
    print("Segmenting database images...")
    db_masks = segments(db_images, **segmentation_params)

    # Compute masked database descriptors once
    print("Extracting database descriptors...")
    db_desc = color_light_block_descriptors(
        db_images,
        masks=db_masks,
        **descriptor_params
    )

    for query_dir in [QST1, QST2]:
        print(f"\nProcessing {query_dir.name}...")

        query_ids, query_images = load_images(query_dir)

        # Segment query images
        print("Segmenting query images...")
        query_masks = segments(query_images, **segmentation_params)

        # Compute masked query descriptors
        print("Extracting query descriptors...")
        query_desc = color_light_block_descriptors(
            query_images,
            masks=query_masks,
            **descriptor_params
        )

        # Compute distances
        dists = get_distances(
            query_desc,
            db_desc,
            metric=METRIC
        )

        # Retrieve top-10
        results = retrieve(dists, db_ids, k=10)

        print(f"Results for {query_dir.name}:")
        print(results)

        # Save retrieval results
        out_dir = OUT_DIR / query_dir.name / "method1"
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "result.pkl", "wb") as f:
            pickle.dump(results, f)

        print(f"Results saved to {out_dir / 'result.pkl'}")

        # QST2: save segmentation masks
        if query_dir == QST2:
            for query_id, mask in zip(query_ids, query_masks):
                mask_path = out_dir / f"{query_id}.png"
                cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

            print(f"Masks saved to {out_dir}")