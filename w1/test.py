from utils import *
import pickle


# Dataset and output paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QST1 = DATA_DIR / "qst1_w1"
OUT_DIR = Path(__file__).resolve().parent.parent / "w1" / "results"

METRIC = "histogram_intersection"
K = 10


if __name__ == "__main__":
    # Load database and query images.
    print("Loading images...")
    db_ids, db_images = load_images(BBDD_DIR)
    query_ids, query_images = load_images(QST1)

    # Compute descriptors and distances for both methods.
    print("Extracting descriptors and computing distances...")
    db_m1 = simple_descriptors(db_images)
    query_m1 = simple_descriptors(query_images)
    db_m2 = complex_descriptors(db_images)
    query_m2 = complex_descriptors(query_images)

    dists_m1 = get_distances(query_m1, db_m1, metric=METRIC)
    dists_m2 = get_distances(query_m2, db_m2, metric=METRIC)

    # Retrieve the top-K results for each method and save them to disk.
    print(f"Retrieving top-{K} results and saving...")
    for method, dists in [("method1", dists_m1), ("method2", dists_m2)]:
        results = retrieve(dists, db_ids, k=K)
        print(f"{method}: {results}")

        # Create the output directory and save the retrieval results.
        out_path = OUT_DIR / method / "result.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "wb") as f:
            pickle.dump(results, f)

        print(f"Results saved to {out_path}")