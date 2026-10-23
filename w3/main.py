"""Week 3 pipeline template for Team 6.

This file owns the overall pipeline. The task methods belong in w3/utils.py.
Until the team methods are implemented, PIPELINE_READY stays False and this
script only checks/loads the development data.
"""

from pathlib import Path
import pickle

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Paths and shared settings
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

BBDD_DIR = DATA_DIR / "BBDD"
QSD1_DIR = DATA_DIR / "qsd1_w3"
QSD2_DIR = DATA_DIR / "qsd2_w3"
QST1_DIR = DATA_DIR / "qst1_w3"
QST2_DIR = DATA_DIR / "qst2_w3"

RESULTS_DIR = Path(__file__).resolve().parent / "results"

K_DEV = 5
K_TEST = 10
METRIC = None  # Set after the team selects a suitable taught distance metric.
APPLY_DENOISING = False  # Set after comparing development-set PSNR results.
PIPELINE_READY = False  # Change to True once the utils.py interface is implemented.


# -----------------------------------------------------------------------------
# Data loading shared by the whole pipeline
# -----------------------------------------------------------------------------
def load_images_bgr(folder: Path) -> tuple[list[int], list[np.ndarray]]:
    """Load JPG images as BGR arrays and derive IDs from their filenames."""
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    paths = sorted(folder.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No JPG images found in: {folder}")

    ids: list[int] = []
    images: list[np.ndarray] = []

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {path}")

        image_id = int(path.stem.split("_")[-1])
        ids.append(image_id)
        images.append(image)

    return ids, images


def load_ground_truth(folder: Path) -> list[list[int]]:
    """Load the development correspondences from gt_corresps.pkl."""
    gt_path = folder / "gt_corresps.pkl"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    with open(gt_path, "rb") as file:
        ground_truth = pickle.load(file)

    if not isinstance(ground_truth, list):
        raise TypeError(
            f"Expected a list in {gt_path}, got {type(ground_truth).__name__}"
        )

    return ground_truth


def describe_dataset(name: str, ids: list[int], images: list[np.ndarray]) -> None:
    """Print a short check that image IDs and image arrays loaded correctly."""
    print(f"{name}: {len(images)} images loaded")
    print(f"  First ID: {ids[0]}")
    print(f"  First image shape (height, width, channels): {images[0].shape}")


# -----------------------------------------------------------------------------
# Task integration points
#
# Team interface to implement in w3/utils.py:
#   denoise_image(image_bgr) -> image_bgr
#   texture_descriptor(image_bgr) -> 1-D descriptor
#   detect_paintings(image_bgr) -> list of 0, 1, or 2 binary masks
#   crop_to_mask(image_bgr, mask) -> painting crop (reuse the Week 2 method)
#   get_distances(query_descriptors, db_descriptors, metric) -> distance matrix
#   retrieve(distance_matrix, db_ids, k) -> list of ranked database-ID lists
#   evaluate(ground_truth, predictions, k) -> mAP@k
#   metrics(predicted_mask, ground_truth_mask) -> dictionary of mask scores
#   visualize_denoising(originals, denoised, output_dir)
#   visualize_texture_retrieval(query_images, predictions, db_ids, db_images, output_dir)
#   visualize_masks(query_images, masks_by_query, output_dir)
#   visualize_painting_count(query_images, masks_by_query, output_dir)
#
# All images passed from this main file are BGR. A detector must return separate
# masks for separate paintings. The combined mask is only for saving/evaluation.
# -----------------------------------------------------------------------------
def apply_query_preprocessing(
    images: list[np.ndarray], denoise_image
) -> list[np.ndarray]:
    """Apply the selected fixed denoising policy to query images."""
    if not APPLY_DENOISING:
        return images
    return [denoise_image(image) for image in images]


def make_descriptors(images: list[np.ndarray], texture_descriptor) -> np.ndarray:
    """Build one descriptor row per image using the team's texture method."""
    return np.stack([texture_descriptor(image) for image in images])


def rank_images(
    query_images: list[np.ndarray],
    db_descriptors: np.ndarray,
    db_ids: list[int],
    denoise_image,
    texture_descriptor,
    get_distances,
    retrieve,
    k: int,
) -> list[list[int]]:
    """Describe queries and return their ranked database matches."""
    prepared_queries = apply_query_preprocessing(query_images, denoise_image)
    query_descriptors = make_descriptors(prepared_queries, texture_descriptor)
    distances = get_distances(query_descriptors, db_descriptors, metric=METRIC)
    return retrieve(distances, db_ids, k=k)


def combined_mask(masks: list[np.ndarray], image_shape: tuple[int, ...]) -> np.ndarray:
    """Combine separate painting masks into one binary image mask."""
    result = np.zeros(image_shape[:2], dtype=np.uint8)
    for mask in masks:
        if np.asarray(mask).shape != result.shape:
            raise ValueError("A painting mask does not match its image dimensions.")
        result[np.asarray(mask) > 0] = 255
    return result


# -----------------------------------------------------------------------------
# Development pipeline
# -----------------------------------------------------------------------------
def run_qsd1_development(
    db_ids,
    db_images,
    query_images,
    ground_truth,
    denoise_image,
    texture_descriptor,
    get_distances,
    retrieve,
    evaluate,
    visualize_texture_retrieval,
) -> list[list[int]]:
    """Week 3 texture retrieval for cropped, one-painting QSD1 images."""
    db_descriptors = make_descriptors(db_images, texture_descriptor)
    predictions = rank_images(
        query_images,
        db_descriptors,
        db_ids,
        denoise_image,
        texture_descriptor,
        get_distances,
        retrieve,
        K_DEV,
    )

    print("\nQSD1-W3 texture retrieval")
    print(f"mAP@1: {evaluate(ground_truth, predictions, k=1):.4f}")
    print(f"mAP@5: {evaluate(ground_truth, predictions, k=5):.4f}")

    visualize_texture_retrieval(
        query_images,
        predictions,
        db_ids,
        db_images,
        RESULTS_DIR / "visualizations" / "texture",
    )
    return predictions


def run_qsd2_development(
    query_ids,
    query_images,
    ground_truth,
    db_ids,
    db_descriptors,
    denoise_image,
    texture_descriptor,
    detect_paintings,
    crop_to_mask,
    get_distances,
    retrieve,
    metrics,
    visualize_masks,
    visualize_painting_count,
) -> list[list[list[int]]]:
    """Task 4 placeholder: segment and retrieve up to two QSD2 paintings.

    Output structure: query -> painting -> ranked BBDD IDs. The separate
    painting masks are also combined only for segmentation scoring/visuals.
    """
    if not (len(query_ids) == len(query_images) == len(ground_truth)):
        raise ValueError(
            "QSD2 images, IDs, and ground truth must have equal lengths."
        )

    all_predictions: list[list[list[int]]] = []
    mask_scores = []
    all_masks = []

    for query_id, image, actual_ids in zip(query_ids, query_images, ground_truth):
        prepared = denoise_image(image) if APPLY_DENOISING else image
        masks = detect_paintings(prepared)

        if len(masks) > 2:
            raise ValueError(
                f"QSD2 query {query_id}: detector returned more than 2 masks."
            )

        for mask in masks:
            if np.asarray(mask).shape != image.shape[:2]:
                raise ValueError(
                    f"QSD2 query {query_id}: mask size does not match image."
                )

        all_masks.append(masks)
        crops = [crop_to_mask(prepared, mask) for mask in masks]

        query_predictions: list[list[int]] = []
        for crop in crops:
            descriptor = texture_descriptor(crop)
            distances = get_distances(
                np.asarray([descriptor]), db_descriptors, metric=METRIC
            )
            query_predictions.append(retrieve(distances, db_ids, k=K_DEV)[0])

        all_predictions.append(query_predictions)

        # QSD2's PNG annotation is the union mask for the image.
        mask_path = QSD2_DIR / f"{query_id:05d}.png"
        annotation = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if annotation is not None:
            predicted_union = combined_mask(masks, image.shape)
            mask_scores.append(metrics(predicted_union, annotation > 0))

        # Ground truth is one or two IDs per image. Confirm the annotation order
        # before calculating per-painting retrieval mAP.
        if len(masks) != len(actual_ids):
            print(
                f"QSD2 query {query_id}: detected {len(masks)} painting(s), "
                f"ground truth lists {len(actual_ids)} ID(s)."
            )

    if mask_scores:
        print("\nQSD2-W3 mask scores")
        for key in mask_scores[0]:
            mean_score = np.mean([score[key] for score in mask_scores])
            print(f"{key}: {mean_score:.4f}")

    visualize_masks(
        query_images, all_masks, RESULTS_DIR / "visualizations" / "masking"
    )
    visualize_painting_count(
        query_images,
        all_masks,
        RESULTS_DIR / "visualizations" / "painting_count",
    )

    print(
        "QSD2 retrieval predictions are nested by image then painting. "
        "Add the per-painting mAP calculation after the team confirms the "
        "ground-truth/mask ordering."
    )
    return all_predictions


# -----------------------------------------------------------------------------
# Task 5 placeholder: blind test pipeline and required submission files
# -----------------------------------------------------------------------------
def save_pickle(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(value, file)


def run_test(
    db_ids,
    db_descriptors,
    denoise_image,
    texture_descriptor,
    detect_paintings,
    crop_to_mask,
    get_distances,
    retrieve,
) -> None:
    """Run QST1/QST2 and save the Week 3 nested results and QST2 masks.

    Pickle structure: query -> painting -> ranked BBDD IDs. QST1 has one
    painting per query, so its painting level contains one ranking. QST2 can
    contain zero, one, or two rankings. Save only method1 (the selected best
    method), with K=10.
    """
    if not QST1_DIR.exists() or not QST2_DIR.exists():
        print("Test folders are not available yet; skipping the test pipeline.")
        return

    qst1_ids, qst1_images = load_images_bgr(QST1_DIR)
    qst1_rankings = rank_images(
        qst1_images,
        db_descriptors,
        db_ids,
        denoise_image,
        texture_descriptor,
        get_distances,
        retrieve,
        K_TEST,
    )

    # Required extra list level: query -> its one painting -> ranked IDs.
    qst1_results = [[ranking] for ranking in qst1_rankings]
    save_pickle(
        qst1_results,
        RESULTS_DIR / "QST1" / "method1" / "result.pkl",
    )

    qst2_ids, qst2_images = load_images_bgr(QST2_DIR)
    qst2_results: list[list[list[int]]] = []
    qst2_dir = RESULTS_DIR / "QST2" / "method1"
    qst2_dir.mkdir(parents=True, exist_ok=True)

    for query_id, image in zip(qst2_ids, qst2_images):
        prepared = denoise_image(image) if APPLY_DENOISING else image
        masks = detect_paintings(prepared)

        if len(masks) > 2:
            raise ValueError(
                f"QST2 query {query_id}: detector returned more than 2 masks."
            )

        for mask in masks:
            if np.asarray(mask).shape != image.shape[:2]:
                raise ValueError(
                    f"QST2 query {query_id}: mask size does not match image."
                )

        query_results: list[list[int]] = []
        for mask in masks:
            crop = crop_to_mask(prepared, mask)
            descriptor = texture_descriptor(crop)
            distances = get_distances(
                np.asarray([descriptor]), db_descriptors, metric=METRIC
            )
            ranking = retrieve(distances, db_ids, k=K_TEST)[0]
            query_results.append(ranking)

        qst2_results.append(query_results)

        # Submission PNG is the union of painting masks for this query.
        mask_image = combined_mask(masks, image.shape)
        cv2.imwrite(str(qst2_dir / f"{query_id:05d}.png"), mask_image)

    save_pickle(qst2_results, qst2_dir / "result.pkl")
    print(f"Test results saved under: {RESULTS_DIR}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    db_ids, db_images = load_images_bgr(BBDD_DIR)
    qsd1_ids, qsd1_images = load_images_bgr(QSD1_DIR)
    qsd2_ids, qsd2_images = load_images_bgr(QSD2_DIR)
    qsd1_gt = load_ground_truth(QSD1_DIR)
    qsd2_gt = load_ground_truth(QSD2_DIR)

    describe_dataset("BBDD", db_ids, db_images)
    describe_dataset("QSD1-W3", qsd1_ids, qsd1_images)
    describe_dataset("QSD2-W3", qsd2_ids, qsd2_images)
    print(f"QSD1 ground-truth entries: {len(qsd1_gt)}")
    print(f"QSD2 ground-truth entries: {len(qsd2_gt)}")

    if not PIPELINE_READY:
        print(
            "\nPipeline scaffold is ready. Implement the listed functions in "
            "w3/utils.py, set METRIC, then change PIPELINE_READY to True."
        )
        return

    # Import only after the shared team functions are ready. This lets the
    # scaffold run its data checks while teammates are still implementing tasks.
    from utils import (
        denoise_image,
        texture_descriptor,
        detect_paintings,
        crop_to_mask,
        get_distances,
        retrieve,
        evaluate,
        metrics,
        visualize_denoising,
        visualize_texture_retrieval,
        visualize_masks,
        visualize_painting_count,
    )

    if METRIC is None:
        raise ValueError("Set METRIC to the selected Week 1/2 distance metric.")

    db_descriptors = make_descriptors(db_images, texture_descriptor)

    if APPLY_DENOISING:
        sample = qsd1_images[:5]
        denoised_sample = [denoise_image(image) for image in sample]
        visualize_denoising(
            sample,
            denoised_sample,
            RESULTS_DIR / "visualizations" / "denoising",
        )

    run_qsd1_development(
        db_ids,
        db_images,
        qsd1_images,
        qsd1_gt,
        denoise_image,
        texture_descriptor,
        get_distances,
        retrieve,
        evaluate,
        visualize_texture_retrieval,
    )

    run_qsd2_development(
        qsd2_ids,
        qsd2_images,
        qsd2_gt,
        db_ids,
        db_descriptors,
        denoise_image,
        texture_descriptor,
        detect_paintings,
        crop_to_mask,
        get_distances,
        retrieve,
        metrics,
        visualize_masks,
        visualize_painting_count,
    )

    run_test(
        db_ids,
        db_descriptors,
        denoise_image,
        texture_descriptor,
        detect_paintings,
        crop_to_mask,
        get_distances,
        retrieve,
    )


if __name__ == "__main__":
    main()