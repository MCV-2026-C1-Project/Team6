from pathlib import Path

import cv2

from utils import best_descriptor_3d

import numpy as np


if __name__ == "__main__":
    # Locate the museum dataset.
    data_dir = Path(__file__).resolve().parent.parent / "data"
    image_paths = sorted((data_dir / "BBDD").glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError("No JPG images found in data/BBDD.")

    # Load one painting and convert it to Lab.
    image_path = image_paths[0]
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read {image_path}")

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    descriptor = best_descriptor_3d(image_lab)
    #Including every pixel through a mask should give the same result.
    full_mask = np.ones(image_lab.shape[:2], dtype=np.uint8)
    masked_descriptor = best_descriptor_3d(image_lab, mask=full_mask)
    assert descriptor.size == 2048
    assert np.isclose(descriptor.sum(), 1.0)
    assert np.allclose(descriptor, masked_descriptor)

        # Changing excluded pixels must not affect the descriptor.
    partial_mask = np.zeros(image_lab.shape[:2], dtype=np.uint8)
    partial_mask[:, :image_lab.shape[1] // 2] = 1

    changed_image = image_lab.copy()
    changed_image[partial_mask == 0] = (0, 0, 0)

    original_desc = best_descriptor_3d(image_lab, mask=partial_mask)
    changed_desc = best_descriptor_3d(changed_image, mask=partial_mask)

    assert np.allclose(original_desc, changed_desc)
    assert np.isclose(original_desc.sum(), 1.0)
    print("Background exclusion: passed")

    print("Image:", image_path.name)
    print("Descriptor length:", descriptor.size)
    print("Sum:", descriptor.sum())
    print("Full-mask equivalence: passed")