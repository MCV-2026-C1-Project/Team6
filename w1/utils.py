from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

#We only use .jpg because these are the only ones that contain the required images. Ask a teacher what .png files are for. 
def load_images(folder):
    """Read every .jpg in folder and return {image_id: BGR image}."""
    images = {}
    for path in sorted(Path(folder).glob('*.jpg')):
        img = cv2.imread(str(path))
        if img is None:
            print(f'Could not read {path}')
            continue
        image_id = int(path.stem.split('_')[-1])
        images[image_id] = img
    return images

def show_histogram(img):
    """Given an assumed BGR image plot its histogram and the hole image"""
    colors = ('b','g', 'r')
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_img.set_title("Original image")
    ax_img.axis('off')

    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax_hist.plot(hist, color=col)
        ax_hist.set_xlim([0, 256])

    ax_hist.set_title("RGB Color Histogram")
    ax_hist.set_xlabel("Pixel Intensity")
    ax_hist.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

#This will later need normalization. 
def compute_histogram(img):
    """Given an image recieve the 3 1d colour histograms compressed in a single numpy array"""
    hists = []
    for i in range(img.shape[2]):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hists.append(hist.flatten())

    return np.concatenate(hists)