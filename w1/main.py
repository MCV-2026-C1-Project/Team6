from utils import *

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
BBDD_DIR = DATA_DIR / 'BBDD'

if __name__ == '__main__':
    bbdd = load_images(BBDD_DIR)
    print(f'Loaded {len(bbdd)} images from {BBDD_DIR}')
    hist = compute_histogram(bbdd[15])
    print(f'Descriptor shape: {hist.shape}')
