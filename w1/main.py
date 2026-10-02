from utils import *

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
BBDD_DIR = DATA_DIR / 'BBDD'

if __name__ == '__main__':
    bbdd = load_images(BBDD_DIR)
    print(f'Loaded {len(bbdd)} images from {BBDD_DIR}')
    img = bbdd[15]
    show_histogram(img)
    hist = complex_descriptor(img)
    print(f'Descriptor shape: {hist.shape}')
