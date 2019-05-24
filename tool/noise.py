import numpy as np
from PIL import Image
from skimage.util import random_noise


def noisy(image):
    np_img = np.array(image)
    out = random_noise(np_img)
    out = 255 * out
    out = out.astype(np.uint8)
    return Image.fromarray(out)
