
from skimage import io, transform
from scipy import ndimage
import numpy as np

from matplotlib import pyplot as plt


# img = io.imread("/Users/chenhongji/Projects/img.png", as_grey=True)
# filtered = ndimage.median_filter(img, 5)
# io.imsave("/Users/chenhongji/Projects/filtered.png", filtered)


# aceptar img en formato de ndarray
def reduce_noise(img):

    io.imshow(ndimage.median_filter(img, 5))
    plt.show()
    return ndimage.median_filter(img, 5)

# Binarización
def two_value(img):
    rows, cols = img.shape

    print("rows: ", rows, "cols: ", cols)

    # Crear un matriz de ceros del mismo tamaño de img
    img2 = np.zeros((rows, cols), dtype = np.uint8)

    for i in range(rows):
        for j in range(cols):
            if img[i][j] > 128:
                img2[i][j] = 255
            else:
                img2[i][j] = 0
    return img2


def image_process(img):
    return reduce_noise(two_value(img))


def normalize(image):
    rows, cols = image.shape
    padding = image
    if rows > cols:
        left = int((rows - cols) / 2)
        right = rows - cols - left
        padding = np.lib.pad(image, ((0, 0), (left, right)), "constant", constant_values=(0, 0))
    elif cols > rows:
        top = int((cols - rows) / 2)
        bottom = cols - rows - top
        padding = np.lib.pad(image, ((top, bottom), (0, 0)), "constant", constant_values=(0, 0))
    return transform.resize(padding, (64, 64))