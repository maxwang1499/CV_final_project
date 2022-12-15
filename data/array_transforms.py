import numpy as np
from PIL import Image

def resize_as_image(arr, size):
    old_size = arr.shape[0]
    if old_size == size:
        return arr
    img = Image.fromarray(arr)
    img = img.resize(size=(size, size))
    return np.array(img)

def upscale(arr, size):
    old_size = arr.shape[0]
    scaling = size // old_size
    if len(arr.shape) > 2:
        return np.kron(arr, np.ones((scaling, scaling, 1)))
    return np.kron(arr, np.ones((scaling, scaling)))

def center_crop(arr, size):
    old_size = arr.shape[0]
    minval = (old_size - size) // 2
    maxval = (old_size + size) // 2
    return arr[minval:maxval, minval:maxval]

def scale_values(arr, minval, maxval):
    return (arr - minval) / (maxval - minval)
