import numpy as np
from PIL import Image
import rasterio

def pad_image_to_square(img):
    height, width = img.shape[:2]
    max_dim = max(height, width)
    padded_img = np.zeros((max_dim, max_dim, 3), dtype=img.dtype)
    y_offset = (max_dim - height) // 2
    x_offset = (max_dim - width) // 2
    padded_img[y_offset:y_offset+height, x_offset:x_offset+width] = img
    return padded_img

def pad_tif_to_square(input_file, output_file):
    with rasterio.open(input_file) as src:
        img = src.read()
        img = np.moveaxis(img, 0, -1)
        padded_img = pad_image_to_square(img)
        transform = src.transform
        profile = src.profile
        profile.update(width=padded_img.shape[1], height=padded_img.shape[0], transform=transform)
    with rasterio.open(output_file, 'w', **profile) as dst:
        padded_img = np.moveaxis(padded_img, -1, 0)
        dst.write(padded_img)