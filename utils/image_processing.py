import numpy as np
from PIL import Image
import rasterio

import cv2
import numpy as np

def pad_image_to_square(img, size=448):
    height, width = img.shape[:2]
    # Calculate the scale factor for resizing the image
    scale_factor = size / max(height, width)
    # Resize the image to the desired size while maintaining the aspect ratio
    resized_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    # Get the height and width of the resized image
    resized_height, resized_width = resized_img.shape[:2]
    padded_img = np.zeros((size, size, 3), dtype=img.dtype)
    # Calculate the offsets for the padded image
    y_offset = (size - resized_height) // 2
    x_offset = (size - resized_width) // 2
    # Copy the resized image into the padded image at the calculated offsets
    padded_img[y_offset:y_offset+resized_height, x_offset:x_offset+resized_width] = resized_img
    # Return the padded image
    return padded_img


# def pad_tif_to_square(input_file, output_file):
#     with rasterio.open(input_file) as src:
#         img = src.read()
#         img = np.moveaxis(img, 0, -1)
#         padded_img = pad_image_to_square(img)
#         transform = src.transform
#         profile = src.profile
# def pad_tif_to_square(input_file, output_file):
#     with rasterio.open(input_file) as src:
#         img = src.read()
#         img = np.moveaxis(img, 0, -1)
#         padded_img = pad_image_to_square(img)
#         transform = src.transform
#         profile = src.profile
#         profile.update(width=padded_img.shape[1], height=padded_img.shape[0], transform=transform)
#     with rasterio.open(output_file, 'w', **profile) as dst:
#         padded_img = np.moveaxis(padded_img, -1, 0)
#         dst.write(padded_img)

from scipy.ndimage import binary_dilation

def pad_tif_to_square(tif_path, padded_tif_path, size):
    with rasterio.open(tif_path) as src:
        img = src.read()
        img_shape = img.shape
        src_profile = src.profile.copy()

    padded_img = np.zeros((img_shape[0], size, size), dtype=img.dtype)

    # Calculate the scaling factor to resize the image
    scaling_factor = size / max(img_shape[1:])

    # Resize the image
    new_height = int(np.round(img_shape[1] * scaling_factor))
    new_width = int(np.round(img_shape[2] * scaling_factor))
    img_resized = np.zeros((img_shape[0], new_height, new_width), dtype=img.dtype)
    for i in range(img_shape[0]):
        img_resized[i] = cv2.resize(img[i], (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Pad the image
    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2
    padded_img[:, y_offset:y_offset+new_height, x_offset:x_offset+new_width] = img_resized

    # Set the padded pixels to the same value as the boundary pixels
    boundary = binary_dilation(padded_img[0] > 0) & (padded_img[0] == 0)
    for i in range(img_shape[0]):
        padded_img[i, boundary] = padded_img[i, ~boundary].mean()
    # Write the padded image to disk
    src_profile.update(width=size, height=size, transform=src.transform)
    with rasterio.open(padded_tif_path, 'w', **src_profile) as dst:
        dst.write(padded_img)


if __name__ == "__main__":
    import os
    root_path = 'DeepAmenitySegmentation'

    label_path = "/home/ducanh/DeepAmenitySegmentation/datasets/labels/label68.tif"
    padded_label_path = "/home/ducanh/DeepAmenitySegmentation/augmented_datasets/labels/labeldemo.tif"

    pad_tif_to_square(label_path, padded_label_path, 512)
