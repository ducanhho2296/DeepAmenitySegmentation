import rasterio
import numpy as np

def convert_label_to_coordinates_with_class_names(predicted_label_image, label_profile, class_mapping):
    # Open the label image using rasterio
    with rasterio.open(predicted_label_image, 'r') as src:
        # Get the transform from the label profile
        transform = src.transform
        
        # Read the label values from the image
