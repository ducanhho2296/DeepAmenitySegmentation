import rasterio
import numpy as np

def convert_label_to_coordinates_with_class_names(predicted_label_image, label_profile, class_mapping):
    # Open the label image using rasterio
    with rasterio.open(predicted_label_image, 'r') as src:
        # Get the transform from the label profile
        transform = src.transform
        
        # Read the label values from the image
        predicted_label_array = src.read(1)
        
        # Convert pixel coordinates to geographical coordinates
        row_indices, col_indices = np.where(predicted_label_array != 0)
        lon_values, lat_values = rasterio.transform.xy(transform, row_indices, col_indices)
