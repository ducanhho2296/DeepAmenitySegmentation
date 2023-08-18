import rasterio
import numpy as np

def convert_label_to_coordinates_with_class_names(predicted_label_image, label_profile, class_mapping):
