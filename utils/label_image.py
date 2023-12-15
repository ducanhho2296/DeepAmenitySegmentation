import numpy as np
import rasterio.features
from rasterio.crs import CRS
from rasterio.transform import from_bounds


def label_image(building_df, image, i=1):
    # Set the CRS of the label image
    label_crs = CRS.from_epsg(4326)

    # Define the dimensions and projection of the label image
    label_width, label_height, dims = image.shape

    # Set the transform of the label image to the bounds of the geodataframe
    label_transform = from_bounds(*building_df.total_bounds, label_width, label_height)

    # Create a new writeable raster dataset for the label image
    label_data = np.zeros((1, label_height, label_width), dtype=np.uint8)
    label_profile = {
        'driver': 'GTiff',
        'height': label_height,
        'width': label_width,
        'count': 1,
        'dtype': np.uint8,
        'crs': label_crs,
        'transform': label_transform
    }
    class_mapping = {
        np.nan: 1,   # NaN function: no amenity
        "retail": 2,
        "food": 3,
        "school": 4,
        "healthcare": 5,
        "entertainment": 6,
        "public": 7,
        "leisure": 8,
        "sport": 9,
        "highway":10
    }
    num_classes = len(class_mapping)
    # Rasterize the geodataframe to the label image
    shapes = ((geom, class_mapping[func]) for func, geom in zip(building_df['function'], building_df['geometry']))
    rasterio.features.rasterize(shapes=shapes, out=label_data[0], transform=label_transform)
    
    return label_data, label_profile, num_classes

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
# def pad_tif_to_square(input_file, output_file):
#     with rasterio.open(input_file) as src:
#         img = src.read()
#         img = np.moveaxis(img, 0, -1)
