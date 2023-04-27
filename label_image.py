import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

def label_image(building_df, image,i=1, write_path=None):
    if write_path == None:
        write_path = "label{}.tif".format(i)
    # Set the CRS of the label image
    label_crs = CRS.from_epsg(4326)

    # Define the dimensions and projection of the label image
    label_width, label_height, dims = image.shape

    # Set the transform of the label image to the bounds of the geodataframe
    label_transform = from_bounds(*building_df.total_bounds, label_width, label_height)

    # Create a new writeable raster dataset for the label image
    label_data = np.zeros((label_height, label_width), dtype=np.uint8)
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
        np.nan: 1,   #NaN fuction: no amenity
        "retail": 2,
        "food": 3,
        "school": 4,
        "healthcare": 5,
        "entertainment": 6,
        "public": 7,
        "leisure": 8
    }
    with rasterio.open(write_path, 'w', **label_profile) as label_dst:
        # Rasterize the geodataframe to the label image
        shapes = ((geom, class_mapping[func]) for func, geom in zip(building_df['function'], building_df['geometry']))
        rasterio.features.rasterize(shapes=shapes, out=label_data, transform=label_transform)
        label_dst.write(label_data, 1)