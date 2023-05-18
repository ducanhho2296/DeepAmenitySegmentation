import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import configparser
from io import BytesIO
from PIL import Image
import osmnx as ox
import rasterio

# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Set the parent directory of the script as a module search path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import custom modules
from utils.buildings_osmnx import CityData
from utils.label_image import label_image
from utils.image_processing import pad_image_to_square, pad_tif_to_square
from shapely.geometry import Polygon
from generate_dataset import extract_point


# Set paths from configuration file
root_path = config['paths']['root_path']

# Set city information from configuration file
city_name = config['city']['name']

#testset
test_image_dir = config['paths']['image_test']
test_label_dir = config['paths']['label_test']
test_image_path = os.path.join('..',root_path, test_image_dir, 'image{}.jpg')
test_label_path = os.path.join('..',root_path, test_label_dir, 'label{}.tif')

# Set image processing parameters from configuration file
img_size = int(config['processing']['img_size'])
dist = int(config['city']['test_dist']) 

city_data = CityData(city_name)
buildings, city_bbox = city_data.get_data()
amenity_points = []
exploded = buildings.explode(index_parts=True)
count = 0
random_points = buildings.sample(n=1000, random_state=1)

list_points = []
for geometry in random_points['geometry']:
    list_points.append(extract_point(geometry))


for j in list_points:
    # raster = np.zeros(shape=(size,size))  # A[i][j] = "(long, latt)"
    center_point = (j[1], j[0]) #center_point = (latitude, longtitude)
    bbox = ox.utils_geo.bbox_from_point(center_point, dist=dist)

    bbox_polygon = Polygon([(bbox[3],bbox[1]), (bbox[2], bbox[1]),(bbox[2], bbox[0]), (bbox[3], bbox[0])])
    #checking if polygons inside the bbox
    check_polygon = exploded.geometry.within(bbox_polygon)

    #return all rows, which have polygons inside bbox
    polygon_inbbox = check_polygon[check_polygon.values == True]

    #using isin() to extract subtable contains only polygon_inbbox.index
    indexs = polygon_inbbox.index
    condition = exploded['geometry'].index.isin(indexs)
    buildings_inbbox = exploded[condition]

    #mapping amenities inside bbox into the new axis
    if len(buildings_inbbox.geometry) != 0: #perform labelling only when there is building inside bbox
    #plotting time
        fig1, ax1 = plt.subplots(figsize=(10,10))
        ax1.set_aspect(1)
        fig1.patch.set_facecolor('black')
        plt.axis('off')
        buildings_inbbox.plot(ax=ax1, edgecolor='white', facecolor='white')
        count += 1
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, dpi=100, format="jpg", bbox_inches='tight')
        img_buffer.seek(0)
        img = Image.open(img_buffer)
        img = np.array(img)  
        
        # Generate label for img
        label_data, label_profile, num_classes = label_image(buildings_inbbox, img, j)
        tif_buffer = BytesIO()
        
        with rasterio.open(tif_buffer, 'w', **label_profile) as label_dst:
            label_dst.write(label_data)
        tif_buffer.seek(0)
        
        # Process padding image from rectangle to fixed square size 
        padded_img = pad_image_to_square(img, img_size)
        cv2.imwrite(test_image_path.format(count), padded_img)
        
        # Process padding label
        pad_tif_to_square(tif_buffer, test_label_path.format(count), img_size)

        plt.close()
        
        img_buffer.close()
        tif_buffer.close()

print("Adding padding to bounding box and processing images successfully")