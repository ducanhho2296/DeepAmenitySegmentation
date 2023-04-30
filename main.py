import os
import sys
import cv2
import matplotlib.pyplot as plt
import configparser

# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Set the parent directory of the script as a module search path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import custom modules
from utils.buildings_osmnx import CityData
from utils.region_sampling import Gridgenerating
from utils.label_image import label_image
from utils.image_processing import pad_image_to_square, pad_tif_to_square

# Set paths from configuration file
root_path = config['paths']['root_path']
raw_img_dir = config['paths']['raw_img_dir']
raw_label_dir = config['paths']['raw_label_dir']
padded_img_dir = config['paths']['padded_img_dir']
padded_label_dir = config['paths']['padded_label_dir']

# Set city information from configuration file
city_name = config['city']['name']

# Set grid generation parameters from configuration file
spacing_x = float(config['grid']['spacing_x'])
spacing_y = float(config['grid']['spacing_y'])

# Set image processing parameters from configuration file
img_size = int(config['processing']['img_size'])

# Define the paths for images and labels
img_path = os.path.join('..', root_path, raw_img_dir, 'figure{}.jpg')
label_path = os.path.join('..', root_path, raw_label_dir, 'label{}.tif')

padded_img_path = os.path.join('..', root_path, padded_img_dir, 'image{}.jpg')
padded_label_path = os.path.join('..', root_path, padded_label_dir, 'label{}.tif')

city_data = CityData(city_name)
buildings, city_bbox = city_data.get_data()

# Create grids for city
grid_generating = Gridgenerating(buildings, spacing_x=spacing_x, spacing_y=spacing_y)
grid_points = grid_generating.grid_points_in_bbox(bbox=city_bbox)


img_size = 512
count = 0
for i in range(0, len(grid_points)-1):
    buildings_within_bbox = None
    bbox_visual = None
    img = None
    buildings_within_bbox, bbox_visual = grid_generating.capture_square_bbox(point_position=i,grid_points=grid_points)  
    #capture only bboxs which have buildings inside
    if len(buildings_within_bbox) !=0 : 
        count += 1

        fig, ax = plt.subplots(figsize=(10,10))
        fig.patch.set_facecolor('black')
        #plotting buildings_within_bbox for generating bbox's figures
        buildings_within_bbox.plot(ax=ax, edgecolor='white', facecolor='white')
        plt.axis('off')
        plt.savefig(img_path.format(i),dpi=100, format="jpg", bbox_inches='tight')
        img = cv2.imread(img_path.format(i))
        #generate label for img
        label_image(buildings_within_bbox, img, write_path=label_path.format(i))
        #process padding image from rectangle to fixed square size 
        padded_img = pad_image_to_square(img, img_size)
        cv2.imwrite(padded_img_path.format(count), padded_img)
        #process padding label
        pad_tif_to_square(label_path.format(i), padded_label_path.format(count), img_size)
        plt.close()

