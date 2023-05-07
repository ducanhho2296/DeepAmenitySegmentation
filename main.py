import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import configparser
from io import BytesIO
from PIL import Image
import rasterio

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
padded_img_path = os.path.join('..', root_path, padded_img_dir, 'image{}.jpg')
padded_label_path = os.path.join('..', root_path, padded_label_dir, 'label{}.tif')

city_data = CityData(city_name)
buildings, city_bbox = city_data.get_data()

# Create grids for city
grid_generating = Gridgenerating(buildings, spacing_x=spacing_x, spacing_y=spacing_y)
grid_points = grid_generating.grid_points_in_bbox(bbox=city_bbox)

img_size = 512
count = 0
total_iterations = len(grid_points)-1
for i in range(0, total_iterations):
        buildings_within_bbox = None
        bbox_visual = None
        img = None
        buildings_within_bbox, bbox_visual = grid_generating.capture_square_bbox(point_position=i,grid_points=grid_points) 
        if i % (total_iterations // 10) == 0:
                progress = (i / total_iterations) * 100
                print(f"Processing figures and extracting label: {progress:.2f}%")

        #capture only bboxs which have buildings inside
        if len(buildings_within_bbox) !=0 : 
                count += 1

                fig, ax = plt.subplots(figsize=(10,10))
                fig.patch.set_facecolor('black')
                #plotting buildings_within_bbox for generating bbox's figures
                buildings_within_bbox.plot(ax=ax, edgecolor='white', facecolor='white')
                plt.axis('off')
                #save figures in buffer memory
                img_buffer = BytesIO()
                plt.savefig(img_buffer, dpi=100, format="jpg", bbox_inches='tight')
                img_buffer.seek(0)
                img = Image.open(img_buffer)
                img = np.array(img)        
                #generate label for img
                # label_image(buildings_within_bbox, img, write_path=label_path.format(i))

                label_data, label_profile = label_image(buildings_within_bbox, img, i)
                tif_buffer = BytesIO()
                with rasterio.open(tif_buffer, 'w', **label_profile) as label_dst:
                        label_dst.write(label_data)
                tif_buffer.seek(0)
                #process padding image from rectangle to fixed square size 
                padded_img = pad_image_to_square(img, img_size)
                cv2.imwrite(padded_img_path.format(count), padded_img)
                #process padding label
                pad_tif_to_square(tif_buffer, padded_label_path.format(count), img_size)
                plt.close()
                # img_buffer.close()
                # tif_buffer.close()
print("Adding padding to bounding box and processing images successfully")

