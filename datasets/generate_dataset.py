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

#training dataset
padded_img_dir = config['paths']['padded_img_dir']
padded_label_dir = config['paths']['padded_label_dir']

#testset
test_image_dir = config['paths']['image_test']
test_label_dir = config['paths']['label_test']

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

#test_path
test_image_path = os.path.join('..',root_path, test_image_dir, 'image{}.jpg')
test_label_path = os.path.join('..',root_path, test_label_dir, 'label{}.tif')

city_data = CityData(city_name)
buildings, city_bbox = city_data.get_data()
global count
count = 0

def grid_sampling():
    

    # Create grids for city
    grid_generating = Gridgenerating(buildings, spacing_x=spacing_x, spacing_y=spacing_y)
    grid_points = grid_generating.grid_points_in_bbox(bbox=city_bbox)

    total_iterations = len(grid_points) - 1
    num_classes = None

    for j in range(total_iterations):
        buildings_within_bbox = None
        bbox_visual = None
        img = None

        buildings_within_bbox, bbox_visual = grid_generating.capture_square_bbox(point_position=j, grid_points=grid_points) 

        if j % (total_iterations // 10) == 0:
            progress = (j / total_iterations) * 100
            print(f"Processing figures and extracting label: {progress:.2f}%")

        # Capture only bounding boxes that have buildings inside
        if len(buildings_within_bbox) != 0:
            global count
            count += 1
            
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.patch.set_facecolor('black')
            
            # Plot buildings_within_bbox for generating bounding box figures
            buildings_within_bbox.plot(ax=ax, edgecolor='white', facecolor='white')
            plt.axis('off')
            
            # Save figures in buffer memory
            img_buffer = BytesIO()
            plt.savefig(img_buffer, dpi=100, format="jpg", bbox_inches='tight')
            img_buffer.seek(0)
            img = Image.open(img_buffer)
            img = np.array(img)  
            
            # Generate label for img
            label_data, label_profile, num = label_image(buildings_within_bbox, img, j)
            num_classes = num
            tif_buffer = BytesIO()
            
            with rasterio.open(tif_buffer, 'w', **label_profile) as label_dst:
                label_dst.write(label_data)
            tif_buffer.seek(0)
            
            # Process padding image from rectangle to fixed square size 
            padded_img = pad_image_to_square(img, img_size)
            cv2.imwrite(padded_img_path.format(count), padded_img)
            
            # Process padding label
            pad_tif_to_square(tif_buffer, padded_label_path.format(count), img_size)
            plt.close()
            
            img_buffer.close()
            tif_buffer.close()

    print("Generating square bbox images with Grid-sampling successfully")
    print("Adding padding to bounding box and processing images successfully")
    print("-----------------------------------------------------------------------------------------------------")

    #overwritting number of classes into config.ini file for training process
    # Load the config file
    config.set('num_classes', 'n', str(num_classes))
    # Save the updated config file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)


from shapely.geometry import Point, Polygon, MultiPolygon
import numpy as np
import osmnx as ox

def extract_point(geometry):
    if isinstance(geometry, Point):
        return np.array([geometry.x, geometry.y])
    elif isinstance(geometry, Polygon):
        return np.array(geometry.centroid.coords[0])
    elif isinstance(geometry, MultiPolygon):
        largest_polygon = max(geometry.geoms, key=lambda x: x.area)
        return np.array(largest_polygon.centroid.coords[0])

def random_generate(save=True):
    amenity_points = []
    # extract all POINT type coordinates
    retail = buildings[buildings.function == "retail"]
    food = buildings[buildings.function == "food"]
    school = buildings[buildings.function == "school"]
    entertainment = buildings[buildings.function == "entertainment"]
    healthcare = buildings[buildings.function == "healthcare"]
    public = buildings[buildings.function == "public"]
    sport = buildings[buildings.function == "sport"]
    leisure = buildings[buildings.function == "leisure"]
    highway = buildings[buildings.function == "highway"]

    for geometry in retail['geometry']:
        amenity_points.append(extract_point(geometry))

    for geometry in food['geometry']:
        amenity_points.append(extract_point(geometry))

    for geometry in school['geometry']:
        amenity_points.append(extract_point(geometry))

    for geometry in entertainment['geometry']:
        amenity_points.append(extract_point(geometry))

    for geometry in healthcare['geometry']:
        amenity_points.append(extract_point(geometry))

    for geometry in public['geometry']:
        amenity_points.append(extract_point(geometry))

    for geometry in sport['geometry']:
        amenity_points.append(extract_point(geometry))   

    for geometry in leisure['geometry']:
        amenity_points.append(extract_point(geometry))   

    for geometry in highway['geometry']:
        amenity_points.append(extract_point(geometry))   

    print("number of amenity points for the whole city: {}".format(len(amenity_points)))
    exploded = buildings.explode(index_parts=True)

    dist = int(config['city']['dist'])
    if save:
        for j in amenity_points:
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
                global count
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
                cv2.imwrite(padded_img_path.format(count), padded_img)
                
                # Process padding label
                pad_tif_to_square(tif_buffer, padded_label_path.format(count), img_size)

                plt.close()
                
                img_buffer.close()
                tif_buffer.close()
        
        print("Adding padding to bounding box and processing images successfully")

    input("press Enter to create test sets from amenity points")
    test_dist = int(config['city']['test_dist'])
    c = 0
    for j in range(round(len(amenity_points)/3)):
        point = amenity_points[j]
        # raster = np.zeros(shape=(size,size))  # A[i][j] = "(long, latt)"
        center_point = (point[1], point[0]) #center_point = (latitude, longtitude)
        bbox_test = ox.utils_geo.bbox_from_point(center_point, dist=test_dist)

        bbox_polygon_test = Polygon([(bbox_test[3],bbox_test[1]), 
                                     (bbox_test[2], bbox_test[1]),
                                     (bbox_test[2], bbox_test[0]), 
                                     (bbox_test[3], bbox_test[0])])
        #checking if polygons inside the bbox
        check_polygon_test = exploded.geometry.within(bbox_polygon_test)

        #return all rows, which have polygons inside bbox
        polygon_inbbox_test = check_polygon_test[check_polygon_test.values == True]

        #using isin() to extract subtable contains only polygon_inbbox.index
        indexs_test = polygon_inbbox_test.index
        condition_test = exploded['geometry'].index.isin(indexs_test)
        buildings_inbbox_test = exploded[condition_test]

        #mapping amenities inside bbox into the new axis
        if len(buildings_inbbox_test.geometry) != 0: #perform labelling only when there is building inside bbox
        #plotting time
            fig2, ax2 = plt.subplots(figsize=(10,10))
            ax2.set_aspect(1)
            fig2.patch.set_facecolor('black')
            plt.axis('off')
            buildings_inbbox_test.plot(ax=ax2, edgecolor='white', facecolor='white')
            c += 1
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, dpi=100, format="jpg", bbox_inches='tight')
            img_buffer.seek(0)
            img = Image.open(img_buffer)
            img = np.array(img)  
            
            # Generate label for img
            label_data, label_profile, num_classes = label_image(buildings_inbbox_test, img, j)
            tif_buffer = BytesIO()
            
            with rasterio.open(tif_buffer, 'w', **label_profile) as label_dst:
                label_dst.write(label_data)
            tif_buffer.seek(0)
            
            # Process padding image from rectangle to fixed square size 
            padded_img = pad_image_to_square(img, img_size)
            cv2.imwrite(test_image_path.format(c), padded_img)
            
            # Process padding label
            pad_tif_to_square(tif_buffer, test_label_path.format(c), img_size)

            plt.close()
            
            img_buffer.close()
            tif_buffer.close()
    print("Generating test dataset successfully with {} images.".format(c))




if __name__ == "__main__":
    grid_sampling()
    input("Press enter to continue generating datasets with amenity sampling method")
    random_generate(save=True) #change save=False to extract only test set 
                                #without generating training set of amenity sampling method.
