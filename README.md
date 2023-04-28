# DeepAmenitySegmentation
DeepAmenitySegmentation is a deep learning project focused on recognizing amenities in OpenStreetMap data using satellite imagery. The primary objective is to classify buildings in urban environments into different amenity categories, such as retail, food, school, healthcare, entertainment, public, and leisure, using semantic segmentation techniques.

## Overview
This project uses a custom semantic segmentation model based on EfficientNet and a dataset of satellite images with corresponding OpenStreetMap (OSM) data. The OSM data contains building polygons and their amenity class information, which are used to create label images for training the model. The trained model can then be used to automatically classify buildings in satellite images and update the amenity information in OSM.

## Features
### Custom dataset class for loading satellite images and label images
- Using Osmnx to extract geo spatial data from open street map into a Geodataframe and using Geopandas to extract geo values of amenities in map
![download](https://user-images.githubusercontent.com/92146886/219333765-b746ee07-e997-42bd-b49d-64c31464274a.png)

### EfficientNet-based semantic segmentation model
- Data preprocessing scripts for converting OSM building polygons into label images
- Training and validation scripts for model training
- Configuration files for easy management of hyperparameters and settings

## Requirements
- Python 3.6 or later
- PyTorch
- torchvision
- geopandas
- rasterio
- PIL (Pillow)

## Usage
- Prepare your dataset: Collect satellite images and corresponding GeoJSON files containing building polygons and amenity information.
- Generate label images: Run the data preprocessing script to convert building polygons and amenity information into label images.
- Train the model: Modify the configuration file with your desired hyperparameters and settings, and run the training script.
- Evaluate the model: Run the validation script to evaluate the trained model on a separate dataset.
- Use the trained model: Apply the trained model on new satellite images to automatically recognize and classify buildings based on their amenity types.
