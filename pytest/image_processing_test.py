import os
import sys
import numpy as np
import rasterio
import cv2
import pytest

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils.image_processing import pad_image_to_square, pad_tif_to_square
import configparser


config = configparser.ConfigParser()
config.read('config.ini')
# Get the paths and other parameters from the configuration file
root_path = os.path.abspath(os.path.join(parent_dir, config.get('paths', 'root_path')))
model_weight_path = config['paths']['model_path']
test_image_path = config['paths']['image_test']
test_padded_image_path = config['paths']['label_test']


def test_pad_image_to_square():
    test_image = cv2.imread(test_image_path)
    padded_image = pad_image_to_square(test_image)
