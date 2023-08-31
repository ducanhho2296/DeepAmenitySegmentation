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
