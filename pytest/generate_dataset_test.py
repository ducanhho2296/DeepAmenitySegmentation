import os
import sys
import pytest
import configparser

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from datasets.generate_dataset import grid_sampling, stratified_sampling


config_path = 'test_config.ini'

def test_grid_sampling():
    test_output_dir = 'test_grid_sampling_output'
    os.makedirs(test_output_dir, exist_ok=True)
    
    config = configparser.ConfigParser()
    config.read(config_path)
    config['paths']['root_path'] = 'DeepAmenitySegmentation'
