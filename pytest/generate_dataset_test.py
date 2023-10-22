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
    config['paths']['padded_img_dir'] = 'datasets/images'
    config['paths']['padded_label_dir'] = 'datasets/labels'
    config['paths']['image_test'] = 'datasets/tests/images'
    config['paths']['label_test'] = 'datasets/tests/labels'
    config['city']['name'] = 'Hannover Mitte'
    config['city']['dist'] = '250'
    config['grid']['spacing_x'] = '0.0015'
    config['grid']['spacing_y'] = '0.0015'
    config['processing']['img_size'] = '512'

    with open(config_path, 'w') as configfile:
        config.write(configfile)
    
    os.environ['CONFIG_PATH'] = config_path
    
    grid_sampling()
    
    os.rmdir(test_output_dir)

def test_stratified_sampling():
    test_output_dir = 'test_stratified_sampling_output'
    os.makedirs(test_output_dir, exist_ok=True)
    
    config = configparser.ConfigParser()
    config.read(config_path)
    config['paths']['root_path'] = 'DeepAmenitySegmentation'
    config['paths']['padded_img_dir'] = 'datasets/images'
    config['paths']['padded_label_dir'] = 'datasets/labels'
    config['paths']['image_test'] = 'datasets/tests/images'
    config['paths']['label_test'] = 'datasets/tests/labels'
    config['city']['name'] = 'Hannover Mitte'
    config['city']['dist'] = '250'
    config['grid']['spacing_x'] = '0.0015'
    config['grid']['spacing_y'] = '0.0015'
    config['processing']['img_size'] = '512'

    with open(config_path, 'w') as configfile:
        config.write(configfile)
    
    os.environ['CONFIG_PATH'] = config_path
    
