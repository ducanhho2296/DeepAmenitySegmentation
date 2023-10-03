import os
import sys
import pytest
import configparser

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from datasets.generate_dataset import grid_sampling, stratified_sampling


config_path = 'test_config.ini'
