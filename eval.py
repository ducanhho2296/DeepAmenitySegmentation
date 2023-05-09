import os
import sys 
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import SatelliteDataset, get_transforms
import segmentation_models_pytorch as smp
import configparser
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='unet', help='Model type (unet or deeplabv3plus)')
parser.add_argument('--gpu', type=int, default=0, help='specific gpu for training')

args = parser.parse_args()

from models.model.get_model import *

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Get the paths and other parameters from the configuration file
model_weight_path = config['paths']['model_path']
test_img_dir = os.path.abspath(os.path.join(parent_dir, config.get('paths', 'test_img_dir')))
test_label_dir = os.path.abspath(os.path.join(parent_dir, config.get('paths', 'test_label_dir')))
city_name = config['city']['name']
num_classes = int(config['training_params']['num_classes'])

# Set the device to use for training
gpu = args.gpu
device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

# Load the trained model
model_name = f"{args.model}_{city_name}_amenity_classification.pth"
model_path = os.path.join(model_weight_path, model_name)

if args.model == 'unet':
    model = get_unet_model(device, in_channels=3, num_classes=num_classes)
elif args.model == 'deeplabv3plus':
    model = get_deeplabv3plus_model(device, in_channels=3, num_classes=num_classes)
else:
    raise ValueError('Invalid model type')

model.load_state_dict(torch.load(model_path))
model.eval()

# Create the test dataset
test_dataset = SatelliteDataset(test_img_dir, test_label_dir, get_transforms())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Initialize lists for ground truth labels and predictions
y_true = []
y_pred = []

# Evaluate the model on the test dataset
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device).float()
        masks = masks.to(device)

        # Make predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Flatten the tensors and move them to CPU for further processing
        masks = masks.view(-1).cpu().numpy()
        preds = preds.view(-1).cpu().numpy()

        # Append the ground truth labels and predictions to the lists
        y_true.extend(masks)
        y_pred.extend(preds)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
