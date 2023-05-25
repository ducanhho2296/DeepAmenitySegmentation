import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from load_datasets import get_transforms, SatelliteDataset

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from models.model.get_model import *
# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='unet', help='Model type (unet or deeplabv3plus)')
parser.add_argument('--weight', type=int, default=None, help='weight number')

parser.add_argument('--batch', type=int, default=8, help='Batch size')
parser.add_argument('--gpu', type=int, default=0, help='specific gpu for evaluation')

args = parser.parse_args()

# Load the configuration file
import configparser
config = configparser.ConfigParser()
config.read('config.ini')


# Get the paths and other parameters from the configuration file
root_path = os.path.abspath(os.path.join(parent_dir, config.get('paths', 'root_path')))
model_weight_path = config['paths']['model_path']
test_image_dir = config['paths']['image_test']
test_label_dir = config['paths']['label_test']

city_name = config['city']['name']
num_classes = int(config['training_params']['num_classes'])

# Construct the full paths to the image and label directories
image_dir = os.path.join(root_path, test_image_dir)
label_dir = os.path.join(root_path, test_label_dir)

# Set the device to use for training
gpu = args.gpu
device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')


# Load the specified weight file if provided
if args.weight == None:
    weight_num = 1
else: weight_num = args.weight

# Load the trained model
model_name = f"{args.model}_weight_{weight_num}_segmentation.pth"
model_path = os.path.join(root_path, model_weight_path, model_name)
checkpoint = torch.load(model_path)

if args.model == 'unet':
    model = get_unet_model(device, in_channels=3, num_classes=num_classes)
elif args.model == 'deeplabv3plus':
    model = get_deeplabv3plus_model(device, in_channels=3, num_classes=num_classes)
else:
    raise ValueError('Invalid model type')

model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.replace('module.', '') in model.state_dict()})
model.eval()

# Create the test dataset
batch_size = args.batch
test_dataset = SatelliteDataset(image_dir, label_dir, get_transforms())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Calculate the evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted',zero_division=1)
recall = recall_score(y_true, y_pred, average='weighted',zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted',zero_division=1)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
