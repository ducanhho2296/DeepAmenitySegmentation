import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import get_transforms, SatelliteDataset
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from models.model.get_model import *


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='unet', help='Model type (unet or deeplabv3plus)')
parser.add_argument('--weight', type=int, default=1, help='weight number')
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

num_classes = int(config['training_params']['num_classes'])

# Construct the full paths to the image and label directories
image_dir = os.path.join(root_path, test_image_dir)
label_dir = os.path.join(root_path, test_label_dir)

# Set the device to use for training
gpu = args.gpu
device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

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

# Create a colormap
cmap = ListedColormap([
    "#000000",  # NaN, no amenity, color: black
    "#FF0000",  # retail, color: red
    "#00FF00",  # food, color: green
    "#0000FF",  # school, color: blue
    "#FFFF00",  # healthcare, color: yellow
    "#00FFFF",  # entertainment, color: cyan
    "#FF00FF",  # public, color: magenta
    "#C0C0C0",  # leisure, color: silver
    "#800000",  # sport, color: maroon
    "#808000",  # highway, color: olive
])
norm = BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)

# Function for converting tensor image to PIL Image
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = image.float()
    image = transforms.ToPILImage()(image)
    return image

# Generate predicted output
model.eval()  # Ensure the model is in evaluation mode
count = 0
with torch.no_grad():
    for i, (images, masks) in enumerate(test_loader):
        count += 1
        images = images.to(device).float()
        masks = masks.to(device)

        # Make predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Convert tensors to numpy arrays for plotting
        original_image = images[0].cpu().numpy().transpose(1, 2, 0)
        mask_image = masks[0].cpu().numpy()
        predicted_image = preds[0].cpu().numpy()

        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(20, 20), constrained_layout=True)

        axs[0].imshow(original_image)
        axs[0].set_title('Original Image')

        axs[1].imshow(mask_image, cmap=cmap, norm=norm)
        axs[1].set_title('Mask Image')

        axs[2].imshow(predicted_image, cmap=cmap, norm=norm)
        axs[2].set_title('Predicted Image')

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.savefig(f'segmentation_result_{i}.png')
        plt.close()  # Close the figure after saving to free up memory
        if count >= 2:
            break