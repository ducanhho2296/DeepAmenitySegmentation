import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import create_train_val_datasets, get_transforms
import segmentation_models_pytorch as smp

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
# Load the configuration file
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

# Get the paths and other parameters from the configuration file
root_path = os.path.abspath(os.path.join(parent_dir, config.get('paths', 'root_path')))
model_weight_path = config['paths']['model_path']
padded_img_dir = config['paths']['padded_img_dir']
padded_label_dir = config['paths']['padded_label_dir']
city_name = config['city']['name']


# Construct the full paths to the image and label directories
image_dir = os.path.join(root_path, padded_img_dir)
label_dir = os.path.join(root_path, padded_label_dir)

# Set the training-validation split ratio
train_val_split_ratio = 0.8

# Create the training and validation datasets
train_dataset, val_dataset = create_train_val_datasets(image_dir, label_dir, train_val_split_ratio, get_transforms())

# Create the data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Set the device to use for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model, loss function, and optimizer
model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=8
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Set the number of epochs to train for
num_epochs = 50

# Train the model
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for images, masks in train_loader:
        images = images.to(device).float()
        masks = masks.to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks.type(torch.int64))

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device).float()
            masks = masks.to(device, dtype=torch.long)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the trained model
model_name = f"unet_efficientnet_{city_name}_amenity_classification.pth"
model_path = os.path.join(root_path, model_weight_path, model_name)
torch.save(model.state_dict(), model_path)
