import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import create_train_val_datasets, get_transforms
import configparser
import mlflow

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from models.model.get_model import *


def train(model_type, learning_rate, batch_size, num_epochs, gpu, weight):
    # Load the configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Get the paths and other parameters from the configuration file
    root_path = os.path.abspath(os.path.join(parent_dir, config.get('paths', 'root_path')))
    model_weight_path = config['paths']['model_path']
    padded_img_dir = config['paths']['padded_img_dir']
    padded_label_dir = config['paths']['padded_label_dir']
    city_name = config['city']['name']
    num_classes = int(config['training_params']['num_classes'])

    # Construct the full paths to the image and label directories
    image_dir = os.path.join(root_path, padded_img_dir)
    label_dir = os.path.join(root_path, padded_label_dir)

    # Set the training-validation split ratio
    train_val_split_ratio = float(config['training_params']['split_ratio'])

    # Create the training and validation datasets
    train_dataset, val_dataset = create_train_val_datasets(image_dir, label_dir, train_val_split_ratio, get_transforms())

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Set the device to use for training
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')


    # Define the model architecture
    if model_type == 'unet':
        model = get_unet_model(device, in_channels=3, num_classes=num_classes)
    elif model_type == 'deeplabv3plus':
        model = get_deeplabv3plus_model(device, in_channels=3, num_classes=num_classes)
    else:
        raise ValueError('Invalid model type')

    # Create the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load the specified weight file if provided
    last_epoch = 0
    if weight == None:
        weight_num = 1
    else: weight_num = weight

    model_name = f"{model_type}_weight_{weight_num}_segmentation.pth"
    model_path = os.path.join(root_path, model_weight_path, model_name)

    if weight:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['epoch']
        else:
            print(f"Weight file not found at: {model_path}")
            sys.exit("Please provide a valid weight file or train the model first.")

    

    # Train the model
    for epoch in range(last_epoch, num_epochs):
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

        # # Log the metrics to MLflow for monitoring training process
        # mlflow.log_metric("train_loss", train_loss, step=epoch)
        # mlflow.log_metric("val_loss", val_loss, step=epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the trained model
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, model_path)

    file_path = os.path.join(root_path, model_weight_path, "last_epochs.txt")

    with open(file_path, 'w') as file:
        file.write(str("weight number:{} has the last epoch is {}".format(weight_num, num_epochs)))
    
    return val_loss, train_loss

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', help='Model type (unet or deeplabv3plus)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--gpu', type=int, default=0, help='specific gpu for training')
    parser.add_argument('--weight', type=int, default=None, help='Continue training from the last checkpoint')
    
    args = parser.parse_args()

    # Call the train function with command-line arguments
    train(args.model, args.learning_rate, args.batch, args.epoch, args.gpu, args.weight)
