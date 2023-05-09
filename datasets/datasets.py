import os
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SatelliteDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_ids = sorted(os.listdir(image_dir))
        self.image_ids = sorted([f for f in os.listdir(image_dir) if re.match(r'.+\.(jpg|png|tif)$', f)])
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)

        image = np.array(Image.open(image_path).convert("L"))  # Convert the image to grayscale
        image = np.stack((image,) * 3, axis=-1)  # Duplicate the grayscale channel to create a 3-channel image

        mask_id = f"label{image_id[5:-4]}.tif"  # Create the mask filename using the extracted image name
        mask_path = os.path.join(self.mask_dir, mask_id)
        mask = np.array(Image.open(mask_path))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

def create_train_val_datasets(image_dir, label_dir, train_val_split_ratio, transform):
    dataset = SatelliteDataset(image_dir, label_dir, transform)
    dataset_size = len(dataset)
    train_size = int(train_val_split_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def get_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Resize(512, 512),
        ToTensorV2()  # Convert the image and mask to PyTorch tensors
    ])


if __name__ == "__main__":
    image_path = "/home/ducanh/DeepAmenitySegmentation/datasets/images"
    label_path = "/home/ducanh/DeepAmenitySegmentation/datasets/labels"
    S = SatelliteDataset(image_dir=image_path, mask_dir=label_path, transform=get_transforms())

    # Iterate over the dataset and display some information
    for idx in range(len(S)):
        image, mask = S[idx]
        print(f"Image {idx}: shape {image.shape}, dtype {image.dtype}")
        print(f"Mask {idx}: shape {mask.shape}, dtype {mask.dtype}")