import albumentations as A
import rasterio
import cv2
import os 

root_path = 'DeepAmenitySegmentation'
# Define the paths for images and labels
img_path = os.path.join('..', root_path, 'datasets', 'images', 'figure{}.jpg')
label_path = os.path.join('..', root_path, 'datasets', 'labels', 'label{}.tif')

aug_img_path = os.path.join('..', root_path, 'augmented_datasets', 'images', 'figure{}.jpg')
aug_label_path = os.path.join('..', root_path, 'augmented_datasets', 'labels', 'label{}.tif')
# define the augmentations
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
])

# load the input and label images
input_image = cv2.imread(img_path.format(1))
with rasterio.open(label_path.format(1), 'r') as src:
    # Read the label image data
    label_image = src.read(1)
# apply the same transformations to both input and label images
transformed = transform(image=input_image, mask=label_image)

# retrieve the augmented images
aug_input_image = transformed["image"]
aug_label_image = transformed["mask"]

# save the augmented images
cv2.imwrite("aug_input_image.jpg", aug_input_image)
cv2.imwrite("aug_label_image.tif", aug_label_image)
