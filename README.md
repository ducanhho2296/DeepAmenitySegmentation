# DeepAmenitySegmentation
DeepAmenitySegmentation is a deep learning project focused on recognizing amenities in OpenStreetMap data using satellite imagery. The primary objective is to classify buildings in urban environments into different amenity categories, such as retail, food, school, healthcare, entertainment, public, and leisure, using semantic segmentation techniques.

## Overview
This project uses a custom semantic segmentation model based on EfficientNet and a dataset of satellite images with corresponding OpenStreetMap (OSM) geo spacial data using Osmnx library. The project focused on detecting and classifying various urban amenities such as retail, food, school, healthcare, entertainment, public, and leisure facilities in satellite images, by utilizing state-of-the-art semantic segmentation techniques to identify building structures and recognize their amenity classes, providing valuable insights for urban planning and development.

## Features
### Custom dataset class for loading satellite images and label images
- Using Osmnx to extract geo spatial data from open street map into a Geodataframe and using Geopandas to extract geo values of amenities in map
<img src="https://user-images.githubusercontent.com/92146886/219333765-b746ee07-e997-42bd-b49d-64c31464274a.png" alt="download" style="width:512px;">

## Train Amenity Segmentation Model

In train.py, a segmentation model will be trained to classify amenity points in satellite images of a city. The model is trained on padded square images of each region inside city with fixed size and the corresponding masks.

### Installation

To train model with `train.py`, it is recommend to a new Python environment, that is separate from the environment used for creating datasets to avoid conflicts. The packages required to run this script are listed in the `requirements.txt` file. You can install the packages by running the following command:

```bash
pip install -r requirements.txt
```

### Usage

1. Set the configuration parameters in the `config.ini` file, including paths, image processing parameters, and training parameters.

2. Run the following command to train the model:

```bash
python train.py --model <model_type> --batch <batch_size> --epoch <num_epochs> --gpu <gpu_index>
```

- `model_type`: The type of the model architecture to use. The two options are `unet` and `deeplabv3plus`.
- `batch_size`: The batch size used during training. The default is `8`.
- `num_epochs`: The number of epochs to train for. The default is `50`.
- `gpu_index`: The index of the GPU to use for training. The default is `0`.

3. After the training is completed, the trained model will be saved in the `model` directory with the name `modeltype_"cityname"_amenity_classification.pth`.

## Hyperparameter Tuning and Monitoring with MLflow

### Hyperparameter Tuning with MLflow
the `MLflow_tuning.py` script was created and is located in the mlflow_optimization folder, this script run the `train.py` script with different combinations of hyperparameters, such as learning rates and batch sizes to get the best hyperparameters during training experiment.

### Monitor Results in the MLflow UI
After running the `tune_hyperparameters.py` script, launch the MLflow UI with `mlflow ui` and navigate to `http://localhost:5000`. Click on the experiment name to see the list of runs and their associated metrics and parameters.

### Extract the Best Hyperparameters
`get_best_hyperparameters.py` was created to use the MLflow Python API to get the best run based on the lowest validation loss and extract the optimized learning rate and batch size. Now we can use these optimized hyperparameters to continue training the model from the best checkpoint.
