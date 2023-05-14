# DeepAmenitySegmentation
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

DeepAmenitySegmentation is a deep learning project for recognizing amenities in OpenStreetMap data using satellite imagery. It classifies urban buildings into amenity categories like retail, food, school, healthcare, entertainment, public, and leisure using semantic segmentation techniques.

### Overview
The project employs a custom semantic segmentation model based on U-net architecture using pre-trained EfficientNet as Backbone model and train model with a satellite image dataset and OpenStreetMap (OSM) geospatial data using Osmnx library. It detects and classifies urban amenities with semantic segmentation to provide insights for urban planning and development.

### Features
#### Custom Dataset Class
- A custom dataset class loads satellite and label images.
- Osmnx extracts geospatial data from OpenStreetMap into a Geodataframe.
- Geopandas extracts geospatial values of amenities.
- Sampling techniques generate images of square regions within a city.
- Pandas extracts building footprints for labeling the image.

<img src="https://user-images.githubusercontent.com/92146886/219333765-b746ee07-e997-42bd-b49d-64c31464274a.png" alt="download" style="width:512px;">

## Train Amenity Segmentation Model
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

`train.py` trains and evaluates the segmentation model using PyTorch. The model is trained on square images of regions within a city and corresponding masks with pre-trained weights of Unet and EfficientNet from `segmentation-models-pytorch`.

### Installation

Create a new Python environment and install packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Usage

1. Configure parameters in `config.ini`.
2. Train the model:

```bash
python train.py --model <model_type> --batch <batch_size> --epoch <num_epochs> --gpu <gpu_index>  --continue_train <True/False> --weight <weight_num>
```

## Hyperparameter Tuning and Monitoring with MLflow 
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)

### Hyperparameter Tuning with MLflow
`mlflow_tuning.py` in the mlflow_optimization folder runs `train.py` with different hyperparameter combinations to find the best ones.

### Monitor Results in the MLflow UI
Launch the MLflow UI with `mlflow ui`, navigate to `http://localhost:5000`, and click on the experiment name to see the list of runs and their metrics and parameters.

### Extract the Best Hyperparameters
`mlflow_tuning.py` uses the MLflow Python API to find the best run based on the lowest validation loss, extract optimized learning rate and batch
