# Generating square bbox images 

In this project, there are two types to sampling dataset:
1. Grid Sampling
2. Amenity Stratified Sampling

Colab link for Visualization of sampling methods:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1evkK3QimFu-sQqUFDyBWpqWFfUPSmFod?usp=sharing)
#### Grid Sampling
The grid sampling method involves dividing the square region into a grid and sampling buildings from specific grid points. The steps involved in grid sampling are as follows:

- Divide the square region into a grid by selecting a specific grid size (spacing_x, spacing_y in `config.ini`).
- For each grid point, find another 3 adjacent grid points, with 4 grid points as 4 corners of a bounding box, create this bounding box using the `Polygon()` function from the `Shapely` library.
- Retrieve and plot all buildings within the bounding box using OSMnx.
- Save the plotted figure of the bounding box, representing the buildings, as a dataset image.

#### Amenity Stratified Sampling
The amenity stratified sampling method involves selecting buildings based on a specific amenity point within the square region. The steps involved in amenity stratified sampling are as follows:

- Choose an amenity point within city.
- Generate a bounding box around the amenity point using the `bbox_from_point()` function from the `OSMnx` library.
- Retrieve and plot all buildings within the bounding box using OSMnx.
- Save the plotted figure of the bounding box, representing the buildings, as a dataset image.

This Python script, `generate_dataset.py`, generates square bounding box images from high-resolution satellite imagery using these two techniques: Grid-sampling and Amenity-sampling. The generated images are labeled using `rasterio` package to create label image of building footprints, each pixel of label image represents a class of amenity inside city.

#### Example:
| Amenity Stratified Sampling| Grid Sampling                           |
| --------------------------------- | --------------------------------- |
| ![image908](https://github.com/ducanhho2296/DeepAmenitySegmentation/assets/92146886/d09c9bfc-1d6c-464e-9343-7e0c47d6d835) | ![image52](https://github.com/ducanhho2296/DeepAmenitySegmentation/assets/92146886/1d4cad48-dd3b-48cd-b2ff-0e17f9b62f19) |


- Note: The bounding box created by Grid Sampling has different size in comparison with that of Amenity Sampling, resulting in the buildings within the Grid Sampling bounding box also being of different size.


## Prerequisites

To run this script without interfering with the environment used for training your model, it is recommended that you create a new Python environment specifically for this script and install the packages listed in the req.txt file. To install these packages, you can run the following command in the command line:

```
pip install -r req.txt
```

You can install these packages using pip or conda package manager.

## Configuration

The script reads the configuration information from the `config.ini` file located in the parent directory of the script. The configuration file contains the following sections and options:

### Paths

- `root_path`: The root directory where the generated images and labels will be saved.
- `padded_img_dir`: The directory where the padded images will be saved.
- `padded_label_dir`: The directory where the padded labels will be saved.

### City

- `name`: The name of the city for which the dataset is generated, the modification of the name of city can be done with config.ini.
- `dist`: The distance from the center point to the edge of the bounding box in meters.

### Grid

- `spacing_x`: The horizontal spacing between the grid points in degrees.
- `spacing_y`: The vertical spacing between the grid points in degrees.

### Processing

- `img_size`: The size of the generated square bounding box images in pixels.

## Running the script

To run the script, simply execute the following command:

```
python generate_dataset.py
```

The script will generate square bounding box images using two techniques: Grid-sampling and Amenity-stratified-sampling. The generated images and labels will be saved in the directories specified in the configuration file.

## Note

This script overwrites the `num_classes` value in the `config.ini` file with the new value obtained from the `label_image` function. This is done for the purpose of using this value during the training process. If you do not want to overwrite this value, you can comment out the relevant code in the script.
