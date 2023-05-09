# Generating square bbox images with Grid-sampling and Random-sampling techniques

This Python script, `generate_dataset.py`, generates square bounding box images from high-resolution satellite imagery using two different techniques: Grid-sampling and Random-sampling. The generated images are labeled using the building footprints of the city.

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

The script will generate square bounding box images using two techniques: Grid-sampling and Random-sampling. The generated images and labels will be saved in the directories specified in the configuration file.

## Note

This script overwrites the `num_classes` value in the `config.ini` file with the new value obtained from the `label_image` function. This is done for the purpose of using this value during the training process. If you do not want to overwrite this value, you can comment out the relevant code in the script.