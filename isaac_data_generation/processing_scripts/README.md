# Other scripts

This package contains the necessary scripts for pre processing of the YOLOv8 input data. 

If it is necessary to generate synthetic data, the current approach is to use Omniverse Replicator to attain the data. The isaac_data_generation folder contains the necessary scripts to convert the numpy arrays to the YOLO annotation format (.txt labels). Besides this there are also scrpits to change the filenames in case of matching names and an script to split the data into training and validation sets.

## np_converter.py
Script to convert numpy annotation files to YOLO annotation format(.txt)

Parameters:
- **directory_path** : Path to the dataset directory
- **IMAGE_WIDTH** : Generated images width (1080, default)
- **IMAGE_HEIGHT** : Generated images height (1080, default)

Parameters:
- **directory_path** : Path to the dataset directory
- **prefix** : Prefix name

## convert_labels.py
Script to convert labels to according to a standart class ids

Parameters:

- **directory_path** : Path to the dataset directory
- **class_mapping** : Class dictionary

## rename.py
Script to rename files by adding a prefix to the filenames

Parameters:
- **directory_path** : Path to the dataset directory
- **prefix** : Prefix name
  
## split.py
Script to split dataset into train and validation set

Parameters:

- **dataset_dir** : Path to dataset directory
- **train_dir** : Path to train directory
- **test_dir** : Path to test directory
- **train_ratio** : Ratio of train/validation split

## minio_upload.py
Script to uplaod YOLO dataset into MinIO, in a AAS compliant format

Parameters:

- **dataset_dir** : Path to dataset directory
- **train_dir** : Path to train directory
- **test_dir** : Path to test directory
- **train_ratio** : Ratio of train/validation split

