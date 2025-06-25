# YOLO + Omniverse Replicator

This package contains the necessary scripts for post processing of the output of Omniverse Replicator for the YOLOv8 input data, and AAS compliant translation.

## yolo_dataset.py
Script to convert synthetic images and annotations from Isaac Sim to the YOLO format dataset

Parameters:
- **dataset_dir** : Path to the Isaac Sim dataset directory
- **output_dir** : Path to the output dataset directory location
- **train_ratio** : Split between the train/validation set, default = 0.8

## yolo_minio.py
Script to convert synthetic images and annotations from Isaac Sim to the YOLO format dataset and also upload to MinIO an compatible AAS compliant dataset.
// python3 yolo_minio.py --input_path /home/rics/omni.replicator_out/battery --output_path /home/rics/Desktop/t --dataset_name Test

Parameters:
- **input_path** : Path to the Isaac Sim dataset directory
- **output_path** : Path to the output dataset directory location
- **dataset_name** : dataset name
- **more_parameters**: train/test/validation split, minio url/bucket
