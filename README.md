# YOLOv8

This package contains the necessary scripts for pre processing of the YOLOv8 input data. 

If it is necessary to generate synthetic data, the current approach is to use Omniverse Replicator to attain the data. The isaac_data_generation folder contains the necessary scripts to convert the numpy arrays to the YOLO annotaion format (.txt labels). Besides this there are also scrpits to change the filenames in case of matching names and an script to split the data into training and validation sets.

##rename.py
Parameters:

## rename.py
Script to rename files by adding a prefix to the filenamenes
Parameters:
path: dataset path
prefix: prefix name
