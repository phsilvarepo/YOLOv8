# YOLOv8

This repository contains scripts for the full YOLOv8 pipeline — from data collection and annotation to training and inference. It supports ROS integration, Docker-based deployment, and Node-RED integration.

## isaac_data_generation
The current approach to generate syhtnehtic images and annotation to trian the model, employs Omniverse Replicator to produce the data. The isaac_data_generation folder contains the necessary scripts to convert the output of the Replicaotr package , as numpy arrays to the YOLO annotation format (.txt labels). Besides these more crirtical scripts, there are also more scrpits to process the dataset, especially if several iterations of the omniverse pisodes are rquired (as the files are the smae and the class vary) 
  
## train
This folder includes a Google Colab notebook to train and validate a custom YOLOv8 model using your Google Drive.

Requirements:

- Dataset in YOLO format
- .yaml configuration file

Note: Trained model weights are not automatically saved to **Google Drive**. Make sure to manually download them after training.

## yolov8_ros

This ROS package enables real-time inference using a trained YOLOv8 model. The node susbribes to the sensor_msgs/Image in the **/rgb** topic. And outputs the follwoing topics: 

- **/ultralytics/detection/image** -- (sensor_msg/Image) An image of the the detected bounding boxes, classes and confidence score
- **/ultralytics/detection/classe** -- (String) Message with the detected classes
- **/ultralytics/detection/bounding_boxes** -- (String) Message with a detailed description of the bounding boxes, including center_x, center_y, width, height, and class name.

Requirements:

- Weight of the model(.pt)

## yolov8_container

This package includes the yolov8_ros ROS package, as well as the necessary files for deploying it in a **Docker** environment.

Requirements:

- Weight of the model(.pt)

## yolo_nodered_container

This package allows you to run a custom YOLOv8 model in the **Node-RED** environment using a **Docker** container deployment.

Requirements:

- Weight of the model(.json and shards)
