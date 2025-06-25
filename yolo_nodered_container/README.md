# YOLO Docker Container

Build **ROS Noetic Ubuntu 20.04 Docker** image and the necessary dependencies to run the **yolov8_ros** package :

 - docker build -t yolov8_ros -f yolov8_ros.dockerfile .

Run the **Docker** container:

 - docker run -it --rm --net=host --gpus all --privileged yolov8_ros

This container utilizes the host's IP and GPU resources to run YOLO, through torch
