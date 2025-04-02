# YOLO Docker Container

docker build -t yolov8_ros -f yolov8_ros.dockerfile .

docker run -it --rm --net=host --gpus all --privileged yolov8_ros

This container utilizes the host's IP and GPU resources to run YOLO, through torch
