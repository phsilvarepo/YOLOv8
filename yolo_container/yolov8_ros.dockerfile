FROM ubuntu:20.04

# Set noninteractive mode to avoid user prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install necessary packages
RUN apt-get update && apt-get install -y \
    lsb-release \
    curl \
    gnupg2 \
    build-essential \
    cmake \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Fix ROS repository key issue
RUN curl -sSL "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key" | apt-key add -

# Add ROS Noetic repository and install ROS Noetic base
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/ros-latest.list > /dev/null \
    && apt-get update \
    && apt-get install -y ros-noetic-ros-base \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies for building ROS packages
RUN apt-get update && apt-get install -y \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install rosnumpy ultralytics

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip3 install --upgrade --force-reinstall numpy

# Initialize rosdep
RUN rosdep init && rosdep update || echo "rosdep already initialized"

# Create Catkin workspace and build it
RUN mkdir -p /root/catkin_ws/src && cd /root/catkin_ws \
    && bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Copy your ROS package into the workspace
COPY yolov8_ros /root/catkin_ws/src/yolov8_ros

# Build the workspace again after adding the package
RUN bash -c "source /opt/ros/noetic/setup.bash && cd /root/catkin_ws && catkin_make"

# Create an entrypoint script to source ROS and workspace setup files
RUN echo '#!/bin/bash\nsource /opt/ros/noetic/setup.bash\nsource /root/catkin_ws/devel/setup.bash\nexec "$@"' > /ros_entrypoint.sh \
    && chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

