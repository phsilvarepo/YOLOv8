import time
import rospy
import ros_numpy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
detection_model = YOLO("/home/rics/catkin_ws/src/yolov8_ros/src/yolov8m.pt")

# Initialize ROS node
rospy.init_node("ultralytics")
time.sleep(1)

# Publishers
det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5)
bbox_pub = rospy.Publisher("/ultralytics/detection/bounding_boxes", String, queue_size=5)

def callback(data):
    """Callback to process image and publish detection results."""
    # Convert ROS Image message to NumPy array
    image_np = ros_numpy.numpify(data)

    if det_image_pub.get_num_connections() > 0:
        # Run inference
        det_result = detection_model(image_np, conf=0.4) # Confidence Threshold
        det_annotated = det_result[0].plot()  
        # Publish annotated image (RGB)
        det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

        # Extract class names
        classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
        names = [det_result[0].names[i] for i in classes]
        classes_pub.publish(String(data=str(names)))

        # Extract bounding boxes (xywh format)
        boxes = det_result[0].boxes.xywh.cpu().numpy()
        yolo_bounding_boxes = []

        for i, box in enumerate(boxes):
            bbox_info = {
                "center_x": float(box[0]),
                "center_y": float(box[1]),
                "width": float(box[2]),
                "height": float(box[3]),
                "classname": names[i]
            }
            yolo_bounding_boxes.append(bbox_info)

        # Publish bounding box info
        bbox_pub.publish(String(data=str(yolo_bounding_boxes)))

# Subscribe to image topic
rospy.Subscriber("/rgb", Image, callback)

# Spin to keep node alive
rospy.spin()
