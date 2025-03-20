import time

import ros_numpy
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ultralytics import YOLO

detection_model = YOLO("/home/rics/catkin_ws/src/yolov8_ros/src/yolov8m.pt")
rospy.init_node("ultralytics")
time.sleep(1)

det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5)
bbox_pub = rospy.Publisher("/ultralytics/detection/bounding_boxes", String, queue_size=5)  # New publisher

def callback(data):
    """Callback function to process image and publish annotated images."""
    array = ros_numpy.numpify(data)
    if det_image_pub.get_num_connections():
        det_result = detection_model(array)
        det_annotated = det_result[0].plot()
        det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

        classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
        names = [det_result[0].names[i] for i in classes]
        classes_pub.publish(String(data=str(names)))

        boxes = det_result[0].boxes.xywh.cpu().numpy()  # (center_x, center_y, width, height)
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
        
        # Publish bounding box information as JSON string
        bbox_pub.publish(String(data=str(yolo_bounding_boxes)))

rospy.Subscriber("/dock_rgb", Image, callback)

while True:
    rospy.spin()
