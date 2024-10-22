import numpy as np
import os
import json

IMAGE_WIDTH = 1080  # Replace with the actual image width
IMAGE_HEIGHT = 1080
directory_path = "/home/rics/omni.replicator_out/stand_environment"

for filename in os.listdir(directory_path):
    if filename.endswith('.npy'):
        # Create the new JSON filename
        np_filename = os.path.splitext(filename)[0]
        numeric_part = np_filename[-4:]  # Extract the last 4 characters (assumes it's always numeric)
        json_filename = np_filename[:-4] + "labels_" + numeric_part + ".json"

        print(json_filename)

        # Construct the path to the JSON file
        json_file_path = os.path.join(directory_path, json_filename)

        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

            for key, value in data.items():
                numeric_value = key
                class_name = value["class"]
                print("\nNumeric Value:" + numeric_value + " Class Name: " + class_name)

        # Normalize bounding box coordinates
        bb_info = np.load(os.path.join(directory_path, filename))

        # Print the loaded array
        print(bb_info)
        print(len(bb_info))
        yolo_path = os.path.join(directory_path, 'rgb_' + numeric_part + '.txt')
        print(yolo_path)

        for item in bb_info:
            semantic_id = int(item['semanticId'])  # Convert the class ID to an integer
            x_min = item['x_min']
            y_min = item['y_min']
            x_max = item['x_max']
            y_max = item['y_max']

            x_center = (x_min + x_max) / (2 * IMAGE_WIDTH)
            y_center = (y_min + y_max) / (2 * IMAGE_HEIGHT)
            width = (x_max - x_min) / IMAGE_WIDTH
            height = (y_max - y_min) / IMAGE_HEIGHT

            annotation_line = "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                semantic_id, x_center, y_center, width, height)
            with open(yolo_path, 'a') as annotation_file:
                annotation_file.write(annotation_line + '\n')
