import numpy as np
import os
import json
import shutil
import random
from pathlib import Path

# Constants
IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 1080

def convert_npy_to_yolo_txt(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.npy'):
            np_filename = os.path.splitext(filename)[0]
            numeric_part = np_filename[-4:]
            json_filename = np_filename[:-4] + "labels_" + numeric_part + ".json"

            json_file_path = os.path.join(directory_path, json_filename)
            if not os.path.exists(json_file_path):
                print(f"⚠️ JSON file not found for {filename}, skipping...")
                continue

            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                # This part seems informational; you can optionally keep or remove
                for key, value in data.items():
                    numeric_value = key
                    class_name = value["class"]
                    print(f"Numeric Value: {numeric_value} Class Name: {class_name}")

            bb_info = np.load(os.path.join(directory_path, filename))

            yolo_path = os.path.join(directory_path, 'rgb_' + numeric_part + '.txt')
            # Clear file if it exists to avoid appending multiple times
            open(yolo_path, 'w').close()

            for item in bb_info:
                semantic_id = int(item['semanticId'])
                x_min = item['x_min']
                y_min = item['y_min']
                x_max = item['x_max']
                y_max = item['y_max']

                x_center = (x_min + x_max) / (2 * IMAGE_WIDTH)
                y_center = (y_min + y_max) / (2 * IMAGE_HEIGHT)
                width = (x_max - x_min) / IMAGE_WIDTH
                height = (y_max - y_min) / IMAGE_HEIGHT

                annotation_line = f"{semantic_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                with open(yolo_path, 'a') as annotation_file:
                    annotation_file.write(annotation_line + '\n')
    print("✅ Conversion from .npy to YOLO .txt done.")

def split_yolo_dataset(dataset_dir, output_dir, train_ratio=0.8, move_files=False):
    images_output = os.path.join(output_dir, 'images')
    labels_output = os.path.join(output_dir, 'labels')

    for subfolder in ['train', 'val']:
        Path(os.path.join(images_output, subfolder)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(labels_output, subfolder)).mkdir(parents=True, exist_ok=True)

    all_files = os.listdir(dataset_dir)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)

    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    def move_or_copy(file_list, subset):
        for img_file in file_list:
            label_file = os.path.splitext(img_file)[0] + '.txt'

            src_img = os.path.join(dataset_dir, img_file)
            src_lbl = os.path.join(dataset_dir, label_file)

            dst_img = os.path.join(images_output, subset, img_file)
            dst_lbl = os.path.join(labels_output, subset, label_file)

            if move_files:
                shutil.move(src_img, dst_img)
            else:
                shutil.copy2(src_img, dst_img)

            if os.path.exists(src_lbl):
                if move_files:
                    shutil.move(src_lbl, dst_lbl)
                else:
                    shutil.copy2(src_lbl, dst_lbl)
            else:
                print(f"⚠️ Warning: No label found for {img_file}")

    move_or_copy(train_files, 'train')
    move_or_copy(val_files, 'val')

    print("✅ Dataset split complete.")
    print(f"➡️ {len(train_files)} training images")
    print(f"➡️ {len(val_files)} validation images")

if __name__ == "__main__":
    dataset_dir = "/home/rics/omni.replicator_out/battery"
    output_dir = "/home/rics/Desktop/Test_Dataset"

    convert_npy_to_yolo_txt(dataset_dir)
    split_yolo_dataset(dataset_dir, output_dir, train_ratio=0.8, move_files=True)
