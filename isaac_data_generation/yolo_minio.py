import os
import json
import shutil
import random
from pathlib import Path
import numpy as np
from minio import Minio
from minio.error import S3Error
import argparse

IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 1080

def convert_npy_to_yolo_txt(directory_path):
    print("🔄 Converting .npy bounding boxes to YOLO .txt labels...")
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

            bb_info = np.load(os.path.join(directory_path, filename), allow_pickle=True)
            yolo_path = os.path.join(directory_path, 'rgb_' + numeric_part + '.txt')

            # Clear the file before writing
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

                line = f"{semantic_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                with open(yolo_path, 'a') as f:
                    f.write(line + '\n')
    print("✅ Conversion complete.\n")

def split_yolo_dataset(dataset_dir, output_dir, train_ratio, test_ratio, val_ratio, move_files=False):
    print(f"🔄 Splitting dataset into train/test/val (ratios: train={train_ratio}, test={test_ratio}, val={val_ratio})...")

    images_out = os.path.join(output_dir, 'images')
    labels_out = os.path.join(output_dir, 'labels')

    subsets = []
    if train_ratio > 0:
        subsets.append('train')
    if test_ratio > 0:
        subsets.append('test')
    if val_ratio > 0:
        subsets.append('val')

    for subset in subsets:
        Path(os.path.join(images_out, subset)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(labels_out, subset)).mkdir(parents=True, exist_ok=True)

    all_files = os.listdir(dataset_dir)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)

    total = len(image_files)
    train_end = int(total * train_ratio)
    test_end = train_end + int(total * test_ratio)

    train_files = image_files[:train_end] if train_ratio > 0 else []
    test_files = image_files[train_end:test_end] if test_ratio > 0 else []
    val_files = image_files[test_end:] if val_ratio > 0 else []

    def move_or_copy(file_list, subset):
        for img_file in file_list:
            label_file = os.path.splitext(img_file)[0] + '.txt'

            src_img = os.path.join(dataset_dir, img_file)
            src_lbl = os.path.join(dataset_dir, label_file)

            dst_img = os.path.join(images_out, subset, img_file)
            dst_lbl = os.path.join(labels_out, subset, label_file)

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
                print(f"⚠️ No label found for {img_file}")

    if train_ratio > 0:
        move_or_copy(train_files, 'train')
    if test_ratio > 0:
        move_or_copy(test_files, 'test')
    if val_ratio > 0:
        move_or_copy(val_files, 'val')

    print(f"✅ Split done: {len(train_files)} train, {len(test_files)} test, {len(val_files)} val images.\n")

def create_aas_structure(dataset_path):
    print("🔄 Creating AAS-compliant dataset structure...")
    folders = ["raw", "processed", "annotations", "history", "train", "val", "test"]
    for folder in folders:
        os.makedirs(os.path.join(dataset_path, folder), exist_ok=True)

    json_files = [
        "metadata.json", "metrics.json", "environmental-conditions.json", "contact-info.json"
    ]
    for jf in json_files:
        path = os.path.join(dataset_path, jf)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump({}, f)
    print(f"✅ Structure created at {dataset_path}\n")

def populate_aas_dataset(yolo_dataset_dir, aas_dataset_dir):
    print("🔄 Populating AAS dataset with YOLO images and labels...")

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(yolo_dataset_dir, 'images', split)
        if not os.path.exists(split_path):
            continue  # Skip if not created (test_ratio might be 0)

        aas_split_path = os.path.join(aas_dataset_dir, split)
        os.makedirs(aas_split_path, exist_ok=True)

        images_src = os.path.join(yolo_dataset_dir, 'images', split)
        labels_src = os.path.join(yolo_dataset_dir, 'labels', split)

        for img_file in os.listdir(images_src):
            shutil.copy2(os.path.join(images_src, img_file), os.path.join(aas_split_path, img_file))

        for label_file in os.listdir(labels_src):
            shutil.copy2(os.path.join(labels_src, label_file), os.path.join(aas_split_path, label_file))

    print("✅ Dataset populated with train/val/test splits.\n")

def get_folder_size_gb(path):
    total_size = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return round(total_size / (1024**3), 2)

def extract_classes_from_jsons(json_dir):
    unique_classes = set()
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(json_dir, filename), 'r') as f:
                    data = json.load(f)

                    # If data is a dict
                    if isinstance(data, dict):
                        iterable = data.values()
                    # If data is a list
                    elif isinstance(data, list):
                        iterable = data
                    else:
                        continue  # Skip unknown structures

                    for item in iterable:
                        if isinstance(item, dict):
                            class_name = item.get("class")
                            if class_name:
                                unique_classes.add(class_name)
            except Exception as e:
                print(f"⚠️ Failed to parse {filename}: {e}")
    return unique_classes

def edit_metadata(dataset_path, minio_url, bucket_name, train_split, val_split, number_classes):
    test_split = 100 - train_split - val_split

    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")

    train_size = get_folder_size_gb(train_path)
    val_size = get_folder_size_gb(val_path)
    test_size = get_folder_size_gb(test_path)
    complete_size = train_size + val_size + test_size

    metadata = {
        "AIDataset": {
            "URIOfTheProduct": f"https://{minio_url}/{bucket_name}/{os.path.basename(dataset_path)}",
            "Version": "1.0",
            "ContactInformation": {
                "Name": "John Doe",
                "Email": "johndoe@example.com"
            },
            "Storage": f"minio://{bucket_name}/{os.path.basename(dataset_path)}",
            "SizeInformation": {
                "CompleteSize": f"{complete_size}GB",
                "TrainSize": f"{train_size}GB",
                "TestSize": f"{test_size}GB",
                "ValSize": f"{val_size}GB",
                "SplitRatio": f"{train_split}:{test_split}:{val_split}"
            },
            "MetaData": {
                "FileType": "PNG",
                "AdditionalInformation": {
                    "Description": "Synthetic imagery for AI model training",
                    "Source": "Isaac Sim",
                    "GeographicalRegion": "Virtual"
                }
            },
            "Metrics": {
                "NumberOfClasses": str(number_classes),
                "Balance": "2:1",
                "MeanPixelIntensity": 128.5
            },
            "BoundaryConditions": {
                "DataCollectors": "DJI Phantom 4 Drone",
                "EnvironmentConditions": {
                    "Weather": "Sunny",
                    "Temperature": "25°C"
                }
            }
        }
    }

    metadata_path = os.path.join(dataset_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Metadata saved at {metadata_path}\n")

def upload_minio_bucket(dataset_path, minio_url, access_key, secret_key, bucket_name):
    print("🔄 Uploading dataset to MinIO...")
    client = Minio(
        minio_url,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )

    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print(f"Created bucket: {bucket_name}")
    else:
        print(f"Bucket {bucket_name} exists.")

    def upload_dir(local_dir, bucket_prefix):
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, dataset_path).replace("\\", "/")
                object_name = os.path.join(bucket_prefix, relative_path).replace("\\", "/")
                try:
                    client.fput_object(bucket_name, object_name, local_file_path)
                    print(f"Uploaded {object_name}")
                except S3Error as e:
                    print(f"Error uploading {object_name}: {e}")

    upload_dir(dataset_path, os.path.basename(dataset_path))
    print("✅ Upload complete.\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path with .npy, .json and images")
    parser.add_argument("--output_path", required=True, help="Output base path for YOLO + AAS dataset")
    parser.add_argument("--dataset_name", required=True, help="Dataset name")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.0, help="Test split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Val split ratio")
    parser.add_argument("--minio_url", default="127.0.0.1:9000")
    parser.add_argument("--minio_bucket", default="image-datasets")
    parser.add_argument("--minio_access_key", default="minioadmin")
    parser.add_argument("--minio_secret_key", default="minioadmin")
    args = parser.parse_args()

    # Step 1: Convert npy + JSON → YOLO labels
    convert_npy_to_yolo_txt(args.input_path)

    # Step 2: Split into train/val YOLO dataset
    yolo_dataset_dir = os.path.join(args.output_path, args.dataset_name + "_yolo")
    os.makedirs(yolo_dataset_dir, exist_ok=True)
    split_yolo_dataset(args.input_path, yolo_dataset_dir, train_ratio=args.train_ratio, test_ratio=args.test_ratio, val_ratio=args.val_ratio, move_files=True)

    # Step 3: Create AAS compliant dataset structure
    aas_dataset_dir = os.path.join(args.output_path, args.dataset_name)
    create_aas_structure(aas_dataset_dir)

    # Step 4: Populate AAS dataset with YOLO split data
    populate_aas_dataset(yolo_dataset_dir, aas_dataset_dir)

    # Step 5: Extract classes count for metadata
    classes = extract_classes_from_jsons(args.input_path)
    num_classes = len(classes)

    # Step 6: Write metadata.json
    edit_metadata(aas_dataset_dir, args.minio_url, args.minio_bucket, 
                  train_split = int(args.train_ratio * 100), val_split = int(args.val_ratio * 100),
                  number_classes=num_classes)

    # Step 7: Upload to MinIO
    upload_minio_bucket(aas_dataset_dir, args.minio_url, args.minio_access_key, args.minio_secret_key, args.minio_bucket)

if __name__ == "__main__":
    main()

#python3 minio.py --input_path /home/rics/omni.replicator_out/battery --output_path /home/rics/Desktop/t --dataset_name Test