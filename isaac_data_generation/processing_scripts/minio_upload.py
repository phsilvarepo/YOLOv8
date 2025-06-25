import numpy as np
import os
import json
from PIL import Image
import argparse
import shutil
import random 
import minio
from minio import Minio
from minio.error import S3Error
import sys

image_filenames = []
label_filenames = []

def extract_classes(replicator_path, dataset_name, base_path):
    unique_classes = set()  # Store unique class names

    # Loop through all JSON files in the directory
    for filename in os.listdir(replicator_path):
        if filename.startswith("bounding_box_2d_tight_labels_") and filename.endswith(".json"):
            json_file_path = os.path.join(replicator_path, filename)

            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)

                # Extract class names from the JSON structure
                for key, value in data.items():
                    class_name = value.get("class", "unknown")  # Default to 'unknown' if no class found
                    unique_classes.add(class_name)

    # Write unique class names to a text file
    dataset_path = os.path.join(base_path, dataset_name)
    output_path = os.path.join(dataset_path, "labels.txt")
    with open(output_path, 'w') as txt_file:
        for class_name in sorted(unique_classes):  # Sort to keep it organized
            txt_file.write(class_name + "\n")

    print(f"Extracted {len(unique_classes)} unique classes to {output_path}")

    return len(unique_classes)

# Define AAS compliant dataset structure
def create_dataset_structure(dataset_name, base_path):
    dataset_path = os.path.join(base_path, dataset_name)
    
    # Define required directories
    folders = [
        "raw", "processed", "annotations", 
        "history", "train", "val", "test"
    ]
    
    # Create directories
    for folder in folders:
        os.makedirs(os.path.join(dataset_path, folder), exist_ok=True)
    
    # Create empty JSON metadata files
    json_files = [
        "metadata.json", "metrics.json", "environmental-conditions.json", "contact-info.json"
    ]
    
    for json_file in json_files:
        file_path = os.path.join(dataset_path, json_file)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump({}, f)  # Empty JSON
    
    print(f"Dataset structure created at: {dataset_path}")

def populate_dataset(dataset_name, base_path, replicator_dataset_path):
    dataset_path = os.path.join(base_path, dataset_name)
    raw_path = os.path.join(dataset_path, "raw")
    annotations_path = os.path.join(dataset_path, "annotations")

    # Copy images and labels
    for filename in os.listdir(replicator_dataset_path):
        file_path = os.path.join(replicator_dataset_path, filename)

        if filename.endswith('.png'):  # Image files
            shutil.copy(file_path, raw_path)
            print(f"Copied image: {filename} -> {raw_path}")

        elif filename.endswith('.txt'):  # Label files
            shutil.copy(file_path, annotations_path)
            print(f"Copied label: {filename} -> {annotations_path}")

    print(f"Dataset population complete for {dataset_name}.")

def get_folder_size_gb(path):
    """Calculate the folder size in GB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp): 
                total_size += os.path.getsize(fp)
    return round(total_size / (1024**3), 2)  # Convert bytes to GB

def edit_metadata(dataset_name, base_path, minio_url, BUCKET_NAME, train_split_quota, test_split_quota, val_split_quota, number_classes):
    dataset_path = os.path.join(base_path, dataset_name)
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")

    train_size = get_folder_size_gb(train_path)
    test_size = get_folder_size_gb(test_path)
    val_size = get_folder_size_gb(val_path)
    complete_size = train_size + test_size + val_size

    metadata_content = {
        "AIDataset": {
            "URIOfTheProduct": f"https://{minio_url}/{BUCKET_NAME}/{dataset_name}",
            "Version": "1.0",
            "ContactInformation": {
                "Name": "John Doe",
                "Email": "johndoe@example.com"
            },
            "Storage": f"minio://{BUCKET_NAME}/{dataset_name}",
            "SizeInformation": {
                "CompleteSize": f"{complete_size}GB",
                "TrainSize": f"{train_size}GB",
                "ValSize": f"{val_size}GB",
                "TestSize": f"{test_size}GB",
                "SplitRatio": f"{train_split_quota}:{test_split_quota}:{val_split_quota}"
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
                "NumberOfClasses": f"{number_classes}",
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
    
    # Save metadata.json
    metadata_path = os.path.join(dataset_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata_content, f, indent=4)

    print(f"Dataset structure created with metadata at: {dataset_path}")

def upload_minio_bucket(dataset_name, base_path, MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, BUCKET_NAME):

    # Initialize MinIO client
    minio_client = Minio(
        MINIO_URL,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False  # Set to True if using HTTPS
    )

    dataset_path = os.path.join(base_path, dataset_name)

    # Ensure the bucket exists, create if not
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' created.")
    else:
        print(f"Bucket '{BUCKET_NAME}' already exists.")
    
    # Function to upload files recursively
    def upload_directory(local_dir, bucket_prefix):
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, dataset_path)  # Keep dataset structure
                object_name = os.path.join(bucket_prefix, relative_path).replace("\\", "/")  # Ensure MinIO format

                try:
                    minio_client.fput_object(BUCKET_NAME, object_name, local_file_path)
                    print(f"Uploaded {local_file_path} to {object_name}")
                except S3Error as e:
                    print(f"Error uploading {file}: {e}")

    # Start uploading dataset folder
    upload_directory(dataset_path, dataset_name)

    print(f"Successfully uploaded dataset '{dataset_name}' to MinIO bucket '{BUCKET_NAME}'.")

# Main execution
def main():
    parser = argparse.ArgumentParser(description="Create MinIO dataset(AAS Compliant)")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--output_path", type=str, default=os.getcwd(), help="Path where dataset will be created")
    parser.add_argument("--train_split_ratio", type=int, default=70, help="Train split percentage of the dataset")
    parser.add_argument("--test_split_ratio", type=int, default=0, help="Test split percentage of the dataset")
    parser.add_argument("--val_split_ratio", type=int, default=30, help="Validation split percentage of the dataset")
    parser.add_argument("--minio_url", type=str, default="127.0.0.1:9000", help="URL of MinIO")
    parser.add_argument("--minio_bucket_name", type=str, default="image-datasets", help="MinIO bucket name")
    parser.add_argument("--minio_access_key", type=str, default="minioadmin", help="Access key of MinIO")
    parser.add_argument("--minio_secret_key", type=str, default="minioadmin", help="Secret key of MinIO")
    args = parser.parse_args()

    if not os.path.exists(args.input_replicator_data_path):
        print(f"Warning: Replicator data directory '{args.input_replicator_data_path}' does not exist.")
        sys.exit(1)

    total_ratio = args.train_split_ratio + args.test_split_ratio + args.val_split_ratio
    if total_ratio != 100:
        print(f"Error: Train, test, and validation split ratios must sum to 100%. Current sum: {total_ratio}")
        sys.exit(1)  # Exit the program with an error code

    create_dataset_structure(args.dataset_name, args.output_path)
    number_classes = extract_classes(args.input_replicator_data_path, args.dataset_name, args.output_path)
    populate_dataset(args.dataset_name, args.output_path, args.input_replicator_data_path)
    edit_metadata(args.dataset_name, args.output_path, args.minio_url, args.minio_bucket_name, args.train_split_ratio, args.test_split_ratio, args.val_split_ratio, number_classes)
    upload_minio_bucket(args.dataset_name, args.output_path, args.minio_url, args.minio_access_key, args.minio_secret_key, args.minio_bucket_name)

if __name__ == "__main__":
    main()