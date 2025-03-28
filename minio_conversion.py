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

def convert_labels(directory_path):
    unique_classes = set()  # Set to store unique class names
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.png'):  # Change to the appropriate image format
            image_filenames.append(filename)
            
            # Get the corresponding label filenames
            np_filename = os.path.splitext(filename)[0]
            numeric_part = np_filename[-4:]  # Extract the last 4 characters (assumes it's always numeric)
            json_filename = np_filename[:-4] + "labels_" + numeric_part + ".json"
            txt_filename = 'rgb_' + numeric_part + '.txt'
            
            label_filenames.append(json_filename)
            label_filenames.append(txt_filename)

    # Process `.npy` files and generate labels
    for filename in os.listdir(directory_path):
        if filename.endswith('.npy'):
            np_filename = os.path.splitext(filename)[0]
            numeric_part = np_filename[-4:]  # Extract the last 4 characters (assumes it's always numeric)
            json_filename = np_filename[:-4] + "labels_" + numeric_part + ".json"
            yolo_filename = 'rgb_' + numeric_part + '.txt'

            print(f"Generating label files for {filename}...")

            # Construct the path to the JSON file
            json_file_path = os.path.join(directory_path, json_filename)

            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)

                for key, value in data.items():
                    numeric_value = key
                    class_name = value["class"]
                    unique_classes.add(class_name)  # Add class name to the set
                    print("\nNumeric Value:" + numeric_value + " Class Name: " + class_name)

            # Normalize bounding box coordinates
            bb_info = np.load(os.path.join(directory_path, filename))
            yolo_path = os.path.join(directory_path, yolo_filename)

            # Image dimensions (get dimensions of the first image file)
            image_file = image_filenames[0]
            image_path = os.path.join(directory_path, image_file)
            with Image.open(image_path) as img:
                IMAGE_WIDTH, IMAGE_HEIGHT = img.size  # Get width and height

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

def remove_extra_files(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Keep only image files (.png) and their corresponding label files (.txt)
        if filename in image_filenames or filename in label_filenames:
            continue  # Skip deleting image and label files

        # Delete all other files
        try:
            os.remove(file_path)
            print(f"Deleted: {filename}")
        except Exception as e:
            print(f"Error deleting {filename}: {e}")

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


#DO TEST IMPLEMENTATION
def split_dataset(dataset_name, base_path, train_ratio, test_ratio):
    
    dataset_path = os.path.join(base_path, dataset_name)
    raw_path = os.path.join(dataset_path, "raw")
    annotations_path = os.path.join(dataset_path, "annotations")
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")

    # List all images in the dataset directory
    all_images = [f for f in os.listdir(raw_path) if f.endswith('.png')]
    
    # Shuffle images
    random.shuffle(all_images)

    # Compute split indices
    num_total = len(all_images)
    num_train = int(num_total * train_ratio)
    num_test = int(num_total * test_ratio)
    
    # Assign images to each split
    train_images = all_images[:num_train]
    test_images = all_images[num_train:num_train + num_test]
    val_images = all_images[num_train + num_test:]

    def move_files(image_list, src_folder, dest_folder):
        """Helper function to move images and labels."""
        for image in image_list:
            image_name, _ = os.path.splitext(image)
            src_image_path = os.path.join(src_folder, image)
            src_label_path = os.path.join(annotations_path, image_name + '.txt')

            dest_image_path = os.path.join(dest_folder, image)
            dest_label_path = os.path.join(dest_folder, image_name + '.txt')

            shutil.copy(src_image_path, dest_image_path)
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dest_label_path)

    # Move images and labels
    move_files(train_images, raw_path, train_path)
    move_files(test_images, raw_path, test_path)
    move_files(val_images, raw_path, val_path)

    print(f"Train set: {len(train_images)} images")
    print(f"Test set: {len(test_images)} images")
    print(f"Validation set: {len(val_images)} images")

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
    parser.add_argument("--input_replicator_data_path", type=str, default="/home/rics/omni.replicator_out/components_dell_3020_sff", help="Path to the dataset generated by Isaac Sim")
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

    train_ratio = args.train_split_ratio / 100
    test_ratio = args.test_split_ratio / 100

    create_dataset_structure(args.dataset_name, args.output_path)
    number_classes = extract_classes(args.input_replicator_data_path, args.dataset_name, args.output_path)
    convert_labels(args.input_replicator_data_path)
    remove_extra_files(args.input_replicator_data_path)
    populate_dataset(args.dataset_name, args.output_path, args.input_replicator_data_path)
    split_dataset(args.dataset_name, args.output_path, train_ratio, test_ratio)
    edit_metadata(args.dataset_name, args.output_path, args.minio_url, args.minio_bucket_name, args.train_split_ratio, args.test_split_ratio, args.val_split_ratio, number_classes)
    upload_minio_bucket(args.dataset_name, args.output_path, args.minio_url, args.minio_access_key, args.minio_secret_key, args.minio_bucket_name)

if __name__ == "__main__":
    main()