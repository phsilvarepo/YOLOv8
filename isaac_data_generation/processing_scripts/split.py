import os
import random
import shutil

# Define paths for your YOLO dataset and where you want to save the train and test splits
dataset_dir = '/home/rics/Desktop/Real_Screws_Dataset'
train_dir = '/home/rics/Desktop/Real_Screws_Dataset/img/train'
test_dir = '/home/rics/Desktop/Real_Screws_Dataset/img/val'

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the ratio of data to be used for training (e.g., 80%)
train_ratio = 0.8

# List all the image files in your dataset directory
all_images = [f for f in os.listdir(dataset_dir) if f.endswith('.JPG')]

# Calculate the number of images for the training set
num_train_images = int(len(all_images) * train_ratio)

# Randomly shuffle the list of images
random.shuffle(all_images)

# Split the dataset into train and test sets
train_images = all_images[:num_train_images]
test_images = all_images[num_train_images:]

# Copy images and their corresponding label files to the train and test directories
for image in train_images:
    image_name, _ = os.path.splitext(image)
    src_image_path = os.path.join(dataset_dir, image)
    src_label_path = os.path.join(dataset_dir, image_name + '.txt')
    
    dest_image_path = os.path.join(train_dir, image)
    dest_label_path = os.path.join(train_dir, image_name + '.txt')
    
    if os.path.exists(src_label_path):
        shutil.copy(src_image_path, dest_image_path)
        shutil.copy(src_label_path, dest_label_path)
    else:
        shutil.copy(src_image_path, train_dir)

for image in test_images:
    image_name, _ = os.path.splitext(image)
    src_image_path = os.path.join(dataset_dir, image)
    src_label_path = os.path.join(dataset_dir, image_name + '.txt')
    
    dest_image_path = os.path.join(test_dir, image)
    dest_label_path = os.path.join(test_dir, image_name + '.txt')
    
    if os.path.exists(src_label_path):
        shutil.copy(src_image_path, dest_image_path)
        shutil.copy(src_label_path, dest_label_path)
    else:
        shutil.copy(src_image_path, test_dir)

print(f"Split {len(train_images)} images and labels for training and {len(test_images)} images and labels for testing.")
