import os

# Define the class mappings as integers
class_mapping = {
    0: 1,
    1: 3,
    2: 0,
    3: 2,
}

directory_path = "/home/rics/Desktop/g/"  # Replace with your directory containing YOLO annotation files

for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)

        # Read the contents of the annotation file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Replace class labels in the lines
        new_lines = []
        for line in lines:
            line_parts = line.strip().split()
            if len(line_parts) >= 5:
                class_label = int(float(line_parts[0]))  # Convert to float and then to int
                if class_label in class_mapping:
                    line_parts[0] = str(class_mapping[class_label])  # Convert back to string
                new_lines.append(" ".join(line_parts))

        # Write the modified lines back to the file
        with open(file_path, 'w') as file:
            file.write("\n".join(new_lines))

        print(f"Converted classes in {filename}.")

print("Class conversions completed.")

