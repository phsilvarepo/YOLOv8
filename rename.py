import os

def rename_files(directory_path, prefix):
    try:
        # Get a list of all files in the directory
        files = os.listdir(directory_path)

        # Filter out only files starting with 'rgb_'
        rgb_files = [file for file in files if file.lower().startswith('rgb_')]

        # Rename each 'rgb_' file with 'a_' prefix
        for rgb_file in rgb_files:
            original_path = os.path.join(directory_path, rgb_file)
            new_name = f"{prefix}_{rgb_file[4:]}"
            new_path = os.path.join(directory_path, new_name)

            # Rename the file
            os.rename(original_path, new_path)

            print(f"Renamed: {rgb_file} to {new_name}")

        print("All files renamed successfully!")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
directory_path =  "/home/rics/omni.replicator_out/stand_environment"
prefix = "stand_"
rename_files(directory_path, prefix)