import os
import shutil

# Define the source and destination directories
source_folder = "data\\synthetic_normal_dataset\\wsp\\image"
destination_folder = "data\\synthetic_normal_dataset\\testing"

# Ensure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Loop through each image in the source folder
for i in range(0, 500, 10):
    # Construct the full file path
    file_name = f'{i}.png'  # Assuming the images have .jpg extension. Change it if different.
    source_file = os.path.join(source_folder, file_name)
    
    if os.path.exists(source_file):  # Check if the file exists
        shutil.copy(source_file, destination_folder)
        print(f'Copied: {file_name}')
    else:
        print(f'File does not exist: {file_name}')

print('Copying process completed.')
