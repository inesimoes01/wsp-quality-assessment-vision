import os

folder_path = "inputs\\wsp_unext\\images"  # Replace this with the path to your folder

for filename in os.listdir(folder_path):
    
    # Check if the file ends with "_shad"
    if filename.endswith('_sha'):
        # Remove the "_shad" part and add ".png"
        new_filename = filename[:-4] + '.png'
        # Rename the file
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
