import os

from color_generator import generate_color
from statistics_generator import calculate_statistics
from wsp_generator import generate_one_wsp

from paths import path_to_images_folder, path_main, path_to_statistics_folder

### VALUES
num_wsp = 5

def delete_old_dataset():
    for filename in os.listdir(path_to_images_folder):
        file_path_image = os.path.join(path_to_images_folder, filename)
        if os.path.isfile(file_path_image):
            os.remove(file_path_image)

    for filename in os.listdir(path_to_statistics_folder):
        file_path_statistic = os.path.join(path_to_statistics_folder, filename)
        if os.path.isfile(file_path_statistic):
            os.remove(file_path_statistic)

def create_folders():
    if not os.path.exists(path_main):
        os.makedirs(path_main)
    if not os.path.exists(path_to_images_folder):
        os.makedirs(path_to_images_folder)
    if not os.path.exists(path_to_statistics_folder):
        os.makedirs(path_to_statistics_folder)
    
# manage folders
create_folders()
delete_old_dataset()
droplet_color, background_color = generate_color()

# generate images
for i in range(num_wsp):
    print(i)
    width, height, image, num_spots, droplets_data = generate_one_wsp(droplet_color, background_color)
    calculate_statistics(i, width, height, image, num_spots, droplets_data, background_color)
