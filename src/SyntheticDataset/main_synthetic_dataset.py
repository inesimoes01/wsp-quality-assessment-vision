import os
import time
import numpy as np

from create_colors import Colors
from dataset_results import DatasetResults
from create_wsp import CreateWSP
import sys
from droplet_shape import DropletShape
import shape_list as shape_list
import csv

sys.path.insert(0, 'src/common')
import config as config
import Util

def generate_normal_distribution_num_droplets():
    num_droplets_per_image = np.random.normal(config.MEAN_DROPLETS, config.STD_DROPLETS, config.NUM_WSP)
    num_droplets_per_image = np.round(num_droplets_per_image).astype(int)  # round to integer values

    # non-negative values
    num_droplets_per_image = np.maximum(num_droplets_per_image, 1)

    with open('results\\latex\\normal_distribution_no_droplets.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['No Droplets'])
        for no in num_droplets_per_image:
            no = int(no)
            csvwriter.writerow([no])

    return num_droplets_per_image

# hard coded for 1000 wsp
def choose_resolution(index):
    if index < 80 or 250 <= index < 330:
        reso = 18
        charac_particle_size = 15
    elif 80 <= index < 160 or 330 <= index < 410:
        reso = 25
        charac_particle_size = 20
    elif 160 <= index < 250 or 410 <= index < 500:
        reso = 32
        charac_particle_size = 25
        
    return reso, charac_particle_size

deleteDataset = input("Do you wanna delete the old dataset? (y)es or (n)o: ")

if deleteDataset == "y":
    deleteDataset = input("Sure? (y)es or (n)o: ")
    
    # manage folders
    Util.create_folders(config.DATA_ARTIFICIAL_WSP_DIR)
    Util.manage_folders([os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_INFO_FOLDER_NAME),
                        os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_IMAGE_FOLDER_NAME),
                        os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_LABEL_FOLDER_NAME),
                        os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_STATS_FOLDER_NAME),
                        os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_MASK_OV_FOLDER_NAME),
                        os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_MASK_SIN_FOLDER_NAME)])

    index = 0

if deleteDataset == "n":
    index = input("What is the last index? ")
    index = int(index) + 1

# IMPORT SHAPES AND COLORS
colors = Colors(config.outside_color1, config.light_color1, config.dark_color1, config.background_color_1,config.background_color_2)
shapes, polygons_by_size = shape_list.save_all_shapes()

num_droplets_list = generate_normal_distribution_num_droplets()

total_time = 0
change_threshold = int(config.NUM_WSP * 0.3)

# generate images
for i in range(config.NUM_WSP):

    start_time = time.time()
    filename = i + index

    # change colors for the other half of the dataset
    if filename == config.NUM_WSP / 2:
        colors = Colors(config.outside_color2, config.light_color2, config.dark_color2, config.background_color_1, config.background_color_2)

    # change resolution of the image to have 3 different resolutions at the end
    image_resolution, characteristic_particle_size = choose_resolution(filename)

    print("Creating image number ", i + index)

    no_droplets = num_droplets_list[i]
    wsp_image = CreateWSP(filename, colors, shapes, polygons_by_size, no_droplets, 0, image_resolution, characteristic_particle_size)

    print("Image created. Calculating statistics...")
    DatasetResults(wsp_image, colors)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken:", elapsed_time, "seconds")

    total_time += elapsed_time

print("Overall time taken:", total_time, "seconds")
print("Average time taken:", total_time / config.NUM_WSP)

with open("results\\latex\\time_statistics", 'w') as f:

    f.write(f"Overall time taken: {total_time} seconds \n")
    f.write(f"Average time taken: {total_time / config.NUM_WSP} seconds \n")



    