import os
import time
import numpy as np
import pandas as pd
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

csv_normal_distribution_file = "results\\latex\\normal_distribution_no_droplets.csv"

def generate_normal_distribution_num_droplets():
    num_droplets_per_image = np.random.normal(config.MEAN_DROPLETS, config.STD_DROPLETS, config.NUM_WSP)
    num_droplets_per_image = np.round(num_droplets_per_image).astype(int)  # round to integer values

    # non-negative values
    num_droplets_per_image = np.maximum(num_droplets_per_image, 1)

    with open(csv_normal_distribution_file,'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['No Droplets'])
        for no in num_droplets_per_image:
            no = int(no)
            csvwriter.writerow([no])

    return num_droplets_per_image

# hard coded for 500 wsp
def choose_resolution(index):
    uniform = 0
    if index < 10 or 150 <= index < 160:
        reso = 19.7
        charac_particle_size = 6
        uniform = 2
    elif 10 <= index < 20 or 160 <= index < 170:
        reso = 19.7 
        charac_particle_size = 6
        uniform = 5
    elif 20 <= index < 30 or 170 <= index < 180:
        reso = 19.7
        charac_particle_size = 8
        uniform = 2
    elif 30 <= index < 40 or 180 <= index < 190:
        reso = 19.7
        charac_particle_size = 15
        uniform = 7
    elif 40 <= index < 50 or 190 <= index < 200:
        reso = 19.7
        charac_particle_size = 14
        uniform = 2

    elif 50 <= index < 60 or 200 <= index < 210:
        reso = 24.62
        charac_particle_size = 8
        uniform = 2
    elif 60 <= index < 70 or 210 <= index < 220:
        reso = 24.62
        charac_particle_size = 9
        uniform = 2
    elif 70 <= index < 80 or 220 <= index < 230:
        reso = 24.62
        charac_particle_size = 10
        uniform = 7
    elif 80 <= index < 90 or 230 <= index < 240:
        reso = 24.62
        charac_particle_size = 11
        uniform = 2
    elif 90 <= index < 100 or 240 <= index < 250:
        reso = 24.62
        charac_particle_size = 12
        uniform = 5


    elif 100 <= index < 110 or 250 <= index < 260:
        reso = 27.696
        charac_particle_size = 10
        uniform = 5
    elif 110 <= index < 120 or 260 <= index < 270:
        reso = 27.696
        charac_particle_size = 14
        uniform = 2
    elif 120 <= index < 130 or 270 <= index < 280:
        reso = 27.696
        charac_particle_size = 16
        uniform = 5
    elif 130 <= index < 140 or 280 <= index < 290:
        reso = 27.696
        charac_particle_size = 17  
        uniform = 7
    elif 140 <= index < 150 or 290 <= index < 300:
        reso = 27.696
        charac_particle_size = 18
        uniform = 2

    return reso, charac_particle_size, uniform

deleteDataset = input("Do you wanna delete the old dataset? (y)es or (n)o: ")

if deleteDataset == "y":
    deleteDataset = input("Sure? (y)es or (n)o: ")
    
    # manage folders
    Util.create_folders(config.DATA_SYNTHETIC_NORMAL_WSP_DIR)
    Util.manage_folders([os.path.join(config.DATA_SYNTHETIC_NORMAL_WSP_DIR, config.DATA_GENERAL_INFO_FOLDER_NAME),
                        os.path.join(config.DATA_SYNTHETIC_NORMAL_WSP_DIR, config.DATA_GENERAL_IMAGE_FOLDER_NAME),
                        os.path.join(config.DATA_SYNTHETIC_NORMAL_WSP_DIR, config.DATA_GENERAL_LABEL_FOLDER_NAME),
                        os.path.join(config.DATA_SYNTHETIC_NORMAL_WSP_DIR, config.DATA_GENERAL_STATS_FOLDER_NAME),
                        os.path.join(config.DATA_SYNTHETIC_NORMAL_WSP_DIR, config.DATA_GENERAL_MASK_OV_FOLDER_NAME),
                        os.path.join(config.DATA_SYNTHETIC_NORMAL_WSP_DIR, config.DATA_GENERAL_MASK_SIN_FOLDER_NAME)])
    num_droplets_list = generate_normal_distribution_num_droplets()

    index = 0

if deleteDataset == "n":
    index = input("What is the last index? ")
    index = int(index) + 1

    df = pd.read_csv(csv_normal_distribution_file)
    num_droplets_list = df['No Droplets'].tolist()

# IMPORT SHAPES AND COLORS
colors = Colors(config.outside_color1, config.light_color1, config.dark_color1, config.background_colors)
shapes, polygons_by_size = shape_list.save_all_shapes()

def get_row_values(file_path, row_number):
    # Read the CSV file into a DataFrame

    
    return row_values
#num_droplets_list = generate_normal_distribution_num_droplets()

total_time = 0
change_threshold = int(config.NUM_WSP * 0.3)

# generate images
for i in range(config.NUM_WSP):

    start_time = time.time()
    filename = i + index

    # change colors for the other half of the dataset
    if filename > 150:
    #if filename == config.NUM_WSP / 2:
        colors = Colors(config.outside_color2, config.light_color2, config.dark_color2, config.background_colors)

    # change resolution of the image to have 3 different resolutions at the end
    image_resolution, characteristic_particle_size, uniform = choose_resolution(filename)

    print("Creating image number ", i + index)

    no_droplets = num_droplets_list[i]
    wsp_image = CreateWSP(filename, colors, shapes, polygons_by_size, no_droplets, 1, image_resolution, characteristic_particle_size, uniform)

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



    