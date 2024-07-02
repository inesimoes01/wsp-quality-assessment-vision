import os
import time
import numpy as np

from CreateColors import Colors
from DatasetResults import DatasetResults
from CreateWSP import CreateWSP
import sys
from DropletShape import DropletShape
import ShapeList


sys.path.insert(0, 'src/common')
import config as config
from Util import *

def generate_normal_distribution_num_droplets():
    num_droplets_per_image = np.random.normal(config.MEAN_DROPLETS, config.STD_DROPLETS, config.NUM_WSP)
    num_droplets_per_image = np.round(num_droplets_per_image).astype(int)  # round to integer values

    # non-negative values
    num_droplets_per_image = np.maximum(num_droplets_per_image, 1)
    return num_droplets_per_image

deleteDataset = input("Do you wanna delete the old dataset? (y)es or (n)o: ")
if deleteDataset == "y":
    deleteDataset = input("Sure? (y)es or (n)o: ")
    
    # manage folders
    create_folders(config.DATA_ARTIFICIAL_RAW_DIR)
    create_folders(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR)
    create_folders(config.DATA_ARTIFICIAL_RAW_STATISTICS_DIR)
    create_folders(config.DATA_ARTIFICIAL_RAW_INFO_DIR)
    create_folders(config.DATA_ARTIFICIAL_RAW_MASK_SIN_DIR)
    create_folders(config.DATA_ARTIFICIAL_RAW_MASK_OV_DIR)
    create_folders(config.DATA_ARTIFICIAL_RAW_LABEL_DIR)

    # manage folders
    delete_folder_contents(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR)
    delete_folder_contents(config.DATA_ARTIFICIAL_RAW_STATISTICS_DIR)
    delete_folder_contents(config.DATA_ARTIFICIAL_RAW_INFO_DIR)
    delete_folder_contents(config.DATA_ARTIFICIAL_RAW_MASK_SIN_DIR)
    delete_folder_contents(config.DATA_ARTIFICIAL_RAW_MASK_OV_DIR)
    delete_folder_contents(config.DATA_ARTIFICIAL_RAW_LABEL_DIR)

    index = 0

if deleteDataset == "n":
    index = input("What is the last index? ")
    index = int(index) + 1

# IMPORT SHAPES AND COLORS
colors = Colors()
shapes, polygons_by_size = ShapeList.save_all_shapes()

num_droplets_list = generate_normal_distribution_num_droplets()

# generate images
for i in range(config.NUM_WSP):
    start_time = time.time()
    filename = i + index

    print("Creating image number ", i + index)

    no_droplets = num_droplets_list[i]
    wsp_image = CreateWSP(filename, colors, shapes, polygons_by_size, no_droplets, 1)

    print("Image created. Calculating statistics...")
    DatasetResults(wsp_image, colors)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken:", elapsed_time, "seconds")

    