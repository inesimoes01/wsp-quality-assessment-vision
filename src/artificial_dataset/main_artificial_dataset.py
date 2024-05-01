import os
import time
import numpy as np

from Colors import Colors
from WSP_Statistics import WSP_Statistics
from WSP_Image import WSP_Image
import sys


sys.path.insert(0, 'src/common')
from Variables import *
from Util import *

def generate_normal_distribution_num_wsp():
    num_droplets_per_image = np.random.normal(mean_droplets, std_droplets, num_wsp)
    num_droplets_per_image = np.round(num_droplets_per_image).astype(int)  # round to integer values

    # non-negative values
    num_droplets_per_image = np.maximum(num_droplets_per_image, 1)
    return num_droplets_per_image


# Prompt the user for their name
deleteDataset = input("Do you wanna delete the old dataset? (y)es or (n)o: ")
if deleteDataset == "y":
    deleteDataset = input("Sure? (y)es or (n)o: ")
    
    # manage folders
    create_folders(path_main_dataset)
    create_folders(path_to_images_folder)
    create_folders(path_to_statistics_gt_folder)
    create_folders(path_to_statistics_pred_folder)
    create_folders(path_to_outputs_folder)
    create_folders(path_to_masks_overlapped_pred_folder)
    create_folders(path_to_masks_single_pred_folder)
    create_folders(path_to_masks_overlapped_gt_folder)
    create_folders(path_to_masks_single_gt_folder)
    


    delete_folder_contents(path_to_images_folder)
    delete_folder_contents(path_to_statistics_gt_folder)
    delete_folder_contents(path_to_statistics_pred_folder)
    delete_folder_contents(path_to_outputs_folder)
    delete_folder_contents(path_to_masks_overlapped_pred_folder)
    delete_folder_contents(path_to_masks_single_pred_folder)
    delete_folder_contents(path_to_masks_overlapped_gt_folder)
    delete_folder_contents(path_to_masks_single_gt_folder)
    delete_folder_contents(path_to_labels_yolo)
    delete_folder_contents(path_to_dropletinfo_gt_folder)
    

    index = 0

if deleteDataset == "n":
    index = input("What is the last index? ")
    index = int(index) + 1

colors = Colors()

num_droplets_list = generate_normal_distribution_num_wsp()

# generate images
for i in range(num_wsp):
    start_time = time.time()
    filename = i + index

    print("Creating image number ", i + index)
    wsp_image = WSP_Image(filename, colors, num_droplets_list)
    print("Image created. Calculating statistics...")
    WSP_Statistics(wsp_image, colors)

    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken:", elapsed_time, "seconds")

    