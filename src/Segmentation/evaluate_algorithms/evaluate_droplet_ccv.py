import os
import sys 
import cv2
import numpy as np
import csv
import time
import copy
import gc
import pandas as pd
import skimage
from matplotlib import pyplot as plt 
from pathlib import Path

sys.path.insert(0, 'src')
import Common.Util as FoldersUtil
import evaluate_algorithms_config
import Common.config as config
import Segmentation.droplet.ccv.Segmentation_CCV as seg


def manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics):
    directory_image = os.path.join(path_dataset, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
    directory_label = os.path.join(path_dataset, config.DATA_GENERAL_LABEL_FOLDER_NAME)
    directory_stats = os.path.join(path_dataset, config.DATA_GENERAL_STATS_FOLDER_NAME)

    # manage folders to store the results of the segmentation
    list_folders = []
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_STATS_FOLDER_NAME))
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME))
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME))
    FoldersUtil.manage_folders(list_folders)

    return directory_image, directory_label, directory_stats

def save_shapes_to_yolo_label(label_path, droplets_detected, width, height):
    with open(label_path, "w") as file:
        for droplet in droplets_detected:
            # write each one of the points in the label file
            normalized_points = []
            for point in droplet:
                x, y = point
                # Normalize coordinates by dividing by the image dimensions and converting to percentages
                x_normalized = x / width
                y_normalized = y / height
                normalized_points.append((x_normalized, y_normalized))

            yolo_line = f"0"
            for (x_norm, y_norm) in normalized_points:
                yolo_line += f" {x_norm:.10f} {y_norm:.10f}"
            file.write(yolo_line + "\n")      


def compute_ccv_segmentation(image_colors, image_gray, filename, results_path):
    # get the predicted droplets with cv algorithm
    predicted_seg:seg.Segmentation_CCV = seg.Segmentation_CCV(image_colors, image_gray, filename, 
                                                save_image_steps = False, 
                                                segmentation_method = 0, 
                                                dataset_results_folder = results_path)

    #sorted_droplets = sorted(predicted_seg.droplets_data, key=lambda droplet: (droplet.center_x, droplet.center_y))
    droplets_detected = []
    for droplet in predicted_seg.droplets_data:
        mask = np.zeros_like(image_gray)

        # contour shape
        if droplet.overlappedIDs == []:
            cv2.drawContours(mask, [predicted_seg.droplet_shapes.get(droplet.id)], -1, (255), cv2.FILLED)
            contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                points = contour.reshape(-1, 2)
                droplets_detected.append(points)
        # perfect circle
        else:
            cv2.circle(mask, (droplet.center_x, droplet.center_y), droplet.radius, (255), cv2.FILLED)
            contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                points = contour.reshape(-1, 2)
                droplets_detected.append(points)
   
       
    return droplets_detected


def main_ccv(fieldnames_segmentation, fieldnames_statistics, fieldnames_time, path_csv_segmentation, path_csv_statistics, path_dataset, path_results):
    directory_image, directory_label, directory_stats = manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics)
    
    segmentation_time_csv_path = os.path.join(path_results, config.RESULTS_GENERAL_SEGMENTATIONTIME_FOLDER_NAME + ".csv")
    with open(segmentation_time_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(fieldnames_time)
        

        # apply the segmentation in each one of the images and then calculate the accuracy and save it
        for i, file in enumerate(os.listdir(directory_image)): 
            start_time = time.time()
            filename = file.split(".")[0]
        
            image_path = os.path.join(directory_image, file)
            image_gray = cv2.imread(os.path.join(directory_image, file), cv2.IMREAD_GRAYSCALE)
            image_colors = cv2.imread(os.path.join(directory_image, file))  
            image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
            height, width = image_colors.shape[:2]
         
            image_area = width * height

            label_path = os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME, filename + ".txt")

            print("Segmentating image", filename + "..." )

            try:
                # apply segmentation
                predicted_droplets = compute_ccv_segmentation(image_colors, image_gray, filename, path_results)
                
                seg_time = time.time()
                segmentation_time = seg_time - start_time

                # save segmentation time to a file
                csv_writer.writerow([filename, segmentation_time])

                save_shapes_to_yolo_label(label_path, predicted_droplets, width, height)

            except np.core._exceptions._ArrayMemoryError as e:
                print(f"Memory error encountered while processing {filename}: {e}")


### SQUARES
# REAL DATASET
main_ccv(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
         evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
         evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
         evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_CV, 
         evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_CV, 
         config.DATA_REAL_WSP_TESTING_DIR, 
         config.RESULTS_REAL_CCV_DIR)

# SYNTHETIC DATASET
main_ccv(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
         evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
         evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
         evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_CV, 
         evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_CV, 
         config.DATA_SYNTHETIC_WSP_TESTING_DIR, 
         config.RESULTS_SYNTHETIC_CCV_DIR)

# ### FULL IMAGE
# # REAL DATASET
# main_ccv(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
#          evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
#          evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
#          evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_FULL_DATASET_CV, 
#          evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_FULL_DATASET_CV, 
#          config.DATA_REAL_FULL_WSP_TESTING_DIR, 
#          config.RESULTS_REAL_FULL_CCV_DIR)

# # SYNTHETIC DATASET
# main_ccv(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
#          evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
#          evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
#          evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_FULL_DATASET_CV, 
#          evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_FULL_DATASET_CV, 
#          config.DATA_SYNTHETIC_FULL_WSP_TESTING_DIR, 
#          config.RESULTS_SYNTHETIC_FULL_CCV_DIR)