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
from ultralytics import YOLO

sys.path.insert(0, 'src')
import Common.Util as FoldersUtil
import evaluate_algorithms_config
import Common.config as config


def compute_yolo_segmentation(image_path, yolo_model):
    predicted_droplets_adjusted = []
    # predict image results
    
    results = yolo_model(image_path, conf=0.1)

    if results[0].masks:
        segmentation_result = results[0].masks.xy

       
        detected_pts = []

        for polygon in segmentation_result:
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            detected_pts.append(pts)

        for coords in detected_pts:
            adjusted_coords = []
            for point in coords:
                x, y = point[0]
                adjusted_coords.append([x, y])
            if adjusted_coords != []:
                predicted_droplets_adjusted.append(np.array(adjusted_coords, dtype=np.int32))

    return predicted_droplets_adjusted

def save_shapes_to_yolo_label(label_path, droplets_detected, width, height):
    with open(label_path, "w") as file:
        for droplet in droplets_detected:
            
            # write each one of the points in the label file
            normalized_points = []
            for point in droplet:
                x, y = point
               
                x_normalized = x / width
                y_normalized = y / height
                normalized_points.append((x_normalized, y_normalized))

            yolo_line = f"0"
            for (x_norm, y_norm) in normalized_points:
                yolo_line += f" {x_norm:.10f} {y_norm:.10f}"
            file.write(yolo_line + "\n")

               
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

    with open(path_csv_segmentation, mode='w', newline='') as file:
        csv.DictWriter(file, fieldnames=fieldnames_segmentation).writeheader()

    with open(path_csv_statistics, mode='w', newline='') as file:
        csv.DictWriter(file, fieldnames=fieldnames_statistics).writeheader()

    return directory_image, directory_label, directory_stats

def main_yolo(fieldnames_segmentation, fieldnames_statistics, fieldnames_time, path_csv_segmentation, path_csv_statistics, path_dataset, path_results, yolo_model_path):
    directory_image, directory_label, directory_stats = manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics)
 
    segmentation_time_csv_path = os.path.join(path_results, config.RESULTS_GENERAL_SEGMENTATIONTIME_FOLDER_NAME + ".csv")
    with open(segmentation_time_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(fieldnames_time)
        
        # apply the segmentation in each one of the images and then calculate the accuracy and save it
        for i, file in enumerate(os.listdir(directory_image)): 
        
            start_time = time.time()
            yolo_model = YOLO(yolo_model_path)
            filename = file.split(".")[0]
        
            image_path = os.path.join(directory_image, file)
            image_colors = cv2.imread(os.path.join(directory_image, file))  
            image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
            height, width = image_colors.shape[:2]
           
            label_path = os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME, filename + ".txt")

            print("Evaluating image", filename + "..." )

            try:
                predicted_droplets = compute_yolo_segmentation(image_path, yolo_model)
                
                seg_time = time.time()
                segmentation_time = seg_time - start_time
                
                # save segmentation time to a file
                csv_writer.writerow([filename, segmentation_time])

                save_shapes_to_yolo_label(label_path, predicted_droplets, width, height)

            except np.core._exceptions._ArrayMemoryError as e:
                print(f"Memory error encountered while processing {filename}: {e}")



yolo_model_path = evaluate_algorithms_config.DROPLET_YOLO_MODEL

# SYNTHETIC DATASET
main_yolo(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
          evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
          evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
          evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO, 
          evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO, 
          config.DATA_SYNTHETIC_NORMAL_WSP_TESTING_DIR,
          config.RESULTS_SYNTHETIC_YOLO_DIR, 
          yolo_model_path)

# REAL DATASET
main_yolo(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
          evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
          evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
          evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_YOLO, 
          evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_YOLO, 
          config.DATA_REAL_WSP_TESTING_DIR, 
          config.RESULTS_REAL_YOLO_DIR, 
          yolo_model_path)
