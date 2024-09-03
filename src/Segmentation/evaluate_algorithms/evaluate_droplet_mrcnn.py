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

sys.path.insert(0, 'src\\Segmentation\\droplet\\cnn\\MaskRCNN\\')
from mrcnn import model as modellib, utils
import custom_mrcnn_classes as custom_mrcnn_classes


def compute_mrcnn_segmentation(image, mrcnn_model_path):
    inference_config = custom_mrcnn_classes.InferenceConfig()

    # recreate the model in inference mode
    mrcnn_model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config, model_dir=mrcnn_model_path)
    mrcnn_model.load_weights(mrcnn_model_path, by_name=True)

    img_arr = np.array(image)
    results = mrcnn_model.detect([img_arr])
    r = results[0]

    droplets_detected = []

    for i in range(r['masks'].shape[-1]):
        mask = r['masks'][:, :, i]
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            points = contour.reshape(-1, 2)
            droplets_detected.append(points)

    return droplets_detected
            

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
               
def manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics):
    directory_image = os.path.join(path_dataset, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
    directory_label = os.path.join(path_dataset, config.DATA_GENERAL_LABEL_FOLDER_NAME)
    directory_stats = os.path.join(path_dataset, config.DATA_GENERAL_STATS_FOLDER_NAME)

    # manage folders to store the results of the segmentation
    list_folders = []
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_STATS_FOLDER_NAME))
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_ACC_FOLDER_NAME))
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME))
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_INFO_FOLDER_NAME))
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME))
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_UNDISTORTED_FOLDER_NAME))
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_MASK_SIN_FOLDER_NAME))
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_MASK_OV_FOLDER_NAME))
    FoldersUtil.manage_folders(list_folders)


    return directory_image, directory_label, directory_stats



def main_mrcnn(fieldnames_segmentation, fieldnames_statistics, fieldnames_time, path_csv_segmentation, path_csv_statistics, path_dataset, path_results, model_path):
    directory_image, directory_label, directory_stats = manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics)
    
    segmentation_time_csv_path = os.path.join(path_results, config.RESULTS_GENERAL_SEGMENTATIONTIME_FOLDER_NAME + ".csv")
    with open(segmentation_time_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow()
        

        # apply the segmentation in each one of the images and then calculate the accuracy and save it
        for i, file in enumerate(os.listdir(directory_image)): 
            start_time = time.time()
            filename = file.split(".")[0]
        
            image_path = os.path.join(directory_image, file)
            image_colors = skimage.io.imread(image_path)
            height, width = image_colors.shape[:2]
            image_area = width * height

            label_path = os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME, filename + ".txt")

            print("Evaluating image", filename + "..." )

            try:
                # apply segmentation
                predicted_droplets = compute_mrcnn_segmentation(image_colors, model_path, width, height)
                
                seg_time = time.time()
                segmentation_time = seg_time - start_time

                # save segmentation time to a file
                csv_writer.writerow([file, segmentation_time])

                save_shapes_to_yolo_label(label_path, predicted_droplets, width, height)

            except np.core._exceptions._ArrayMemoryError as e:
                print(f"Memory error encountered while processing {filename}: {e}")


# SYNTHETIC DATASET
main_mrcnn(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
           fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
           path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_MRCNN, 
           path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_MRCNN, 
           path_dataset=config.DATA_SYNTHETIC_NORMAL_WSP_TESTING_DIR, 
           path_results=config.RESULTS_SYNTHETIC_MRCNN_DIR, 
           model_path=evaluate_algorithms_config.DROPLET_MRCNN_MODEL)

# REAL DATASET
main_mrcnn(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
           fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
           path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_YOLO, 
           path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_YOLO, 
           path_dataset=config.DATA_REAL_WSP_TESTING_DIR, 
           path_results=config.RESULTS_REAL_MRCNN_DIR, 
           model_path=evaluate_algorithms_config.DROPLET_MRCNN_MODEL)
