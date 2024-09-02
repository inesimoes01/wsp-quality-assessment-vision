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

    for i in range(r['masks'].shape[-1]):
        mask = r['masks'][:, :, i]
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            points = contour.reshape(-1, 2)
            print(f"Object {i + 1} points:")
            print(points)

            cv2.polylines(image, [contour], isClosed=True, color=(0, 255, 0), thickness=2)



    return

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

    with open(path_csv_segmentation, mode='w', newline='') as file:
        csv.DictWriter(file, fieldnames=fieldnames_segmentation).writeheader()

    with open(path_csv_statistics, mode='w', newline='') as file:
        csv.DictWriter(file, fieldnames=fieldnames_statistics).writeheader()

    return directory_image, directory_label, directory_stats


def main_mrcnn(fieldnames_segmentation, fieldnames_statistics, path_csv_segmentation, path_csv_statistics, path_dataset, path_results, model_path, iou_threshold, distance_threshold, width_mm):
    directory_image, directory_label, directory_stats = manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics)

    # apply the segmentation in each one of the images and then calculate the accuracy and save it
    for i, file in enumerate(os.listdir(directory_image)): 
        start_time = time.time()
        filename = file.split(".")[0]
       
        image_path = os.path.join(directory_image, file)
        image_colors = skimage.io.imread(image_path)
        width, height = image_colors.shape[:2]
        image_area = width * height
        image_correct_predictions = copy.copy(image_colors)

        print("Evaluating image", filename + "..." )

        try:
            predicted_droplets, predicted_droplets_centroid, predicted_stats = compute_mrcnn_segmentation(image_colors, model_path)
            
            seg_time = time.time()
            segmentation_time = seg_time - start_time

        except np.core._exceptions._ArrayMemoryError as e:
            print(f"Memory error encountered while processing {filename}: {e}")
    
    return image_correct_predictions


main_mrcnn(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_MRCNN, evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_MRCNN, config.DATA_SYNTHETIC_NORMAL_WSP_TESTING_DIR, config.RESULTS_SYNTHETIC_CCV_DIR, evaluate_algorithms_config.DROPLET_MRCNN_MODEL, 0.5, 10, 76)

# REAL DATASET
main_mrcnn(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_YOLO, evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_YOLO, config.DATA_REAL_WSP_TESTING_DIR, config.RESULTS_REAL_CCV_DIR, evaluate_algorithms_config.DROPLET_MRCNN_MODEL, 0.5, 10, 76)
