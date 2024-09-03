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
from joblib import Parallel, delayed
from matplotlib import pyplot as plt 


import Segmentation.droplet.ccv.Segmentation_CCV as seg
import Common.config as config 
import Common.Util as FoldersUtil
import Segmentation.evaluate_algorithms.util_evaluate_droplet as util



from Common.Statistics import Statistics as stats

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


def compute_ccv_segmentation(image_colors, image_gray, filename, results_path):
    # get the predicted droplets with cv algorithm
    predicted_seg:seg.Segmentation_CCV = seg.Segmentation_CCV(image_colors, image_gray, filename, 
                                                save_image_steps = False, 
                                                create_masks = False, 
                                                segmentation_method = 0, 
                                                dataset_results_folder = results_path)
    
    # calculate stats
    droplet_area = [d.area for d in predicted_seg.droplets_data]
    diameter_list = sorted(stats.area_to_diameter_micro(droplet_area, predicted_seg.width, config.WIDTH_MM))

    # calculate statistics
    image_area = predicted_seg.width * predicted_seg.height
    vmd_value, coverage_percentage, rsf_value, _ = stats.calculate_statistics(diameter_list, image_area, predicted_seg.contour_area)
    no_droplets_overlapped = 0

    for drop in predicted_seg.droplets_data:
        if len(drop.overlappedIDs) > 0:
            no_droplets_overlapped += 1

    overlaped_percentage = no_droplets_overlapped /  predicted_seg.final_no_droplets * 100
    predicted_stats = stats(vmd_value, rsf_value, coverage_percentage, predicted_seg.final_no_droplets, no_droplets_overlapped, overlaped_percentage, predicted_seg.droplets_data)
    
    # save statistics file
    data = {
        '': ['VMD', 'RSF', 'Coverage %', 'Nº Droplets', 'Overlapped Droplets %', 'Number of overlapped droplets'],
        'Predicted': [predicted_stats.vmd_value, predicted_stats.rsf_value, predicted_stats.coverage_percentage, predicted_stats.no_droplets, predicted_stats.overlaped_percentage, predicted_stats.no_droplets_overlapped], 
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(results_path, config.RESULTS_GENERAL_STATS_FOLDER_NAME, filename + '.csv'), index=False, float_format='%.2f')

    sorted_droplets = sorted(predicted_seg.droplets_data, key=lambda droplet: (droplet.center_x, droplet.center_y))

    return sorted_droplets, predicted_seg.droplet_shapes, predicted_stats


def main_ccv(fieldnames_segmentation, fieldnames_statistics, path_csv_segmentation, path_csv_statistics, path_dataset, path_results, iou_threshold, distance_threshold):
    directory_image, directory_label, directory_stats = manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics)

    # apply the segmentation in each one of the images and then calculate the accuracy and save it
    for i, file in enumerate(os.listdir(directory_image)): 
        start_time = time.time()

        filename = file.split(".")[0]

        image_gray = cv2.imread(os.path.join(directory_image, file), cv2.IMREAD_GRAYSCALE)
        image_colors = cv2.imread(os.path.join(directory_image, file))  
        image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
        width, height = image_colors.shape[:2]
        image_area = width * height

        image_correct_predictions = copy.copy(image_colors)

        print("Evaluating image", filename + "..." )

        try:
            predicted_droplets, droplet_shapes, predicted_stats, image_results = compute_ccv_segmentation(image_colors, image_gray, filename, path_results)
            
            seg_time = time.time()
            segmentation_time = seg_time - start_time

            # save image results
            cv2.colorChange(image_results, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(path_dataset, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME, filename + ".png"), image_results)     

            # get groundtruth
            groundtruth_polygons = util.create_yolo_mask(os.path.join(directory_label, filename + ".txt"), width, height)
            gt_stats = util.read_stats_file(filename, os.path.join(path_dataset, config.DATA_GENERAL_STATS_FOLDER_NAME))
            
            # match groundtruth to predicted and calculate metrics
            results = util.match_predicted_to_groundtruth_ccv(predicted_droplets, droplet_shapes, groundtruth_polygons, distance_threshold, image_colors)
            image_correct_predictions, precision, recall, f1_score, map5, map595, tp, fp, fn = util.evaluate_matches_ccv(groundtruth_polygons, results, image_correct_predictions, iou_threshold, droplet_shapes)
        
            
            util.write_final_csv_metrics((filename, precision, recall, f1_score, map5, map595, tp, fp, fn, segmentation_time), path_csv_segmentation, fieldnames_segmentation)
            util.write_stats_csv(filename, predicted_stats, gt_stats, path_csv_statistics, fieldnames_statistics)

            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print("Time taken:", elapsed_time, "seconds")
 
            cv2.imwrite(os.path.join(path_results, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME, filename + "_correct_predictions.png"), image_correct_predictions)

        except np.core._exceptions._ArrayMemoryError as e:
            print(f"Memory error encountered while processing {filename}: {e}")
    
    return image_correct_predictions

def main_yolo(fieldnames_segmentation, fieldnames_statistics, path_csv_segmentation, path_csv_statistics, path_dataset, path_results, yolo_model, iou_threshold, distance_threshold, width_mm):
    directory_image, directory_label, directory_stats = manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics)

    # apply the segmentation in each one of the images and then calculate the accuracy and save it
    for i, file in enumerate(os.listdir(directory_image)): 
        start_time = time.time()

        filename = file.split(".")[0]
       
        image_path = os.path.join(directory_image, file)
        
        image_gray = cv2.imread(os.path.join(directory_image, file), cv2.IMREAD_GRAYSCALE)
        image_colors = cv2.imread(os.path.join(directory_image, file))  
        image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
        width, height = image_colors.shape[:2]
        image_area = width * height

        image_correct_predictions = copy.copy(image_colors)

        print("Evaluating image", filename + "..." )

        try:
            predicted_droplets, predicted_droplets_centroid, predicted_stats = compute_yolo_segmentation(image_path, width, height, width_mm, yolo_model)
            
            seg_time = time.time()
            segmentation_time = seg_time - start_time
            
            # save image results
            # cv2.colorChange(image_results, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(os.path.join(path_dataset, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME, filename + ".png"), image_results)          

            # get groundtruth from saved files
            groundtruth_polygons = util.create_yolo_mask(os.path.join(directory_label, filename + ".txt"), width, height)
            gt_stats = util.read_stats_file(filename, os.path.join(path_dataset, config.DATA_GENERAL_STATS_FOLDER_NAME))

            # calculate evaluation metrics
            results = util.match_predicted_to_groundtruth_yolo(predicted_droplets_centroid, groundtruth_polygons, distance_threshold, image_colors)
            image_correct_predictions, precision, recall, f1_score, map5, map595, tp, fp, fn = util.evaluate_matches_yolo(groundtruth_polygons, results, image_correct_predictions, iou_threshold)

            # write results of predictions in the correct files
            util.write_final_csv_metrics((filename, precision, recall, f1_score, map5, map595, tp, fp, fn, segmentation_time), path_csv_segmentation, fieldnames_segmentation)
            util.write_stats_csv(filename, predicted_stats, gt_stats, path_csv_statistics, fieldnames_statistics)

            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print("Time taken:", elapsed_time, "seconds")
 
            cv2.imwrite(os.path.join(path_results, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME, filename + "_correct_predictions.png"), image_correct_predictions)

        except np.core._exceptions._ArrayMemoryError as e:
            print(f"Memory error encountered while processing {filename}: {e}")
    
    return image_correct_predictions

