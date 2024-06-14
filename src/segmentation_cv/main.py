import sys
import cv2
from matplotlib import pyplot as plt 
import os
import copy
import numpy as np
import pandas as pd

from Calculated_Statistics import Calculated_Statistics
from GroundTruth_Statistics import GroundTruth_Statistics
from Accuracy import Accuracy
from Statistics import Statistics
from Distortion import Distortion

import config


# delete old outputs

# delete_folder_contents(path_to_outputs_folder)
# delete_folder_contents(path_to_real_dataset_inesc_undistorted)
# delete_folder_contents(path_to_statistics_pred_folder)
# delete_folder_contents(path_to_masks_single_pred_folder)
# delete_folder_contents(path_to_masks_overlapped_pred_folder)
# delete_folder_contents(path_to_detected_circles)


isArtificialDataset = True

TP_overlapped = 0
FP_overlapped = 0
FN_overlapped = 0
IOU = 0
dice = 0

vmd_accum_gt = 0
rsf_accum_gt = 0
perc_accum_gt = 0
no_accum_gt = 0

vmd_accum_c = 0
rsf_accum_c = 0
perc_accum_c = 0
no_accum_c = 0


k = 0
# for each one of the images of the artificial dataset
if isArtificialDataset:
    for file in os.listdir(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR):
        # get name of the file
        parts = file.split(".")
        filename = parts[0]
        
        # treat image
        in_image = cv2.imread(os.path.join(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR, filename + ".png"), cv2.IMREAD_GRAYSCALE)
        in_image_colors = cv2.imread(os.path.join(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR, filename + ".png"))  
        out_image = copy.copy(in_image)
        
        # calculate statistics
        calculated:Calculated_Statistics = Calculated_Statistics(in_image_colors, filename, False, True)
        droplets_calculated_dict = {droplet.id: droplet for droplet in calculated.droplets_data}
        stats_calculated:Statistics = calculated.stats

        vmd_accum_c += stats_calculated.vmd_value
        rsf_accum_c += stats_calculated.rsf_value
        perc_accum_c += stats_calculated.coverage_percentage
        no_accum_c += stats_calculated.no_droplets

        # save ground truth
        groundtruth:GroundTruth_Statistics = GroundTruth_Statistics(filename, out_image)
        droplets_groundtruth_dict = {droplet.id: droplet for droplet in groundtruth.droplets}
        stats_groundtruth:Statistics = groundtruth.stats

        vmd_accum_gt += stats_groundtruth.vmd_value
        rsf_accum_gt += stats_groundtruth.rsf_value
        perc_accum_gt += stats_groundtruth.coverage_percentage
        no_accum_gt += stats_groundtruth.no_droplets

        # calculate accuracy values
        acc:Accuracy = Accuracy(droplets_calculated_dict, droplets_groundtruth_dict, filename, stats_calculated, stats_groundtruth)

        TP_overlapped += acc.true_positives_overlapped
        FP_overlapped += acc.false_positives_overlapped
        FN_overlapped += acc.false_negatives_overlapped

        IOU += acc.iou 
        dice += acc.dice_coefficient
        k+=1
        print(k, filename)
        if k==10: 
            print("end")
            break
        
    IOU = IOU / 10
    dice = dice / 10
    precision_overlapped, recall_overlapped, f1_score_overlapped, = Accuracy.calculate_parameters(acc, TP_overlapped, 0, FP_overlapped, FN_overlapped)

    Accuracy.write_scores_file(precision_overlapped, recall_overlapped, f1_score_overlapped, IOU, dice)
    Accuracy.save_stats_cvs(stats_groundtruth, stats_calculated)

else: 
    for file in os.listdir(config.DATA_REAL_RAW_IMAGE_DIR):
        # get name of the file
        parts = file.split(".")
        filename = parts[0]

        # generate undistorted image
        try:    
            image = cv2.imread(os.path.join(config.DATA_REAL_RAW_IMAGE_DIR, file), cv2.IMREAD_GRAYSCALE)
            image_color = cv2.imread(os.path.join(config.DATA_REAL_RAW_IMAGE_DIR, file))
            dist = Distortion(image, image_color, filename, save_photo=True)
            if dist.noPaper: continue

            undistorted_image = cv2.imread(os.path.join(config.RESULTS_CV_UNDISTORTED_DIR, filename + '.png'), cv2.IMREAD_GRAYSCALE)
            undistorted_image_color = cv2.imread(os.path.join(config.RESULTS_CV_UNDISTORTED_DIR, filename + '.png'))
            
            calc_stats = Calculated_Statistics(undistorted_image_color, filename, save_images=True, create_masks=True)

            calc_stats.stats.save_stats_file(filename)

        except Exception as e:
            print("An error occurred:", e)
        






        






        
