import sys
import cv2
from matplotlib import pyplot as plt 
import os
import copy
import numpy as np

from Calculated_Statistics import Calculated_Statistics
from GroundTruth_Statistics import GroundTruth_Statistics
from Accuracy import Accuracy

sys.path.insert(0, 'src/common')
from Util import *
from Variables import *
from Droplet import *
from Statistics  import * 
from Distortion import *

# delete old outputs
delete_folder_contents(path_to_outputs_folder)
delete_folder_contents(path_to_real_dataset_inesc_undistorted)
delete_folder_contents(path_to_statistics_pred_folder)
delete_folder_contents(path_to_masks_single_pred_folder)
delete_folder_contents(path_to_masks_overlapped_pred_folder)


isArtificialDataset = True

TP_overlapped = 0
FP_overlapped = 0
FN_overlapped = 0
IOU = 0
dice = 0


# for each one of the images of the artificial dataset
if isArtificialDataset:
    for file in os.listdir(path_to_images_folder):
        # get name of the file
        parts = file.split(".")
        filename = parts[0]
        
        # treat image
        in_image = cv2.imread(os.path.join(path_to_images_folder, filename + ".png"), cv2.IMREAD_GRAYSCALE)
        #in_image = cv2.cvtColor(in_image)
        out_image = copy.copy(in_image)

        path_to_save_contours_single = os.path.join(path_to_outputs_folder, "single", filename)
        path_to_save_contours_overlapped = os.path.join(path_to_outputs_folder, "overlapped", filename)
        create_folders(path_to_save_contours_overlapped)
        create_folders(path_to_save_contours_single)
        
        # calculate statistics
        calculated:Calculated_Statistics = Calculated_Statistics(out_image, filename, path_to_save_contours_overlapped, path_to_save_contours_single)
        droplets_calculated_dict = {droplet.id: droplet for droplet in calculated.droplets_data}
        stats_calculated:Statistics = calculated.stats

        # save ground truth
        groundtruth:GroundTruth_Statistics = GroundTruth_Statistics(filename, out_image)
        droplets_groundtruth_dict = {droplet.id: droplet for droplet in groundtruth.droplets}
        stats_groundtruth:Statistics = groundtruth.stats

        # calculate accuracy values
        acc:Accuracy = Accuracy(droplets_calculated_dict, droplets_groundtruth_dict, filename, stats_calculated, stats_groundtruth)

        TP_overlapped += acc.true_positives_overlapped
        FP_overlapped += acc.false_positives_overlapped
        FN_overlapped += acc.false_negatives_overlapped

        IOU += acc.iou 
        dice += acc.dice_coefficient
        
    IOU = IOU / num_wsp
    dice = dice / num_wsp
    precision_overlapped, recall_overlapped, f1_score_overlapped, = Accuracy.calculate_parameters(acc, TP_overlapped, 0, FP_overlapped, FN_overlapped)

    Accuracy.write_scores_file(precision_overlapped, recall_overlapped, f1_score_overlapped, IOU, dice)

else: 
    for file in os.listdir(path_to_real_dataset_inesc_original):
        # get name of the file
        parts = file.split(".")
        filename = parts[0]
        path_to_save_contours_overlapped = os.path.join(path_to_outputs_folder, "single", filename)
        path_to_save_contours_single = os.path.join(path_to_outputs_folder, "overlapped", filename)
        create_folders(path_to_save_contours_overlapped)
        create_folders(path_to_save_contours_single)

        # generate undistorted image
        try:    
            image = cv2.imread(path_to_real_dataset_inesc_original + '\\' + file, cv2.IMREAD_GRAYSCALE)
            dist = Distortion(image, filename)
            if dist.noPaper: continue

            undistorted_image = cv2.imread(path_to_real_dataset_inesc_undistorted + '\\' + filename + '.png', cv2.IMREAD_GRAYSCALE)
            
            calc_stats = Calculated_Statistics(undistorted_image, filename, path_to_save_contours_overlapped, path_to_save_contours_single)

            calc_stats.stats.save_stats_file(filename)

        except Exception as e:
            print("An error occurred:", e)
        




        






        
