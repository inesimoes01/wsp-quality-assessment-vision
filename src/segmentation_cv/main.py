import cv2
from matplotlib import pyplot as plt 
import os
import copy
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, 'src/common')
import config
import Util
from Segmentation import Segmentation

from GroundTruth_Statistics import GroundTruth_Statistics
from Accuracy import Accuracy
from Statistics import Statistics
from Distortion import Distortion

# make sure all folders exist
Util.create_folders(config.RESULTS_CV_ACCURACY_DIR)
Util.create_folders(config.RESULTS_CV_DROPLETCLASSIFICATION_DIR)
Util.create_folders(config.RESULTS_CV_INFO_DIR)
Util.create_folders(config.RESULTS_CV_MASK_OV_DIR)
Util.create_folders(config.RESULTS_CV_MASK_SIN_DIR)
Util.create_folders(config.RESULTS_CV_STATISTICS_DIR)
Util.create_folders(config.RESULTS_CV_UNDISTORTED_DIR)

# delete folder contents
Util.delete_folder_contents(config.RESULTS_CV_ACCURACY_DIR)
Util.delete_folder_contents(config.RESULTS_CV_DROPLETCLASSIFICATION_DIR)
Util.delete_folder_contents(config.RESULTS_CV_INFO_DIR)
Util.delete_folder_contents(config.RESULTS_CV_MASK_OV_DIR)
Util.delete_folder_contents(config.RESULTS_CV_MASK_SIN_DIR)
Util.delete_folder_contents(config.RESULTS_CV_STATISTICS_DIR)
Util.delete_folder_contents(config.RESULTS_CV_UNDISTORTED_DIR)

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
percov_accum_gt = 0

vmd_accum_c = 0
rsf_accum_c = 0
perc_accum_c = 0
no_accum_c = 0
percov_accum_c = 0


k = 0

# for each one of the images of the artificial dataset
if isArtificialDataset:
    for file in os.listdir(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR):
        # get name of the file
        parts = file.split(".")
        filename = parts[0]
        
        # read image
        image_gray = cv2.imread(os.path.join(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR, filename + ".png"), cv2.IMREAD_GRAYSCALE)
        image_colors = cv2.imread(os.path.join(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR, filename + ".png"))  
        
        # calculate statistics
        calculated:Segmentation = Segmentation(image_colors, image_gray, filename, False, True)
        droplets_calculated_dict = {droplet.id: droplet for droplet in calculated.droplets_data}
        stats_calculated:Statistics = calculated.stats

        vmd_accum_c += stats_calculated.vmd_value
        rsf_accum_c += stats_calculated.rsf_value
        perc_accum_c += stats_calculated.coverage_percentage
        no_accum_c += stats_calculated.no_droplets
        percov_accum_c += stats_calculated.overlaped_percentage

        # save ground truth
        groundtruth:GroundTruth_Statistics = GroundTruth_Statistics(filename)
        droplets_groundtruth_dict = {droplet.id: droplet for droplet in groundtruth.droplets}
        stats_groundtruth:Statistics = groundtruth.stats

        vmd_accum_gt += stats_groundtruth.vmd_value
        rsf_accum_gt += stats_groundtruth.rsf_value
        perc_accum_gt += stats_groundtruth.coverage_percentage
        no_accum_gt += stats_groundtruth.no_droplets
        percov_accum_gt += stats_groundtruth.overlaped_percentage

        # calculate accuracy values
        acc:Accuracy = Accuracy(droplets_calculated_dict, droplets_groundtruth_dict, filename, stats_calculated, stats_groundtruth)

        TP_overlapped += acc.true_positives_overlapped
        FP_overlapped += acc.false_positives_overlapped
        FN_overlapped += acc.false_negatives_overlapped

        IOU += acc.iou_overall
        dice += acc.dice_coefficient
        k+=1
        print(k, filename)
        if k==10: 
            print("end")
            break
        
    IOU = IOU / 10
    dice = dice / 10
    precision_overlapped, recall_overlapped, f1_score_overlapped, = Accuracy.calculate_parameters(acc, TP_overlapped, 0, FP_overlapped, FN_overlapped)

    Accuracy.write_final_accuracy_file(precision_overlapped, recall_overlapped, f1_score_overlapped, IOU, dice)
    Accuracy.write_final_statistics_file(stats_groundtruth, vmd_accum_gt, rsf_accum_gt, perc_accum_gt, no_accum_gt, percov_accum_gt, vmd_accum_c, rsf_accum_c, perc_accum_c, no_accum_c, percov_accum_c)

# else: 
    # for file in os.listdir(config.DATA_REAL_RAW_IMAGE_DIR):
    #     # get name of the file
    #     parts = file.split(".")
    #     filename = parts[0]

    #     # generate undistorted image
    #     try:    
    #         image = cv2.imread(os.path.join(config.DATA_REAL_RAW_IMAGE_DIR, file), cv2.IMREAD_GRAYSCALE)
    #         image_color = cv2.imread(os.path.join(config.DATA_REAL_RAW_IMAGE_DIR, file))
    #         dist = Distortion(image, image_color, filename, save_photo=True)
    #         if dist.noPaper: continue

    #         undistorted_image = cv2.imread(os.path.join(config.RESULTS_CV_UNDISTORTED_DIR, filename + '.png'), cv2.IMREAD_GRAYSCALE)
    #         undistorted_image_color = cv2.imread(os.path.join(config.RESULTS_CV_UNDISTORTED_DIR, filename + '.png'))
            
    #         calc_stats = Segmentation(undistorted_image_color, filename, save_images=True, create_masks=True)

    #         calc_stats.stats.save_stats_file(filename)

    #     except Exception as e:
    #         print("An error occurred:", e)
        


file = "Convencional_Interior_Arv1_2m.png"
filename = "Convencional_Interior_Arv1_2m"
image = cv2.imread(os.path.join(config.DATA_REAL_RAW_IMAGE_DIR, file), cv2.IMREAD_GRAYSCALE)
print(image)
plt.imshow(image)
plt.show()
image_color = cv2.imread(os.path.join(config.DATA_REAL_RAW_IMAGE_DIR, file))
dist = Distortion(image, image_color, filename, save_photo=True)

undistorted_image = cv2.imread(os.path.join(config.RESULTS_CV_UNDISTORTED_DIR, filename + '.png'), cv2.IMREAD_GRAYSCALE)
undistorted_image_color = cv2.imread(os.path.join(config.RESULTS_CV_UNDISTORTED_DIR, filename + '.png'))

calc_stats = Segmentation(undistorted_image_color, undistorted_image, filename, save_images=True, create_masks=True)

calc_stats.stats.save_stats_file(filename)


        






        
