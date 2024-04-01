import sys
import cv2
from matplotlib import pyplot as plt 
import os
import copy
import numpy as np

from Calculated_Statistics import Calculated_Statistics
from GroundTruth_Statistics import GroundTruth_Statistics
from Accuracy import Accuracy

sys.path.insert(0, 'src/others')
from util import *
from variables import *

sys.path.insert(0, 'src')
from Droplet import *

# delete old outputs
delete_folder_contents(path_to_separation_folder)
delete_folder_contents(path_to_outputs_folder)

# for each one of the images of the dataset
for file in os.listdir(path_to_images_folder):
    # get name of the file
    parts = file.split(".")
    filename = parts[0]

    # treat image
    in_image = cv2.imread(os.path.join(path_to_images_folder, filename + ".png"))
    in_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
    out_image = copy.copy(in_image)

    # calculate statistics
    stats_calculated:list[Droplet] = Calculated_Statistics(out_image, filename).droplets_data

    # save ground truth
    stats_groundtruth:list[Droplet] = GroundTruth_Statistics(filename, out_image).droplets

    Accuracy(stats_calculated, stats_groundtruth, filename)









    # calculate metrics for measuring accuracy
    # Accuracy(stats_calculated, stats_groundtruth)

    # print("Number of droplets detected: ", self.final_no_droplets)
    # print("Number of overlapped droplets: ", self.final_no_droplets - self.object_count)
    # print("Number of droplets real: ", self.number_of_droplets)
    # print("Number of overlapped droplets real: ", self.no_overlapped_droplets)
    # print("")


    # path_to_save_contours_single = os.path.join(path_to_outputs_folder, "single", filename)
    # path_to_save_contours_overlapped = os.path.join(path_to_outputs_folder, "overlapped", filename)

    # # create folder to save outputs
    # create_folders(path_to_save_contours_overlapped)
    # create_folders(path_to_save_contours_single)

    # # read files
    # in_image, out_image, stats_file_path = read_files(filename)

    # # get all the contours
    # contours, contour_image, object_count = get_contours(out_image)

    # # ground truth of number of contours
    # number_of_droplets, no_overlapped_droplets = get_ground_truth(stats_file_path)

    # # save each step in a different variable
    # roi_image = copy.copy(in_image)
    # enumerate_image = copy.copy(in_image)
    # diameter_image = copy.copy(in_image)
    # separate_image = copy.copy(in_image)

    # # calculate diameter + save each contour
    # final_no_droplets = object_count
    
    # for i, contour in enumerate(contours):    
    #     separate_image, diameter_image, final_no_droplets, isOverlapped = measure_diameter(contour, final_no_droplets, diameter_image, separate_image)
    
    #     crop_ROI(i, contour, enumerate_image, roi_image, isOverlapped, filename)

    #     separate_image = cv2.cvtColor(separate_image, cv2.COLOR_RGB2BGR)
    
    # # save final image
    # cv2.imwrite(path_to_separation_folder + f'\\result_image_' + filename + '.png', separate_image)

    # print("Number of droplets detected: ", final_no_droplets)
    # print("Number of overlapped droplets: ", final_no_droplets - object_count)
    # print("Number of droplets real: ", number_of_droplets)
    # print("Number of overlapped droplets real: ", no_overlapped_droplets)
    # print("")

    # #display_results(number_of_droplets, object_count, separate_image, in_image)







   



