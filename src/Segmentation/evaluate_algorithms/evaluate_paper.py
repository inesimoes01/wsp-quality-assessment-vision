import os
import sys 
import cv2
import numpy as np
import csv
import time

import paper.ccv.Distortion as dist

sys.path.insert(0, 'src/common')
import config

def calculate_iou(mask1, mask2):
    _, mask1_binary = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
    _, mask2_binary = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    union = np.logical_or(mask1_binary, mask2_binary).sum()
    
    if union == 0: return 0.0
    iou = intersection / union

    return iou


def create_yolo_mask(file_path, width, height):
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()             
            coordinates = list(map(float, parts[1:]))
            polygon = [(coordinates[i] * width, coordinates[i+1] * height) for i in range(0, len(coordinates), 2)]

        
    mask = np.zeros((width, height), dtype=np.uint8)
    
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], color=255)
    return mask

def write_final_csv(metric, fieldnames_csv, path_rectangle):
    with open(path_rectangle, mode='a', newline='') as file:
        new_row = {
                "file": metric[0], "iou": metric[1], "segmentation_time": metric[2]
            }
        writer = csv.DictWriter(file, fieldnames=fieldnames_csv)
        writer.writerow(new_row)

def create_binary_mask(segmentation_result, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for polygon in segmentation_result:
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask

def main_ccv(path_evalutation_paper_ccv, paper_fieldnames, path_dataset):
    directory_image = os.path.join(path_dataset, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
    directory_label = os.path.join(path_dataset, config.DATA_GENERAL_LABEL_FOLDER_NAME)

    # start file
    with open(path_evalutation_paper_ccv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=paper_fieldnames)
        writer.writeheader()

    # apply the segmentation in each one of the images and then calculate the accuracy and save it
    for i, file in enumerate(os.listdir(directory_image)):
        start_time = time.time()
        filename = os.path.splitext(os.path.basename(file))[0]

        # apply cv algorithm
        im = cv2.imread(os.path.join(directory_image, file))
        width, height = im.shape[:2]
        
        contour = dist.detect_rectangle_alternative(im)

        seg_time = time.time()

        mask_predicted = np.zeros((width, height), dtype=np.uint8)
        cv2.drawContours(mask_predicted, [contour], -1, 255, cv2.FILLED)
        
        # compare with real mask
        mask_groundtruth = create_yolo_mask(os.path.join(directory_label, filename + ".txt"), width, height)
        
        iou = calculate_iou(mask_predicted, mask_groundtruth)
    
        segmentation_time = seg_time - start_time
        write_final_csv((filename, iou, segmentation_time), paper_fieldnames, path_evalutation_paper_ccv)
       
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("Time taken:", elapsed_time, "seconds")


def main_yolo(path_evalutation_paper_ccv, paper_fieldnames, path_dataset, model_yolo):
    directory_image = os.path.join(path_dataset, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
    directory_label = os.path.join(path_dataset, config.DATA_GENERAL_LABEL_FOLDER_NAME)

    # start file
    with open(path_evalutation_paper_ccv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=paper_fieldnames)
        writer.writeheader()

    # apply the segmentation in each one of the images and then calculate the accuracy and save it
    for i, file in enumerate(os.listdir(directory_image)):
        start_time = time.time()
        filename = os.path.splitext(os.path.basename(file))[0]

        # apply cv algorithm
        im = cv2.imread(os.path.join(directory_image, file))
        width, height = im.shape[:2]
        
        image = cv2.imread(os.path.join(directory_image, file))
        results = model_yolo(image)
        segmentation_result = results[0].masks.xy
        mask_predicted = create_binary_mask(segmentation_result, image.shape)

        seg_time = time.time()

        # compare with real mask
        mask_groundtruth = create_yolo_mask(os.path.join(directory_label, filename + ".txt"), width, height)
        
        iou = calculate_iou(mask_predicted, mask_groundtruth)
    
        segmentation_time = seg_time - start_time
        write_final_csv((filename, iou, segmentation_time))
       
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("Time taken:", elapsed_time, "seconds")

        