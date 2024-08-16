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
    print(mask1_binary.shape)
    print(mask2_binary.shape)

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

def write_final_csv(metric):
    with open(os.path.join(config.RESULTS_ACCURACY_DIR, "paper_evaluation_cv.csv"), mode='a', newline='') as file:
        new_row = {
                "file": metric[0], "iou": metric[1], "segmentation_time": metric[2]
            }
        writer = csv.DictWriter(file, fieldnames=["file", "iou", "segmentation_time"])
        writer.writerow(new_row)


def main():
    directory_image = os.path.join(config.DATA_REAL_RAW_DIR, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
    directory_label = os.path.join(config.DATA_REAL_RAW_DIR, config.DATA_GENERAL_LABEL_FOLDER_NAME)
    file_count = len([entry for entry in os.listdir(directory_image) if os.path.isfile(os.path.join(directory_image, entry))])

    # start file
    with open(os.path.join(config.RESULTS_ACCURACY_DIR, "paper_evaluation_cv.csv"), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file", "iou", "segmentation_time"])
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
        write_final_csv((filename, iou, segmentation_time))
       
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("Time taken:", elapsed_time, "seconds")
        
main()
