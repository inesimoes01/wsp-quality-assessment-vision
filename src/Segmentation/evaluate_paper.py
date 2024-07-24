import os
import sys 
import cv2
import numpy as np
sys.path.insert(0, 'src/common')
import config
import Segmentation.paper.ccv.Distortion as dist

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0: return 0.0
    iou = intersection / union

    return iou

def create_yolo_mask(file_path, width, height):
    with open(file_path, 'r') as file:
        parts = file.strip().split()            
        coordinates = list(map(float, parts[1:]))
        polygon = [(coordinates[i] * width, coordinates[i+1] * height) for i in range(0, len(coordinates), 2)]

    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], color=255)
    return mask


def evaluate_paper_segmentation():
    TP, FN, TN, FP = 0
    iou_sum = 0
    iou_threshold = 0.5

    directory = os.path.join(config.DATA_REAL_PAPER_DIR, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
    file_count = len([entry for entry in os.listdir(directory) if os.path.isfile(os.path.join(directory, entry))])
    gt_matched = [False] * file_count

    # apply the segmentation in each one of the images and then calculate the accuracy and save it
    for i, file in enumerate(os.listdir(directory)):
        parts = file.split(".")
        filename = parts[0]

        # apply cv algorithm
        im = cv2.imread(file)
        width, height = im.shape[:2]
        contour = dist.detect_rectangle_alternative(im)

        mask_predicted = np.zeros_like(im)
        cv2.drawContours(mask_predicted, [contour], -1, 255, cv2.FILLED)

        # compare with real mask
        mask_yolo = create_yolo_mask(os.path.join(config.DATA_REAL_PAPER_DIR, config.DATA_GENERAL_MASK_FOLDER_NAME, filename + ".txt"), width, height)

        iou = calculate_iou(mask_predicted, mask_yolo)
        iou_sum += iou
        print(iou)

        if iou >= iou_threshold:
            if not gt_matched[i]:
                TP += 1
                gt_matched[i] = True
            else:
                FP += 1
        else:
            FP += 1
        
    FN = gt_matched.count(False)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    average_iou = iou_sum / file_count

