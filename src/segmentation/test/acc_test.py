import cv2
import numpy as np
from matplotlib import pyplot as plt 
def calculate_iou(gt_mask, pred_mask):
    
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_dice(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    dice = 2 * np.sum(intersection) / (np.sum(gt_mask) + np.sum(pred_mask))
    return dice

# Read the predicted and ground truth masks
predicted_mask = cv2.imread('images\\artificial_dataset\\masks\\c\\single\\2024-04-25_0.png', cv2.IMREAD_GRAYSCALE)
ground_truth_mask = cv2.imread('images\\artificial_dataset\\masks\\gt\\single\\2024-04-25_0.png', cv2.IMREAD_GRAYSCALE)


# Calculate IoU and Dice coefficient for each class



iou_per_class=  calculate_iou(predicted_mask, ground_truth_mask)
dice_coefficient_per_class= calculate_dice(predicted_mask, ground_truth_mask)

# Print or display the results for each class


print(iou_per_class)
print(dice_coefficient_per_class)
