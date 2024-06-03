import cv2
import os
import numpy as np
import sys

from matplotlib import pyplot as plt 

sys.path.insert(0, 'src/segmentation')
sys.path.insert(0, 'src/common')
from Distortion import *
from Util import * 

def segment_droplets(image_path, output_folder_single, output_folder_overlapped):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rectangle_roi, x_roi, y_roi = get_rectangle(image)
    if rectangle_roi is not None and rectangle_roi.size> 0:
        gray = cv2.cvtColor(rectangle_roi, cv2.COLOR_BGR2GRAY)
        # Masks for single and overlapped droplets
        single_mask = np.zeros_like(image)
        overlapped_mask = np.zeros_like(image)

        # Apply binary threshold
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        for contour in contours:
            # Calculate area and perimeter
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue

            adjusted_contour = contour + np.array([x_roi, y_roi])
            # Calculate circularity
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity > 0.8:  # Threshold for single droplets
                cv2.drawContours(single_mask, [adjusted_contour], -1, 255, thickness=cv2.FILLED)
            else:
                cv2.drawContours(overlapped_mask, [adjusted_contour], -1, 255, thickness=cv2.FILLED)
        
        # Save masks
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        cv2.imwrite(os.path.join(output_folder_single, f"{name}{ext}"), single_mask)
        cv2.imwrite(os.path.join(output_folder_overlapped, f"{name}{ext}"), overlapped_mask)

def get_rectangle(image):
    dist = Distortion(image, '_', False)
    if dist.noPaper: return

    rectangle = dist.largest_contour

    approx = cv2.approxPolyDP(rectangle, 0.009 * cv2.arcLength(rectangle, True), closed=True) 
    if len(approx) > 5: 
        return
        
    # order corners
    approx = sorted(approx, key=lambda x: x[0][0] + x[0][1])
    top_left = approx[0][0]
    top_right = approx[1][0]
    bottom_left = approx[2][0]
    bottom_right = approx[3][0]

    x1, y1 = top_left
    x2, y2 = bottom_right
    margin = 0
    x1 = max(x1 - margin, 0)
    y1 = max(y1 - margin, 0)
    x2 = min(x2 + margin, image.shape[1] - 1)
    y2 = min(y2 + margin, image.shape[0] - 1)
    
    rectangle_img = image[y1:y2, x1:x2]
    return rectangle_img, x1, y1


def process_folder(input_folder, output_folder_single, output_folder_overlapped):
    if not os.path.exists(output_folder_single):
        os.makedirs(output_folder_single)
    if not os.path.exists(output_folder_overlapped):
        os.makedirs(output_folder_overlapped)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            segment_droplets(image_path, output_folder_single, output_folder_overlapped)


# Usage
input_folder = 'images\\dataset_rectangle\\chen'  # Replace with your input folder path
output_folder_single = 'images\\dataset_rectangle\\masks\\single'  # Replace with your output folder path
output_folder_overlapped = 'images\\dataset_rectangle\\masks\\overlapped' 

delete_folder_contents(output_folder_overlapped)
delete_folder_contents(output_folder_single)

process_folder(input_folder, output_folder_single, output_folder_overlapped)
