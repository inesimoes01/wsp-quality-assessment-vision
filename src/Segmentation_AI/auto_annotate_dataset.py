import cv2
import os
import numpy as np
import sys

from matplotlib import pyplot as plt 

sys.path.insert(0, 'src/segmentation')
sys.path.insert(0, 'src/common')
import config as config
from Util import * 

def segment_droplets(image_path, output_folder_single, output_folder_overlapped):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # initialize masks for single and overlapped droplets
    single_mask = np.zeros_like(image)
    overlapped_mask = np.zeros_like(image)

    # binary threshold
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
      
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            cv2.drawContours(overlapped_mask, [contour], -1, 255, thickness=cv2.FILLED)
            continue

        # calculate circularity and apply threshold
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.8: 
            cv2.drawContours(single_mask, [contour], -1, 255, thickness=cv2.FILLED)
        else:
            cv2.drawContours(overlapped_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Save masks
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    cv2.imwrite(os.path.join(output_folder_single, f"{name}{ext}"), single_mask)
    cv2.imwrite(os.path.join(output_folder_overlapped, f"{name}{ext}"), overlapped_mask)

def mask_to_label(output_dir, class_id, filename):
    # load mask and get its contours
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # convert the contours to polygons
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 5:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append(polygon)
    
    # print the polygons
    with open('{}.txt'.format(os.path.join(output_dir, filename)), 'a') as f:
        for polygon in polygons:
            f.write(f"{class_id} {' '.join(map(str, polygon))}\n")


input_folder = config.DATA_REAL_PROC_IMAGE_DIR
output_folder_single = config.DATA_REAL_PROC_MASK_SIN_DIR
output_folder_overlapped = config.DATA_REAL_PROC_MASK_OV_DIR

if not os.path.exists(output_folder_single):
    os.makedirs(output_folder_single)
if not os.path.exists(output_folder_overlapped):
    os.makedirs(output_folder_overlapped)

delete_folder_contents(output_folder_overlapped)
delete_folder_contents(output_folder_single)

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        image_path = os.path.join(input_folder, filename)
        segment_droplets(image_path, output_folder_single, output_folder_overlapped)


output_label = config.DATA_REAL_PROC_LABEL_DIR

delete_folder_contents(output_label)

for file in os.listdir(output_folder_single):
    parts = file.split(".")
    filename = parts[0]
    
    image_path = os.path.join(output_folder_single, file)
    mask_to_label(output_label, 0, filename)

for file in os.listdir(output_folder_overlapped):
    parts = file.split(".")
    filename = parts[0]

    image_path = os.path.join(output_folder_single, file)
    mask_to_label(output_label, 1, filename)

    


