import cv2
import sys
import time
import os
import numpy as np
import skimage

sys.path.insert(0, 'src')
import Common.Util as FoldersUtil
import Common.config as config
import Segmentation.droplet.ccv.Segmentation_CCV as seg
import WebService.mainServer as server
import Segmentation.paper.ccv.Distortion as distortion
from Common.Statistics import Statistics as stats



def extract_contour_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Get the largest contour (assuming the target object is the largest detected region)
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    else:
        raise ValueError("No contours found in the mask.")

def apply_mask_to_image(image, mask):
    # Convert the mask to a 3-channel image so it matches the original image's channels
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    masked_image = cv2.bitwise_and(image, mask_rgb)
    
    return masked_image

   
def calculate_points(contour):
    approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), closed=True) 
    
    if len(contour) > 5: 
        noPaper = True
        
    # order corners
    approx = sorted(approx, key=lambda x: x[0][0] + x[0][1])
    top_left = approx[0][0]
    top_right = approx[1][0]
    bottom_left = approx[2][0]
    bottom_right = approx[3][0]

    # L2 norm
    width_AD = np.sqrt(((top_left[0] - top_right[0]) ** 2) + ((top_left[1] - top_right[1]) ** 2))
    width_BC = np.sqrt(((bottom_left[0] - bottom_right[0]) ** 2) + ((bottom_left[1] - bottom_right[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
    
    height_AB = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    height_CD = np.sqrt(((bottom_right[0] - top_right[0]) ** 2) + ((bottom_right[1] - top_right[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))
    
    input_pts = np.float32([top_left, bottom_left, bottom_right, top_right])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight + 1],
                            [maxWidth +  1, maxHeight + 1],
                            [maxWidth + 1, 0]])
  
    return maxWidth, maxHeight, input_pts, output_pts



def create_binary_mask(segmentation_result, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for polygon in segmentation_result:
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask
 

def find_paper_yolo(image_to_analyze, filename, model_yolo):
    results = model_yolo(image_to_analyze)
    segmentation_result = results[0].masks.xy

    mask_predicted = create_binary_mask(segmentation_result, image_to_analyze.shape)

    masked_image = apply_mask_to_image(image_to_analyze, mask_predicted)

    contour = extract_contour_from_mask(mask_predicted)

    maxWidth, maxHeight, pts_src, pts_dst = calculate_points(contour)

    h, status = cv2.findHomography(pts_src, pts_dst)
    undistorted_image = cv2.warpPerspective(image_to_analyze, h, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    return undistorted_image

def find_paper_ccv(image_to_analyze, filename):
    contour = distortion.detect_rectangle_alternative(image_to_analyze, filename)

    maxWidth, maxHeight, pts_src, pts_dst = calculate_points(contour)

    h, status = cv2.findHomography(pts_src, pts_dst)
    undistorted_image = cv2.warpPerspective(image_to_analyze, h, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    undistorted_image_file = os.path.join(server.AUX_FOLDER, "undistorted_" + filename + ".png")
    cv2.imwrite(undistorted_image_file, undistorted_image)
    
    return undistorted_image

    
    
    
    