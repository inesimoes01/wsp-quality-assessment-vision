import cv2
import numpy as np
import copy
import os
import sys

sys.path.insert(0, 'src/others')
from util import *
from paths import *

class Calculated_Statistics:
    def __init__(self, image, filename):
        # file management
        self.path_to_save_contours_single = os.path.join(path_to_outputs_folder, "single", filename)
        self.path_to_save_contours_overlapped = os.path.join(path_to_outputs_folder, "overlapped", filename)
        create_folders(self.path_to_save_contours_overlapped)
        create_folders(self.path_to_save_contours_single)

        self.image = copy.copy(image)

        # get all the contours
        self.get_contours()

        # save each step in a different image
        self.roi_image = copy.copy(image)
        self.enumerate_image = copy.copy(image)
        self.diameter_image = copy.copy(image)
        self.separate_image = copy.copy(image)

        # calculate diameter + save each contour        
        for i, contour in enumerate(self.contours):    
            center_x, center_y, radius = self.measure_diameter_droplet(contour)

            overlapped_ids, isOverlapped = self.check_for_overlapped(contour, radius)

            self.save_droplet_information(i, center_x, center_y, radius, overlapped_ids)
        
            self.crop_ROI(contour, isOverlapped, i)

            self.separate_image = cv2.cvtColor(self.separate_image, cv2.COLOR_RGB2BGR)
        
        # save final image
        cv2.imwrite(path_to_separation_folder + f'\\result_image_' + filename + '.png', self.separate_image)

    def get_contours(self):
        # grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.gray_image = copy.copy(gray)

        # canny edge detectio + thresholding + contours
        edges = cv2.Canny(gray, 50, 150)
        _, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
        self.contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # draw contours
        cv2.drawContours(self.image, self.contours, -1, (0, 255, 0), 2)
        self.contour_image = copy.copy(self.image)

        # number of contours
        self.final_no_droplets = len(self.contours)
    
    def crop_ROI(self, contour, isOverlapped, index):
        # crop region of interest
        x, y, w, h = cv2.boundingRect(contour)
        expansion_factor = 2
        expanded_w = int(w * expansion_factor)
        expanded_h = int(h * expansion_factor)
        x -= int((expanded_w - w) / 2)
        y -= int((expanded_h - h) / 2)
        x = max(x, 0)
        y = max(y, 0)
        object_roi = self.roi_image[y:y+expanded_h, x:x+expanded_w]
        (x, y), radius = cv2.minEnclosingCircle(contour)
        cv2.putText(self.enumerate_image, f'{index}', (int(x-radius), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # save outputs
        object_roi = cv2.cvtColor(object_roi, cv2.COLOR_RGB2BGR)

        if (isOverlapped): cv2.imwrite(self.path_to_save_contours_overlapped + '\\' + str(index) + '.png', object_roi)
        else: cv2.imwrite(self.path_to_save_contours_single + '\\' + str(index) + '.png', object_roi)
    
    def measure_diameter_droplet(self, contour):   
        # find minimum enclosing circle
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        diameter = radius * 2

        # annotate image for diameter values
        center = (int(center_x), int(center_y))
        radius = int(radius)
        cv2.circle(self.diameter_image, center, radius, (255, 0, 0), 2)
        cv2.putText(self.diameter_image, f'{diameter:.2f}', (int(center_x-radius), int(center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2))

        return center_x, center_y, radius

    def check_for_overlapped(self, contour, radius):
        # bool variable to check if the algorithm sees the object as overlapped or single droplets
        isOverlapped = 0

        # perform shape analysis
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # droplets of only 1 pixel
        if(radius <= 1):    
            cv2.drawContours(self.separate_image, [contour], -1, (102, 0, 204), 2)
            return isOverlapped

        # calculate parameter to evaluate the circularity of the droplet
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # classify based on properties
        if circularity > 0.8: 
            cv2.drawContours(self.separate_image, [contour], -1, (102, 0, 204), 2)
        else:
            cv2.drawContours(self.separate_image, [contour], -1, (44, 156, 63), 2)
            isOverlapped = 1
            self.final_no_droplets += 1

        return isOverlapped
    
    def save_droplet_information(self, index, center_x, center_y, radius, overlapped_ids):
        self.droplets_data.append({
            'id': int(index),
            'center_x': int(center_x),
            'center_y': int(center_y),
            'radius': int(radius),
            'overlappedIds': overlapped_ids
        })    


