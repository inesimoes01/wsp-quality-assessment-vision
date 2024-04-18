import cv2
import numpy as np
import copy
import os
import sys

sys.path.insert(0, 'src/common')
from Util import *
from Variables import *
from Droplet import *
from Statistics import * 
from Distortion import *

#TODO better overlapping count
#TODO better coverage
#TODO remove contour inside contour in real images

#TODO quando uso

class Calculated_Statistics:
    def __init__(self, image, filename, path_to_save_contours_overlapped, path_to_save_contours_single):
        self.path_to_save_contours_overlapped = path_to_save_contours_overlapped
        self.path_to_save_contours_single = path_to_save_contours_single
        # get objects from image
        self.image = copy.copy(image)
        self.get_contours()

        # save each step in a different image
        self.roi_image = copy.copy(image)
        self.enumerate_image = copy.copy(image)
        self.diameter_image = copy.copy(image)
        self.separate_image = copy.copy(image)

        # calculate diameter + save each contour   
        self.droplets_data:list[Droplet]=[]     
        for i, contour in enumerate(self.contours):
            overlapped_ids = []    

            # find values for the contour
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)

            # annotate image for diameter values
            center = (int(center_x), int(center_y))
            radius = int(radius)
            cv2.circle(self.diameter_image, center, radius, (255, 0, 0), 2)

            # check if the contour is a singular or overlapped droplet
            isOverlapped = self.check_for_overlapped(contour, radius)

            # save droplet information
            if isOverlapped: 
                overlapped_ids.append(i+1)
                self.droplets_data.append(Droplet(False, int(center_x), int(center_y), int(radius), int(i), overlapped_ids))
                overlapped_ids = []
                overlapped_ids.append(i)
                self.droplets_data.append(Droplet(False, int(center_x), int(center_y), int(radius), int(i+1), overlapped_ids))

            else: self.droplets_data.append(Droplet(False, int(center_x), int(center_y), int(radius), int(i), overlapped_ids))
        
            # crop ROI of the droplet
            self.crop_ROI(contour, isOverlapped, i)
            
            i += 1
        
        # calculate values for statistics
        self.calculate_stats()

    def get_contours(self):
        # grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.gray_image = copy.copy(gray)
        
        # thesholding
        img = cv2.medianBlur(self.gray_image, 5)
        #th3 = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 3)
        # th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,17, 5)
        # blur = cv2.GaussianBlur(img,(5,5),0)
        # _,th1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        _, otsu_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #th4 = cv2.adaptiveThreshold(otsu_threshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

        edges = cv2.Canny(otsu_threshold, 150, 200)
        # plt.imshow(edges)
        # plt.show()
        self.contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.contour_area = 0
        for contour in self.contours:
            self.contour_area += cv2.contourArea(contour)
        
        # draw contours
        cv2.drawContours(self.image, self.contours, -1, (0, 255, 0), 1)

        self.contour_image = copy.copy(self.image)

        # save number of contours
        self.final_no_droplets = len(self.contours)
    
    def crop_ROI(self, contour, isOverlapped, index):
        # crop region of interest
        x, y, w, h = cv2.boundingRect(contour)

        expanded_w = int(w * border_expand)
        expanded_h = int(h * border_expand)
        x -= int((expanded_w - w) / 2)
        y -= int((expanded_h - h) / 2)
        x = max(x, 0)
        y = max(y, 0)
        object_roi = self.roi_image[y:y+expanded_h, x:x+expanded_w]
        (x, y), radius = cv2.minEnclosingCircle(contour)
        #cv2.putText(self.enumerate_image, f'{index}', (int(x-radius), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

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
        #cv2.putText(self.diameter_image, f'{diameter:.2f}', (int(center_x-radius), int(center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2))

        return center_x, center_y, radius

    def check_for_overlapped(self, contour, radius):
        # bool variable to check if the algorithm sees the object as overlapped or single droplets
        isOverlapped = 0

        # perform shape analysis
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # droplets of only 1 pixel
        if(radius <= 2):    
            cv2.drawContours(self.separate_image, [contour], -1, (102, 0, 204), 2)
            return isOverlapped

        # calculate parameter to evaluate the circularity of the droplet
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # classify based on properties
        if circularity > circularity_threshold: 
            cv2.drawContours(self.separate_image, [contour], -1, (102, 0, 204), 2)
        else:
            cv2.drawContours(self.separate_image, [contour], -1, (44, 156, 63), 2)
            isOverlapped = 1
            self.final_no_droplets += 1

        return isOverlapped

    def calculate_stats(self):
        droplet_radii = [d.radius for d in self.droplets_data]
        image_height, image_width = self.image.shape[:2]

        cumulative_fraction = Statistics.calculate_cumulative_fraction(droplet_radii)
        vmd_value = Statistics.calculate_vmd(cumulative_fraction, droplet_radii)
        rsf_value = Statistics.calculate_rsf(cumulative_fraction, vmd_value)
        coverage_percentage = Statistics.calculate_coverage_percentage_c(self.image, image_height, image_width, self.contour_area)

        self.stats = Statistics(vmd_value, rsf_value, coverage_percentage, self.final_no_droplets, self.droplets_data)

        # print(vmd_value)
        # print(self.final_no_droplets)
        # print(rsf_value)
        # print(coverage_percentage)

