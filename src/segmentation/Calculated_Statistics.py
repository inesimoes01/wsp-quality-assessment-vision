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
from Algorithms import *

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
        
        # create masks
        mask = np.zeros_like(image)
        self.mask_overlapped = copy.copy(mask)
        self.mask_single = copy.copy(mask)

        # calculate diameter + save each contour   
        self.droplets_data:list[Droplet]=[]     
        for i, contour in enumerate(self.contours): 
            overlapped_ids = []
            
            # if contour is too small
            if len(contour) < 5: continue

            # if contour is not closed
            if cv2.contourArea(contour) < cv2.arcLength(contour, True): contour = cv2.convexHull(contour)
            
            isOverlapped = self.check_for_shape(contour)

            # crop ROI of the droplet
            file, x_roi, y_roi = self.crop_ROI(contour, isOverlapped, i)

            # #contour = self.close_contour(contour)
            # contour = cv2.convexHull(contour)

            if isOverlapped: 
                img = cv2.imread(file)
                edges = self.process_image(img)
                circles = self.hough_tansform(file, edges, img)

                # save each circle detected
                # if no circles detected, we assume its an ellipse
                #TODO maybe remove the circles too close together? 
                if circles is not None:
                    circle_ids = list(range(i, i + len(circles) + 1))
                    j = 0
                    for circle in circles[0,:]:
                        overlapped_ids = []
                        overlapped_ids = circle_ids[:j] + circle_ids[j+1:]
                        j+=1
                        
                        self.droplets_data.append(Droplet(True, int(circle[0] + x_roi), int(circle[1] + y_roi), float(circle[2]*2), int(i), overlapped_ids))
                       
                        i+=1 
                    self.save_contour(0, contour)
                   
                else: 
                    (x, y), (_, minor), _ = cv2.fitEllipse(contour)
                    overlapped_ids = []
                  
                    self.droplets_data.append(Droplet(True, int(x), int(y) , float(minor), int(i), overlapped_ids))
                    self.save_contour(1, contour)
                    #cv2.drawContours(mask_single, [contour], -1, 255, thickness=cv2.FILLED)

            else: 
                # find values for the contour
                area = cv2.contourArea(contour)
                (center_x, center_y), _ = cv2.minEnclosingCircle(contour)
                diameter = 0.95*(np.sqrt((4*area)/np.pi))**0.91
                self.droplets_data.append(Droplet(False, int(center_x), int(center_y), float(diameter), int(i), overlapped_ids))
                self.save_contour(1, contour)
                #cv2.drawContours(mask_single, [contour], -1, 255, thickness=cv2.FILLED)
                
            i += 1

        # create the masks and calculate values for statistics
        
        cv2.imwrite(path_to_masks_overlapped_pred_folder + '\\' + filename + '.png', self.mask_overlapped)
        cv2.imwrite(path_to_masks_single_pred_folder + '\\' + filename + '.png', self.mask_single)

        self.calculate_stats()
        

    
    def save_contour(self, category, contour):
        if category == 0:
            cv2.drawContours(self.mask_overlapped, [contour], -1, 255, thickness=cv2.FILLED)
        elif category == 1:
            cv2.drawContours(self.mask_single, [contour], -1, 255, thickness=cv2.FILLED)



    def get_contours(self):        
        # thesholding
        img = copy.copy(self.image)
        img = cv2.GaussianBlur(img, (5, 5), 3, 3)
        th3 = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 3)
        _, otsu_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        edges = cv2.Canny(th3, 150, 200)

        kernel = np.ones((100,100),np.uint8)
        cv2.dilate(edges, kernel, iterations=1)

        self.contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.image, self.contours, -1, (255), 1)
        

        self.contour_area = 0
        for contour in self.contours:
            self.contour_area += cv2.contourArea(contour)
        
        # cv2.drawContours(self.image, self.contours, -1, (255), 1)
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
        cv2.putText(self.enumerate_image, f'{index}', (int(x-radius), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # save outputs
        object_roi = cv2.cvtColor(object_roi, cv2.COLOR_RGB2BGR)
        if (isOverlapped): 
            path = self.path_to_save_contours_overlapped + '\\' + str(index) + '.png'
            cv2.imwrite(path, object_roi)
            return path, x, y
        else: 
            path = self.path_to_save_contours_single + '\\' + str(index) + '.png'
            cv2.imwrite(path, object_roi)
            return path, x, y

    def check_for_shape(self, contour):
        # bool variable to check if the algorithm sees the object as overlapped / ellipse or single droplets
        isOverlapped = False
        # fit ellipse
        if len(contour) < 5:
            return 
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse

        # check if contour is a circle or an ellipse based on aspect ratio
        aspect_ratio = axes[0] / axes[1]
        # if aspect_ratio < 0.9:
        #     isOverlapped = True


        # # perform shape analysis
        # area = cv2.contourArea(contour)
        # perimeter = cv2.arcLength(contour, True)

        # # droplets of only 1 pixel
        # if(diameter <= 4):    
        #     cv2.drawContours(self.separate_image, [contour], -1, (102, 0, 204), 2)
        #     return isOverlapped

        # # calculate parameter to evaluate the circularity of the droplet
        # circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # classify based on properties
        if aspect_ratio > circularity_threshold: 
            # single
            cv2.drawContours(self.separate_image, [contour], -1, (102, 0, 204), 2)
        else:
            # ellipse or overlapped
            cv2.drawContours(self.separate_image, [contour], -1, (44, 156, 63), 2)
            isOverlapped = True

        return isOverlapped

    def calculate_stats(self):
        droplet_diameter = [d.diameter for d in self.droplets_data]

        image_height, image_width = self.image.shape[:2]

        self.volume_list = sorted(Statistics.diameter_to_volume(droplet_diameter, image_width))

        cumulative_fraction = Statistics.calculate_cumulative_fraction(self.volume_list)
        vmd_value = Statistics.calculate_vmd(cumulative_fraction, self.volume_list)
        rsf_value = Statistics.calculate_rsf(cumulative_fraction, self.volume_list, vmd_value)
        coverage_percentage = Statistics.calculate_coverage_percentage_c(self.image, image_height, image_width, self.contour_area)

        self.stats = Statistics(vmd_value, rsf_value, coverage_percentage, self.final_no_droplets, self.droplets_data)

    def process_image(self, image):
        image_blur = cv2.GaussianBlur(image, (7, 7), 1.5)
        gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 150, 200)

        _, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_points = [point[0] for contour in contours for point in contour]
        return edges

    def hough_tansform(self, file_path, edges, image):
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.3, minDist=1, param1=200, param2=23, minRadius=1, maxRadius=0)
        
        # draw detected circles
        if circles is not None:
            circles = np.uint16(np.around(circles))

            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(image, center, radius, (0, 255, 0), 2)
  
            cv2.imwrite(file_path, image)
        
        return circles
        #else:
            #print("No circles detected in the image.")
