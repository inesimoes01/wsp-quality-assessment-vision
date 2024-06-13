import cv2
import copy
import numpy as np
import sys
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import array as arr

sys.path.insert(0, 'src/common')
from Util import *
from Variables import *

show_plots = False

class HoughTransform:
    def __init__(self, roi, no_convex_points, contour, area, roi_img):
        self.circles = self.hough_tansform(roi, no_convex_points, contour, area, roi_img)

    def hough_tansform(self, roi, no_convex_points, contour, area, roi_img):
    
        #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(roi, 
                                   cv2.HOUGH_GRADIENT, 
                                   dp=1,                # Inverse ratio of accumulator resolution to image resolution. Higher values mean lower resolution/precision but potentially faster processing.
                                   minDist=1,           # Minimum distance between centers of detected circles.
                                   param1=200,          # Higher threshold of Canny edge detector.
                                   param2=10,           # Accumulator threshold for circle centers at the detection stage. Smaller values may lead to more false detections.
                                   minRadius=1,         # Minimum radius of circles to be detected.
                                   maxRadius=0)         # Maximum radius of circles to be detected. If negative, it defaults to the maximum image dimension.
        
        roi_final1 = copy.copy(roi)
        roi_final2 = copy.copy(roi)
        roi_final3 = copy.copy(roi_img)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            circles = circles[0,:]

            # kmeans approximation with the final number of circles
            circles = self.k_means_approximation(circles, roi_final1, no_convex_points)

            if len(circles) > 1:
                # create masks
                mask_check_calculated = np.zeros_like(roi)
                for circle in circles:
                    r = int(circle[2])
                    x = int(circle[0])
                    y = int(circle[1])
                    cv2.circle(mask_check_calculated, (x, y), r, 255, thickness=cv2.FILLED)
                    cv2.circle(roi_img, (x, y), r, 255, thickness=1)

                mask_check_groundtruth = np.zeros_like(roi)
                original_contour, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in original_contour:
                    cv2.drawContours(mask_check_groundtruth, [contour], -1, 255, thickness=cv2.FILLED)
                
                iou_original = np.sum(np.logical_and(mask_check_calculated, mask_check_groundtruth)) / np.sum(np.logical_or(mask_check_calculated, mask_check_groundtruth))

                # plt.imshow(roi_img)
                # plt.show()
                
                if iou_original < 0.8:
                    j_count = 0
                    
                    iou_array = []
                    for i_count in range(len(circles)):
                        mask_check_calculated = np.zeros_like(roi)

                        for j_count, circle in enumerate(circles):
                            if (j_count == i_count):
                                continue

                            r = int(circle[2])
                            x = int(circle[0])
                            y = int(circle[1])
                            cv2.circle(mask_check_calculated, (x, y), r, 255, thickness=cv2.FILLED)

                        iou_aux = np.sum(np.logical_and(mask_check_calculated, mask_check_groundtruth)) / np.sum(np.logical_or(mask_check_calculated, mask_check_groundtruth))
                        iou_array.append(iou_aux)
                    

                    max_iou = max(iou_array)
                    if max_iou > iou_original:
                        circles = np.delete(np.array(circles), iou_array.index(max_iou), axis = 0)
                          
                    mask_check_calculated = np.zeros_like(roi)
                    for circle in circles:
                        r = int(circle[2])
                        x = int(circle[0])
                        y = int(circle[1])
                        cv2.circle(mask_check_calculated, (x, y), r, 255, thickness=cv2.FILLED)
                        cv2.circle(roi_final3, (x, y), r, 255, thickness=2)
                    

                    # plt.imshow(roi_final3)
                    # plt.show()

            circles = self.remove_bad_circles(circles, area, roi, roi_final2)

        return circles

            
    def k_means_approximation(self, circles, roi, no_convex_points):
        results = []

        circles = np.unique(circles, axis=0)

        # if the number of convex point is not even, we assume the next value to be the number of circles
        if (no_convex_points % 2 == 1):
            no_convex_points += 1
       

        if (len(circles) < no_convex_points): 
            return circles
        
        roi_aux = copy.copy(roi)
        model = KMeans(n_clusters=no_convex_points, random_state=0, n_init='auto').fit(circles)
        
        for circle in model.cluster_centers_:
            results.append(circle)

        return results

    def remove_bad_circles(self, circles, area, roi, gray):
        i_count = 0
        j_count = 0

 
        while(len(circles) > 1 and i_count < len(circles) - 2):
            circle1 = circles[i_count]
            j_count = 0

            while(len(circles) > 1 and j_count < len(circles)): 
                if (j_count != i_count):
                    circle2 = circles[j_count]
                    r1 = int(circle1[2])
                    r2 = int(circle2[2])
                    x1 = int(circle1[0])
                    x2 = int(circle2[0])
                    y1 = int(circle1[1])
                    y2 = int(circle2[1])

                    # calculate distance between centers
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)   
                    distance_threshold_related_to_area = distance_threshold * area  

                    # if dist is bigger than the sum of the radius, circles dont overlap
                    if dist >= r1 + r2:
                        overlapping_area = 0
                        j_count += 1
                    
                    # if one circle is almost completely within the other
                    elif dist <= r2 or dist <= r1 or (dist <= distance_threshold_related_to_area):
                        overlapping_area = math.pi * min(r1, r2) ** 2

                        iou1, iou2 = self.check_iou(x1, x2, y1, y2, r1, r2, roi, gray)

                        if iou1 >= iou2:
                            circles = np.delete(np.array(circles), j_count, axis = 0)
                        elif iou1 < iou2:
                            circles = np.delete(np.array(circles), i_count, axis = 0)

                    # calculate overlapping area
                    else: 
                        #TODO change this to be masks and overlap 
                        part1 = r1**2 * math.acos((dist**2 + r1**2 - r2**2) / (2 * dist * r1))
                        part2 = r2**2 * math.acos((dist**2 + r2**2 - r1**2) / (2 * dist * r2))
                        part3 = 0.5 * math.sqrt((-dist + r1 + r2) * (dist + r1 - r2) * (dist - r1 + r2) * (dist + r1 + r2))
                        overlapping_area = part1 + part2 + part3

                        # if circles are too overlapped they loose meaning
                        if overlapping_area > area * 0.60:
                            circles = np.delete(np.array(circles), j_count, axis = 0)
                        else: j_count += 1
                    
                    if i_count >= len(circles): 
                        break
                
                else: j_count += 1
            
            i_count += 1
        return circles
    
    def check_iou(self, x1, x2, y1, y2, r1, r2, roi, gray):
        mask_check1 = np.zeros_like(roi)
        mask_check2 = np.zeros_like(roi)
        mask_check_original = np.zeros_like(roi)

        original_contour, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in original_contour:
            cv2.drawContours(mask_check_original, [contour], -1, 255, thickness=cv2.FILLED)
        
        # check which circle hits more pixels from the original image
        cv2.circle(mask_check1, (x1, y1), r1, 255, thickness=cv2.FILLED)
        cv2.circle(mask_check2, (x2, y2), r2, 255, thickness=cv2.FILLED)
        
        iou1 = np.sum(np.logical_and(mask_check1, mask_check_original)) / np.sum(np.logical_or(mask_check1, mask_check_original))
        iou2 = np.sum(np.logical_and(mask_check2, mask_check_original)) / np.sum(np.logical_or(mask_check2, mask_check_original))
        
        return iou1, iou2
    
    def check_overlapped(self, x1, x2, y1, y2, r1, r2, roi, gray):
        # create mask1 and mask2
        mask_check1 = np.zeros_like(roi)
        mask_check2 = np.zeros_like(roi)
        total_mask = np.zeros_like(roi)        
        cv2.circle(mask_check1, (x1, y1), r1, 255, thickness=cv2.FILLED)
        cv2.circle(mask_check2, (x2, y2), r2, 255, thickness=cv2.FILLED)
        cv2.circle(total_mask, (x1, y1), r1, 255, thickness=cv2.FILLED)
        cv2.circle(total_mask, (x2, y2), r2, 255, thickness=cv2.FILLED)
       
        return np.sum(np.logical_and(mask_check1, mask_check2)), np.sum(total_mask == 255)

    def safe_acos(self, value):
        # Ensure value is within the valid range for acos
        return math.acos(min(1, max(-1, value)))
        # mask_check1 = np.zeros_like(roi)
        # mask_check2 = np.zeros_like(roi)
        
        # # check which circle hits more pixels from the original image
        # cv2.circle(mask_check1, (x1, y1), r1, 255, thickness=cv2.FILLED)
        # cv2.circle(mask_check2, (x2, y2), r2, 255, thickness=cv2.FILLED)

        # iou = np.sum(np.logical_and(mask_check1, mask_check2)) / np.sum(np.logical_or(mask_check1, mask_check2))
        
        # return iou     
    def calculate_no_pixels(self, x, y, r, roi):
        mask = np.zeros_like(roi)
        cv2.circle(mask, (x, y), r, 255, thickness=cv2.FILLED)
        return np.sum(mask == 255)

            # good_circles1 = self.remove_bad_circles(circles, area, roi, gray)

            # for circle in good_circles1:
            #     cv2.circle(roi_final, (int(circle[0]), int(circle[1])), int(circle[2]), (204, 51, 153), 1)

            # if (show_plots):
            #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            #     axes[0].imshow(edges)
            #     axes[0].axis('off')  # Hide the axis
            #     axes[1].imshow(edges_final)
            #     axes[1].axis('off') 
            #     plt.show()


            # if good_circles1 is not None:
            #     good_circles2 = self.remove_bad_circles(good_circles1, area, edges, gray)
            #     return good_circles2
            
            #     if (show_plots):
            #         for circle in good_circles2:
            #             cv2.circle(edges_final2, (int(circle[0]), int(circle[1])), int(circle[2]), (204, 51, 153), 1)

            #         fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            #         axes[0].imshow(edges)
            #         axes[0].axis('off')  # Hide the axis
            #         axes[1].imshow(edges_final2)
            #         axes[1].axis('off') 
            #         plt.show()

            # else: return good_circles1
