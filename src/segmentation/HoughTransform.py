import cv2
import copy
import numpy as np
import sys
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

sys.path.insert(0, 'src/common')
from Util import *
from Variables import *

show_plots = False

class HoughTransform:
    def __init__(self, edges, x_roi, y_roi, area, no_circles):
        self.circles = self.hough_tansform(edges, x_roi, y_roi, area, no_circles)

    def hough_tansform(self, edges, x_roi, y_roi, area, no_circles):
    
        gray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(gray, 
                                   cv2.HOUGH_GRADIENT, 
                                   dp=1,                # Inverse ratio of accumulator resolution to image resolution. Higher values mean lower resolution/precision but potentially faster processing.
                                   minDist=1,           # Minimum distance between centers of detected circles.
                                   param1=200,          # Higher threshold of Canny edge detector.
                                   param2=12,           # Accumulator threshold for circle centers at the detection stage. Smaller values may lead to more false detections.
                                   minRadius=1,         # Minimum radius of circles to be detected.
                                   maxRadius=0)         # Maximum radius of circles to be detected. If negative, it defaults to the maximum image dimension.
        
        edges_final = copy.copy(edges)
     

        if circles is not None:
            circles = np.uint16(np.around(circles))
            circles = circles[0,:]

            # sort them so the smallest circles are less likely to be deleted ????
            circles = sorted(circles, key=lambda x: x[2], reverse=False)


            # if (show_plots):
            #for circle in circles:
                #cv2.circle(edges, (int(circle[0]), int(circle[1])), int(circle[2]), (204, 51, 153), 1)
                #cv2.circle(self.detected_image, (int(circle[0] + x_roi), int(circle[1] + y_roi)), circle[2], (204, 51, 153), 2)
            # plt.imshow(edges)
                # plt.show()

            # kmeans approximation
            circles = self.k_means_approximation(circles, edges_final, no_circles)


        return circles

    def k_means_approximation(self, circles, edges, no_circles):
        results = []

        if (len(circles) < no_circles): 
            return circles

        edges_aux = copy.copy(edges)
        model = KMeans(n_clusters=no_circles, random_state=0, n_init='auto').fit(circles)
        
        for circle in model.cluster_centers_:
            results.append(circle)

        return results

    def remove_bad_circles(self, circles, area, edges, gray):
        i_count = 0
        j_count = 0
        while(len(circles) > 1 and i_count < len(circles) - 2):
            circle1 = circles[i_count]
            j_count = 0

            while(len(circles) >= 2 and j_count < len(circles)): 
                if (j_count != i_count):
                    circle2 = circles[j_count]
                    r1 = int(circle1[2])
                    r2 = int(circle2[2])
                    x1 = int(circle1[0])
                    x2 = int(circle2[0])
                    y1 = int(circle1[1])
                    y2 = int(circle2[1])

                    no_pixels1 = self.calculate_no_pixels(x1, y1, r1, edges)
                    no_pixels2 = self.calculate_no_pixels(x2, y2, r2, edges)

                    # calculate distance between centers
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)   
                    distance_threshold_related_to_area = distance_threshold * area  

                    overlapping_pixels, total_pixels = self.check_overlapped(x1, x2, y1, y2, r1, r2, edges, gray) 

                    # if dist is bigger than the sum of the radius, circles dont overlap
                    if dist >= r1 + r2:
                        j_count += 1
                    
                    # one circle is within another
                    # elif overlapping_pixels > total_pixels * 0.40:
                    #     circles = np.delete(np.array(circles), j_count, axis = 0)

                    elif overlapping_pixels > min(no_pixels1, no_pixels2) * 0.90:
                        iou1, iou2 = self.check_iou(x1, x2, y1, y2, r1, r2, edges, gray)

                        if iou1 >= iou2:
                            circles = np.delete(np.array(circles), j_count, axis = 0)
                        else:
                            circles = np.delete(np.array(circles), i_count, axis = 0)
                        # if (no_pixels2 < no_pixels1):
                        #     circles = np.delete(np.array(circles), j_count, axis = 0)   
                        # else:
                        #     circles = np.delete(np.array(circles), j_count, axis = 0)   

                    # distnce between centers is small
                    elif (dist <= distance_threshold_related_to_area) or dist <= min(r1, r2) + 2:
                        iou1, iou2 = self.check_iou(x1, x2, y1, y2, r1, r2, edges, gray)

                        if iou1 >= iou2:
                            circles = np.delete(np.array(circles), j_count, axis = 0)
                        elif iou1 < iou2:
                            circles = np.delete(np.array(circles), i_count, axis = 0)
                  
                    # calculate overlapping area
                    else:
                        #TODO change this to be masks and overlap 
                        overlapping_pixels, total_pixels = self.check_overlapped(x1, x2, y1, y2, r1, r2, edges, gray) 
                        if overlapping_pixels > total_pixels * 0.40:
                            circles = np.delete(np.array(circles), j_count, axis = 0)
                        j_count += 1

                    
                    if i_count >= len(circles): 
                        break
                
                else: j_count += 1
            
            i_count += 1

        





        return circles
    
    def check_iou(self, x1, x2, y1, y2, r1, r2, edges, gray):
        mask_check1 = np.zeros_like(edges)
        mask_check2 = np.zeros_like(edges)
        mask_check_original = np.zeros_like(edges)

        original_contour, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in original_contour:
            cv2.drawContours(mask_check_original, [contour], -1, 255, thickness=cv2.FILLED)
        
        # check which circle hits more pixels from the original image
        cv2.circle(mask_check1, (x1, y1), r1, 255, thickness=cv2.FILLED)
        cv2.circle(mask_check2, (x2, y2), r2, 255, thickness=cv2.FILLED)
        
        iou1 = np.sum(np.logical_and(mask_check1, mask_check_original)) / np.sum(np.logical_or(mask_check1, mask_check_original))
        iou2 = np.sum(np.logical_and(mask_check2, mask_check_original)) / np.sum(np.logical_or(mask_check2, mask_check_original))
        
        return iou1, iou2
    
    def check_overlapped(self, x1, x2, y1, y2, r1, r2, edges, gray):
        # create mask1 and mask2
        mask_check1 = np.zeros_like(edges)
        mask_check2 = np.zeros_like(edges)
        total_mask = np.zeros_like(edges)        
        cv2.circle(mask_check1, (x1, y1), r1, 255, thickness=cv2.FILLED)
        cv2.circle(mask_check2, (x2, y2), r2, 255, thickness=cv2.FILLED)
        cv2.circle(total_mask, (x1, y1), r1, 255, thickness=cv2.FILLED)
        cv2.circle(total_mask, (x2, y2), r2, 255, thickness=cv2.FILLED)
       
        return np.sum(np.logical_and(mask_check1, mask_check2)), np.sum(total_mask == 255)

    def safe_acos(self, value):
        # Ensure value is within the valid range for acos
        return math.acos(min(1, max(-1, value)))
        # mask_check1 = np.zeros_like(edges)
        # mask_check2 = np.zeros_like(edges)
        
        # # check which circle hits more pixels from the original image
        # cv2.circle(mask_check1, (x1, y1), r1, 255, thickness=cv2.FILLED)
        # cv2.circle(mask_check2, (x2, y2), r2, 255, thickness=cv2.FILLED)

        # iou = np.sum(np.logical_and(mask_check1, mask_check2)) / np.sum(np.logical_or(mask_check1, mask_check2))
        
        # return iou     
    def calculate_no_pixels(self, x, y, r, edges):
        mask = np.zeros_like(edges)
        cv2.circle(mask, (x, y), r, 255, thickness=cv2.FILLED)
        return np.sum(mask == 255)

            # good_circles1 = self.remove_bad_circles(circles, area, edges, gray)

            # for circle in good_circles1:
            #     cv2.circle(edges_final, (int(circle[0]), int(circle[1])), int(circle[2]), (204, 51, 153), 1)

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
