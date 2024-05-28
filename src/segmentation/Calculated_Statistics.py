import cv2
import numpy as np
import copy
import os
import sys

sys.path.insert(0, 'src/common')
from Util import *
#from Variables import *
from Droplet import *
from Statistics import * 
from Distortion import *
from Algorithms import *

#TODO better overlapping count
#TODO better coverage
#TODO remove contour inside contour in real images
#TODO quando uso

class Calculated_Statistics:
    """
    Calculates the statistics of one image of a WSP 

    Attributes:
        image (MatLike): original image
        filename (str): name of the image to save the corresponding in between steps of the algorithm
        save_photo (bool): bool variable that indicates if the intermediate steps results should be saved
    """
    def __init__(self, image, color_image, filename, save_images:bool, create_masks:bool):
        self.save_images = save_images
        self.create_masks = create_masks
        if save_images:
            self.path_to_save_contours_overlapped = os.path.join(path_to_outputs_folder, "overlapped", filename)
            self.path_to_save_contours_single = os.path.join(path_to_outputs_folder, "single", filename)
            create_folders(self.path_to_save_contours_overlapped)
            create_folders(self.path_to_save_contours_single)

        # get objects from image
        self.image = copy.copy(image)
        self.get_contours()

        # save each step in a different image
        self.roi_image = copy.copy(image)
        self.enumerate_image = copy.copy(image)
        self.diameter_image = copy.copy(image)
        self.separate_image = copy.copy(image)
        self.detected_image = copy.copy(color_image)

        # create masks
        if self.create_masks:
            mask = np.zeros_like(image)
            self.mask_overlapped = copy.copy(mask)
            self.mask_single = copy.copy(mask)

        # calculate diameter + save each contour   
        self.droplets_data:list[Droplet]=[]

        for i, contour in enumerate(self.contours): 
            overlapped_ids = []
            contour_area = cv2.contourArea(contour)
            
            # eliminate small contours and try to close the contours
            if len(contour) < 5: continue
            if contour_area < cv2.arcLength(contour, True): contour = cv2.convexHull(contour)

            # crop ROI of the droplet to analyze
            roi_img, x_roi, y_roi = self.crop_ROI(contour)

            # check the shape to categorize it into 0: single, 1: elipse, 2: overlapped
            shape = self.check_for_shape(contour, roi_img)

            match shape:
                # circle single
                case 0:
                    area = cv2.contourArea(contour)
                    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
                    diameter = 0.95*(np.sqrt((4*area)/np.pi))**0.91
                    self.droplets_data.append(Droplet(False, int(center_x), int(center_y), float(diameter), int(i), overlapped_ids))

                    cv2.circle(self.detected_image, (int(center_x), int(center_y)), int(radius), (255, 255, 255), 2)
                    
                # elipse
                case 1:
                    (x, y), (major, minor), angle = cv2.fitEllipse(contour)
                    elipse = cv2.fitEllipse(contour)
                    overlapped_ids = []
                    
                    self.droplets_data.append(Droplet(True, int(x), int(y) , float(minor), int(i), overlapped_ids))
                    cv2.ellipse(self.detected_image, elipse, color = (204, 51, 153), thickness=2)
                                
                # circles overlapped
                case 2:
                    # detect circles
                    circles = self.hough_tansform(roi_img, x_roi, y_roi, contour_area)
            
                    # save each one of the overlapped circles
                    if circles is not None:
                        circle_ids = list(range(i, i + len(circles) + 1))
                        j = 0
                        
                        for circle in circles:
                            overlapped_ids = []
                            overlapped_ids = circle_ids[:j] + circle_ids[j+1:]
                            j+=1
                            
                            self.droplets_data.append(Droplet(True, int(circle[0] + x_roi), int(circle[1] + y_roi), float(circle[2]*2), int(i), overlapped_ids))
                            cv2.circle(self.detected_image, (int(circle[0] + x_roi), int(circle[1] + y_roi)), circle[2], (0, 255, 0), 2)
                            i+=1 
            
            if self.create_masks: self.create_mask(shape, contour)  
    
             
            i += 1

        # create the masks and calculate values for statistics
        if self.create_masks:
            cv2.imwrite(os.path.join(path_to_masks_overlapped_pred_folder, filename + '.png'), self.mask_overlapped)
            cv2.imwrite(os.path.join(path_to_masks_single_pred_folder, filename + '.png'), self.mask_single)

        cv2.imwrite(os.path.join(path_to_detected_circles, filename + ".png"), self.detected_image)

        self.calculate_stats()
    

    def create_mask(self, category, contour):
        if category == 2:
            cv2.drawContours(self.mask_overlapped, [contour], -1, 255, thickness=cv2.FILLED)
        else:
            cv2.drawContours(self.mask_single, [contour], -1, 255, thickness=cv2.FILLED)

    def get_contours(self):        
        # thesholding
        img = copy.copy(self.image)
        img = cv2.GaussianBlur(img, (5, 5), 3, 3)
        th3 = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 3)
        # _, otsu_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        edges = cv2.Canny(th3, 170, 200)
        kernel = np.ones((100, 100),np.uint8)
        cv2.dilate(edges, kernel, iterations=1)
        self.canny = copy.copy(edges)

        self.contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = self.image.shape
        self.contour_image = np.full((height, width, 3), (0), dtype=np.uint8)
        cv2.drawContours(self.contour_image, self.contours, -1, (255), 1)
        
        # calculate total area with detected droplets
        self.contour_area = 0
        for contour in self.contours:
            self.contour_area += cv2.contourArea(contour)

        # save number of contours
        self.final_no_droplets = len(self.contours)
    
    def crop_ROI(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        
        x_border = max(0, x - border_expand)
        y_border = max(0, y - border_expand)
        w_border = min(self.contour_image.shape[1] - x_border, w + 2 * border_expand)
        h_border = min(self.contour_image.shape[0] - y_border, h + 2 * border_expand)

        object_roi = self.contour_image[y_border:y_border + h_border, x_border:x_border + w_border]
        object_roi = cv2.cvtColor(object_roi, cv2.COLOR_RGB2BGR)
              
        return object_roi, x_border, y_border

    def check_for_shape(self, contour, roi_img):
        if len(contour) < 5:
            return 
        
        # fit elipse to the shape
        ellipse = cv2.fitEllipse(contour)
        (_, axes, _) = ellipse

        # check if contour is a circle or an ellipse based on aspect ratio, circularity and area
        aspect_ratio = axes[0] / axes[1]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        area_elipse = np.pi * axes[0]/2 * axes[1]/2
        
        # circle
        if aspect_ratio > elipse_threshold and circularity > circularity_threshold: 
            cv2.drawContours(self.separate_image, [contour], -1, (102, 0, 204), 2)
            shape = 0
        # elipse
        elif area_elipse - area < elipse_area_threshold and circularity < circularity_threshold:
            cv2.drawContours(self.separate_image, [contour], -1, (204, 51, 153), 2)
            shape = 1
        # overlapped
        else:
            cv2.drawContours(self.separate_image, [contour], -1, (44, 156, 63), 2)
            shape = 2

        return shape

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

    def hough_tansform(self, edges, x_roi, y_roi, area):
    
        gray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(gray, 
                                   cv2.HOUGH_GRADIENT, 
                                   dp=1,                # Inverse ratio of accumulator resolution to image resolution. Higher values mean lower resolution/precision but potentially faster processing.
                                   minDist=1,           # Minimum distance between centers of detected circles.
                                   param1=200,          # Higher threshold of Canny edge detector.
                                   param2=10,           # Accumulator threshold for circle centers at the detection stage. Smaller values may lead to more false detections.
                                   minRadius=1,         # Minimum radius of circles to be detected.
                                   maxRadius=0)         # Maximum radius of circles to be detected. If negative, it defaults to the maximum image dimension.
        
        edges_final = copy.copy(edges)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            circles = circles[0,:]
            i_count = 0
            j_count = 0

            # sort them so the smallest circles are less likely to be deleted 
            circles = sorted(circles, key=lambda x: x[2])

            for circle in circles:
                cv2.circle(edges, (int(circle[0]), int(circle[1])), int(circle[2]), (204, 51, 153), 1)
                #cv2.circle(self.detected_image, (int(circle[0] + x_roi), int(circle[1] + y_roi)), circle[2], (204, 51, 153), 2)
            # plt.imshow(edges)
            # plt.show()
            while(len(circles) > 1 and i_count < len(circles) - 1 ):
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

                            iou1, iou2 = self.check_iou(x1, x2, y1, y2, r1, r2, edges, gray)

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

            
            for circle in circles:
                cv2.circle(edges_final, (int(circle[0]), int(circle[1])), int(circle[2]), (204, 51, 153), 1)


            # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # axes[0].imshow(edges)
            # axes[0].axis('off')  # Hide the axis
            # axes[1].imshow(edges_final)
            # axes[1].axis('off') 


            return circles
            
        
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


#  for i, circle in enumerate(circles[0,:]):
#                 cv2.circle(self.detected_image, (int(circle[0] + x_roi), int(circle[1] + y_roi)), circle[2], (204, 51, 153), 2)
#                 bool_list = [False] * len(circles[0,:])
#                 for j, other_circle in enumerate(final_list[0,:]):
#                     # same circle
#                     if j == i:
#                         bool_list[j] = True
#                     else: 
#                         dist = np.linalg.norm(np.array(circle[:2]) - np.array(other_circle[:2]))
#                         if dist > distance_threshold: bool_list[j] = True                    
#                         else: bool_list[j] = False
                
#                 keep_circles.append(bool_list)
            
#             keep_one_overlapped_circle = False
#             for circle_no in range(len(circles[0,:])):
#                 keep_circle_count = 0
                
#                 for bool in keep_circles:
#                     if bool[circle_no] == True: keep_circle_count += 1
                
#                 if (keep_circle_count == len(circles[0,:])):
#                     filtered_circles.append(circles[0, circle_no])
#                 else: 
#                     if (keep_one_overlapped_circle == True):
#                         filtered_circles.append(circles[0, circle_no])
#                         keep_one_overlapped_circle = False
#                     else: keep_one_overlapped_circle = True
                    
            
#             # if all circles are too close together, keep the first
#             if len(filtered_circles) == 0:
#                 filtered_circles.append(circles[0, 0])
#             return filtered_circles
        # remove circles with center too close together
        # if circles is not None:
        #     circles = np.uint16(np.around(circles))
        #     circles = circles.tolist()
          
        #     eps = 50
        #     new_circles = set()
        #     while circles:
        #         circle = circles.pop()
        #         for other in circles:
        #             distance = np.sqrt((circle[0] - other[0])**2 + (circle[1] - other[1])**2)   
        #             if distance < 3:
        #                 break
        #         else:
        #             new_circles.add(circle)

        
        # if circles is not None:
        #     filtered_circles = []
        #     if self.save_images:
        #         circles = np.uint16(np.around(circles))

        #         filtered_circles = []
        #         for circle in circles:
        #             centers = circle[:, :2]
        #             radii = circle[:, 2]
        #             num_circles = len(circle)
        #             if num_circles == 1:
        #                 filtered_circles.append(circle)
        #             else:
        #                 keep_circle = True
        #                 for i in range(num_circles - 1):
                            
        #                     for j in range(i + 1, num_circles):
        #                         distance = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2) 
        #                         print(distance)
        #                         if distance < 3:
        #                         #if distance(centers[i], centers[j]) < min_distance * (radii[i] + radii[j]):
        #                             keep_circle = False
                                    
        #                             break
        #                     if not keep_circle:
        #                         break
        #                 if keep_circle:

        #                     filtered_circles.append(circle)
        #         #print("SIZE", len(circles))

                # circles_to_eliminate = set()
                
                # i = 0
                # for circle1 in circles[0,:]:
                #     for circle2 in circles[0,:]:
                #         distance = math.sqrt(abs((circle1[0] - circle2[0])**2 + (circle1[1] - circle2[1])**2))
                #         #print(distance)
                            
                #         if distance < 1 :
                #             circles_to_eliminate.add(circle2)
                            
                #     i+=1

                # circles = [center[0] for idx, center in enumerate(circles) if idx not in circles_to_eliminate]
                #print("TAMANHO", len(circles))
                #circles = np.uint16(np.around(circles))
                # img = copy.copy(edges)
                # # draw detected circles
                # for circle in circles[0,:]:
                #     center = (circle[0], circle[1])
                #     radius = circle[2]
                #     cv2.circle(img, center, radius, (0, 255, 0), 2)
                # images.append(img)
                #cv2.imwrite(file_path, image)