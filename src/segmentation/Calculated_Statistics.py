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
from HoughTransform import *

#TODO better overlapping count
#TODO better coverage
#TODO remove contour inside contour in real images
#TODO quando uso

class Calculated_Statistics:
    def __init__(self, image, color_image, filename, save_images:bool, create_masks:bool):
        self.save_images = save_images
        self.create_masks = create_masks
        # if save_images:
        #     self.path_to_save_contours_overlapped = os.path.join(path_to_outputs_folder, "overlapped", filename)
        #     self.path_to_save_contours_single = os.path.join(path_to_outputs_folder, "single", filename)
        #     create_folders(self.path_to_save_contours_overlapped)
        #     create_folders(self.path_to_save_contours_single)

        # save each step in a different image
        self.roi_image = copy.copy(image)
        self.contour_image = copy.copy(color_image)
        self.detected_image = copy.copy(color_image)
        self.hull_image = copy.copy(color_image)
        self.separate_image = copy.copy(color_image)
        # get objects from image
        self.image = copy.copy(image)
        self.get_contours()



        # create masks
        if self.create_masks:
            mask = np.zeros_like(image)
            self.mask_overlapped = copy.copy(mask)
            self.mask_single = copy.copy(mask)

        # initialize variables
        self.droplets_data:list[Droplet]=[]
        self.contour_area = 0
        i=0
        
        # sort contours to catch the biggest ones first and remove the smaller ones inside before
        self.contours = sorted(self.contours, key=cv2.contourArea, reverse=True)

        self.ignore_ids = []

        while(i < len(self.contours)):
            if i in ignore_ids: 
                i += 1
                continue

            contour = self.contours[i]
            contour_area = cv2.contourArea(contour)
            self.contour_area += contour_area
            overlapped_ids = []

            # treat small contours like a perfect circle
            if len(contour) < 5: 
                area = cv2.contourArea(contour)
                (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
                diameter = 0.95*(np.sqrt((4*area)/np.pi))**0.91
                self.droplets_data.append(Droplet(False, int(center_x), int(center_y), float(diameter), int(i), overlapped_ids))
                cv2.circle(self.detected_image, (int(center_x), int(center_y)), int(radius), (255, 255, 255), 2)
                i += 1
                continue
           
            # when contour is not closed there is a chance of wrongly detected contours inside
            if contour_area < cv2.arcLength(contour, True): 
                self.remove_inside_contours(contour)

            if i in self.ignore_ids: 
                i += 1
                continue             
                
            # treat small contours like a perfect circle
            if len(contour) < 5: 
                area = cv2.contourArea(contour)
                (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
                diameter = 0.95*(np.sqrt((4*area)/np.pi))**0.91
                self.droplets_data.append(Droplet(False, int(center_x), int(center_y), float(diameter), int(i), overlapped_ids))
                cv2.circle(self.detected_image, (int(center_x), int(center_y)), int(radius), (255, 255, 255), 2)
                i += 1
                continue
            
            # get ROI of the contour to analyze
            roi_mask, roi_img, x_roi, y_roi = self.crop_ROI(contour)

            # check the shape to categorize it into 0: single, 1: elipse, 2: overlapped
            shape, no_circles = self.check_for_shape(contour, roi_img, roi_mask)

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
                    # detect circles and only return the main ones by clustering
                    circles = HoughTransform(roi_mask, no_circles, contour, contour_area).circles
            
                    # save each one of the overlapped circles
                    if circles is not None:
                        circle_ids = list(range(i, i + len(circles)))
                        j = 0
                        
                        for circle in circles:
                            overlapped_ids = []
                            overlapped_ids = circle_ids[:j] + circle_ids[j+1:]
                            
                            self.droplets_data.append(Droplet(True, int(circle[0] + x_roi), int(circle[1] + y_roi), float(circle[2]*2), int(i + j), overlapped_ids))
                            cv2.circle(self.detected_image, (int(circle[0] + x_roi), int(circle[1] + y_roi)), int(circle[2]), (0, 255, 0), 2)
                            j+=1
            
            if self.create_masks: self.create_mask(shape, contour)  
            cv2.drawContours(self.contour_image, [contour], -1, (204, 51, 153), 1)

            i += 1

        # create the masks and calculate values for statistics
        if self.create_masks:
            cv2.imwrite(os.path.join(path_to_masks_overlapped_pred_folder, filename + '.png'), self.mask_overlapped)
            cv2.imwrite(os.path.join(path_to_masks_single_pred_folder, filename + '.png'), self.mask_single)

        cv2.imwrite(os.path.join(path_to_detected_circles, filename + ".png"), self.detected_image)
        cv2.imwrite(os.path.join(path_to_detected_circles, filename + "_countour.png"), self.contour_image)
        cv2.imwrite(os.path.join(path_to_detected_circles, filename + "_canny.png"), self.canny) 

        self.calculate_stats()

    def remove_inside_contours(self, contour, index):
        contour = cv2.convexHull(contour)
        count = 0

        (x2,y2), radius2 = cv2.minEnclosingCircle(contour)
        
        for contour_inside in self.contours:
            if count != index:
                for pt in contour_inside:
                    point = tuple(pt[0])
                    point = tuple([int(round(point[0]) ), int(round( point[1] )) ])
                    result = cv2.pointPolygonTest(contour, point, False)
                    if result >= 0:
                        (x1,y1),radius1 = cv2.minEnclosingCircle(contour_inside)
                        if radius1 > radius2:
                            self.ignore_ids.append(index)
                        else: self.ignore_ids.append(count)
                        break
                

            count += 1   
    
    def is_point_in_contour(self, contour, point):
        result = cv2.pointPolygonTest(contour, point, False)
        return result >= 0

    def find_number_of_circles(self, contour, roi_image):
        output_img = np.zeros_like(self.image)
        hull = cv2.convexHull(contour, returnPoints = False)
        hull_points = cv2.convexHull(contour)

        # if (self_intersections is False):
        try:
            defects = cv2.convexityDefects(contour, hull)
            no_convex_points = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    _, _, f, d = defects[i, 0]
                    far = tuple(contour[f][0])
                    depth = d / 256.0 

                    if depth > 1:
                        no_convex_points +=1
                        cv2.circle(output_img, far, 1, 255, -1)

            if no_convex_points > 0:
                return no_convex_points 
            else:
                return 1
        except Exception as e:
            print(e)
           
       
            # plt.imshow(roi_image)
            # plt.show()

            return -1

   
    def create_mask(self, category, contour):
        if category == 2:
            cv2.drawContours(self.mask_overlapped, [contour], -1, 255, thickness=cv2.FILLED)
        else:
            cv2.drawContours(self.mask_single, [contour], -1, 255, thickness=cv2.FILLED)

    def get_contours(self):        
        # thesholding
        img = copy.copy(self.image)
        img = cv2.GaussianBlur(img, (5, 5), 3, 3)

        # th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
        # th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
        # th4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 3)
        th5 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 1)
        #_, otsu_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # plt.imshow(th5)
        # plt.show()
        edges = cv2.Canny(th5, 170, 200)
        kernel = np.ones((5,5),np.uint8)
        #edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
        # edges = cv2.dilate(edges, kernel)
        # edges = cv2.erode(edges, kernel)
        #plotFourImages(th2, edges, th4, th5)

        self.canny = copy.copy(edges)
        self.contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = self.image.shape
        #self.contour_image = np.full((height, width, 3), (0), dtype=np.uint8)
        cv2.drawContours(self.contour_image, self.contours, -1,(255, 255, 255), 1)

        # save number of contours
        self.final_no_droplets = len(self.contours)
    
    def crop_ROI(self, contour):
        # create aux image
        output_image = np.zeros_like(self.roi_image)
        cv2.drawContours(output_image, [contour], -1, 255, 1)

        x, y, w, h = cv2.boundingRect(contour)

        x_border = max(0, x - border_expand)
        y_border = max(0, y - border_expand)
        w_border = min(self.contour_image.shape[1] - x_border, w + 2 * border_expand)
        h_border = min(self.contour_image.shape[0] - y_border, h + 2 * border_expand)

        object_roi_mask = output_image[y_border:y_border + h_border, x_border:x_border + w_border]
        object_roi_img = self.roi_image[y_border:y_border + h_border, x_border:x_border + w_border]
        object_roi_img = cv2.cvtColor(object_roi_img, cv2.COLOR_RGB2BGR)
              
        return object_roi_mask, object_roi_img, x_border, y_border

    def check_for_shape(self, contour, roi_img, roi_mask):
        # fit elipse to the shape
        ellipse = cv2.fitEllipse(contour)
        (_, axes, _) = ellipse

        # check if contour is a circle or an ellipse based on aspect ratio, circularity and area
        aspect_ratio = axes[0] / axes[1]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        area_elipse = np.pi * axes[0]/2 * axes[1]/2


        no_circles = self.find_number_of_circles(contour, roi_img)

        # circle
        if no_circles == -1:
    
            cv2.drawContours(self.separate_image, [contour], -1, (102, 0, 204), 2)
            shape = 0

        elif aspect_ratio > elipse_threshold and circularity > circularity_threshold and no_circles == 1: 
            cv2.drawContours(self.separate_image, [contour], -1, (102, 0, 204), 2)
            shape = 0
        # elipse
        elif area_elipse - area < elipse_area_threshold and circularity < circularity_threshold and no_circles == 1:
            cv2.drawContours(self.separate_image, [contour], -1, (204, 51, 153), 2)
            shape = 1
        # elipse second chance
        elif no_circles == 1:
            cv2.drawContours(self.separate_image, [contour], -1, (204, 51, 153), 2)
            shape = 1
        else:
            cv2.drawContours(self.separate_image, [contour], -1, (44, 156, 63), 2)
            shape = 2

        return shape, no_circles

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

         # check for self intersections
        # self_intersections = False
        # for i in range(len(contour)):
        #     for j in range(i + 2, len(contour) - (1 if i == 0 else 0)):
        #         if cv2.norm(contour[i] - contour[j]) < 1e-5:
        #             cv2.drawContours()
        #             cv2.drawContours(output_img, [hull_points], -1, 255, 1)
        #             plt.imshow(output_img)
        #             plt.show()

        #             self_intersections = True  
        
            
                    # if one circle is almost completely within the other, remove the outside one
                    #elif dist <= r2 or dist <= r1:
                    # elif dist + min(r1, r2) <= max(r1, r2) or dist <= min(r1, r2):
                    #     if (r2 <= r1):
                    #         circles = np.delete(np.array(circles), i_count, axis = 0)
                    #     else: 
                    #         circles = np.delete(np.array(circles), j_count, axis = 0)
                               # part1 = r1**2 * math.acos((dist**2 + r1**2 - r2**2) / (2 * dist * r1))
                        # part2 = r2**2 * math.acos((dist**2 + r2**2 - r1**2) / (2 * dist * r2))
                        # part3 = 0.5 * math.sqrt((-dist + r1 + r2) * (dist + r1 - r2) * (dist - r1 + r2) * (dist + r1 + r2))
                        # overlapping_area = part1 + part2 + part3

                        # # if circles are too overlapped they loose meaning
                        # if overlapping_area > area * 0.60:
                        #     circles = np.delete(np.array(circles), j_count, axis = 0)
                        # else: j_count += 1


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