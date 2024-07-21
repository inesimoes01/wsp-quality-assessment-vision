import cv2
import numpy as np
import copy
import os
import sys
from matplotlib import pyplot as plt 

sys.path.insert(0, 'src/common')
import config

from Droplet import Droplet
from Statistics import Statistics
import HoughTransform

from shapely.geometry import Polygon

circle_color = (255, 0, 0)
elipse_color = (0, 255, 255)
overlapped_color = (0, 255, 0)
edge_color = (0, 0, 255)
detected_colors = [circle_color, elipse_color, overlapped_color, edge_color]


class Segmentation_CV:
    def __init__(self, image_color, image_gray, filename, save_image_steps:bool, create_masks:bool, segmentation_method:int, real_width, real_height):
        self.save_image_steps = save_image_steps
        self.create_masks = create_masks
        self.filename = filename
    

        self.initialize_variables(image_gray, image_color)
       
        self.get_contours()
       
        for i, contour in enumerate(self.contours):
            contour_area = cv2.contourArea(contour)
            self.contour_area += contour_area 
            overlapped_ids = []

            x_rect, y_rect, w_rect, h_rect  = cv2.boundingRect(contour)
            
            # if contour is small, it is assumed to be almost a circle
            if contour_area < 5 or len(contour) < 5:
                (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
                self.save_single_droplet(contour, i, center_x, center_y, contour_area, overlapped_ids, 0)
                continue

            M = cv2.moments(contour)
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

            # check for the shape by calculating the circularity, fitelipse and the number of convex points
            shape, no_droplets, isOnEdge = self.check_for_contour_shape(contour, contour_area, x_rect, y_rect, w_rect, h_rect)                

            if shape < 2 and not isOnEdge:   # single circle or single elipse
                self.save_single_droplet(contour, i, center_x, center_y, contour_area, overlapped_ids, shape)
                continue
            
            elif isOnEdge:    # single circle or single elipse on the edge of the frame

                original_contour = copy.copy(contour)
                roi_mask_list, roi_mask_filled_list, roi_img, roi_img_color, x_roi, y_roi, h_roi, w_roi, _ = self.crop_roi(contour, x_rect, y_rect, w_rect, h_rect)

                for roi_mask, roi_mask_filled in zip(roi_mask_list, roi_mask_filled_list):
                    circles, _, _, _ = HoughTransform.apply_hough_circles_with_kmeans(roi_mask, roi_mask_filled, no_droplets, contour, contour_area, roi_img_color, isOnEdge,  h_roi, w_roi)
                    
                    if circles is not None and len(circles) > 1:
                        self.save_overlapped_droplets(i, original_contour, circles, x_roi, y_roi, isOnEdge)
                    else:
                        # if no circles are found, it is assumed the contour is an elipse
                        self.save_single_droplet(contour, i, center_x, center_y, contour_area, overlapped_ids, 3)

            else:  # overlapping droplets

                match segmentation_method:
                    case 0: # hough circles with clustering
                        original_contour = copy.copy(contour)

                        # for hough, create a roi mask with only the shape contour we are trying to identify
                        # careful with the changes in coordinates of the roi
                        roi_mask_list, roi_mask_filled_list, roi_img, roi_img_color, x_roi, y_roi, h_roi, w_roi, final_contours = self.crop_roi(contour, x_rect, y_rect, w_rect, h_rect)
                        
                        for j, (contour, roi_mask, roi_mask_filled) in enumerate(zip(final_contours, roi_mask_list, roi_mask_filled_list)):
     

                            circles, img1, img2, img3 = HoughTransform.apply_hough_circles_with_kmeans(roi_mask, roi_mask_filled, no_droplets, contour, contour_area, roi_img_color, isOnEdge, w_roi, h_roi)
                            
        
                            if self.save_image_steps:
                                roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
                                cv2.imwrite(os.path.join(config.RESULTS_LATEX_PIP_ROI_DIR, filename + "_" + str(i) + "roi.png"), roi_img_color)
                                cv2.imwrite(os.path.join(config.RESULTS_LATEX_PIP_ROI_DIR, filename + "_" + str(i) + "_" + str(j) + "roi_mask.png"), roi_mask_filled)

                                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                                img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

                                cv2.imwrite(os.path.join(config.RESULTS_LATEX_PIP_ROI_DIR, filename + "_" + str(i) + "roi_hough1.png"), img1)
                                cv2.imwrite(os.path.join(config.RESULTS_LATEX_PIP_ROI_DIR, filename + "_" + str(i) + "roi_hough2.png"), img2)
                                cv2.imwrite(os.path.join(config.RESULTS_LATEX_PIP_ROI_DIR, filename + "_" + str(i) + "_" + str(j) + "roi_hough3.png"), img3)


                            if circles is not None and len(circles) > 1:
                                self.save_overlapped_droplets(i, original_contour, circles, x_roi, y_roi, isOnEdge)
                            else:
                                # if no circles are found, it is assumed the contour is an elipse
                                self.save_single_droplet(original_contour, i, center_x, center_y, contour_area, overlapped_ids, 1)
                    
                    case 1:

                        roi_mask_list, roi_mask_filled_list, roi_img, roi_img_color, x_roi, y_roi, h_roi, w_roi, final_contours = self.crop_roi(contour, x_rect, y_rect, w_rect, h_rect)
                        
                        for contour, roi_mask, roi_mask_filled in zip(final_contours, roi_mask_list, roi_mask_filled_list):
                           
                            circles = HoughTransform.apply_hough_circles_with_skeletonization(roi_mask_filled, contour_area, w_roi, h_roi, roi_img)

                            if circles is not None and len(circles) > 1:
                                self.save_overlapped_droplets(i, contour, circles, x_roi, y_roi, isOnEdge)
                            else:
                                # if no circles are found, it is assumed the contour is an elipse
                                self.save_single_droplet(contour, i, center_x, center_y, contour_area, overlapped_ids, 1)
                            

        # # create the masks and calculate values for statistics
        if self.create_masks:
            cv2.imwrite(os.path.join(config.RESULTS_CV_DIR, config.RESULTS_GENERAL_MASK_OV_FOLDER_NAME, filename + '.png'), self.mask_overlapped)
            cv2.imwrite(os.path.join(config.RESULTS_CV_DIR, config.RESULTS_GENERAL_MASK_SIN_FOLDER_NAME, filename + '.png'), self.mask_single)
        

        self.detected_image = cv2.cvtColor(self.detected_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(config.RESULTS_CV_DIR, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME, filename + ".png"), self.detected_image)

        if self.save_image_steps:
            #self.detected_image = cv2.cvtColor(self.detected_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(config.RESULTS_LATEX_PIP_DIR, filename + "detected.png"), self.detected_image)
            
            self.separate_image = cv2.cvtColor(self.separate_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(config.RESULTS_LATEX_PIP_DIR, filename + "contours.png"), self.separate_image)

        #cv2.imwrite(os.path.join(config.RESULTS_CV_DROPLETCLASSIFICATION_DIR, filename + "_countour.png"), self.contour_image)

        self.calculate_stats(real_width)

    def crop_roi(self, contour, x, y, w, h):
        full_image_mask = np.zeros_like(self.image_color)
        cv2.drawContours(full_image_mask, [contour], -1, 255, cv2.FILLED)
        
        # calculate the coordinates of the roi image to cut from the original image with a small border around the contour
        pixel_border_expand = config.BORDER_EXPAND
        x_border = max(0, x - pixel_border_expand)
        y_border = max(0, y - pixel_border_expand)
        w_border = min(self.width - x_border, w + 2*pixel_border_expand)
        h_border = min(self.height - y_border, h + 2*pixel_border_expand)

        object_roi_mask = full_image_mask[y_border:y_border + h_border, x_border:x_border + w_border]
        object_roi_mask = cv2.cvtColor(object_roi_mask, cv2.COLOR_BGR2GRAY)
        
        # create an roi with only the shape we are interested in to apply a harsher threshold
        object_roi_img_new_contours_color = self.image_color[y_border:y_border + h_border, x_border:x_border + w_border]
        object_roi_img_new_contours = self.image_gray[y_border:y_border + h_border, x_border:x_border + w_border]
        _, threshold = cv2.threshold(object_roi_img_new_contours, 150, 255, cv2.THRESH_BINARY)
        inverted_threshold = cv2.bitwise_not(threshold)
        overlapped_contours, _ = cv2.findContours(inverted_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        object_roi_mask_new_contours = np.zeros_like(object_roi_img_new_contours)
        cv2.drawContours(object_roi_mask_new_contours, overlapped_contours, -1, 255, 1)
        object_roi_mask_new_contours = (cv2.bitwise_and(object_roi_mask_new_contours, object_roi_mask_new_contours, mask = object_roi_mask))

        object_roi_mask_new_contour_filled = np.zeros_like(object_roi_img_new_contours)
        cv2.drawContours(object_roi_mask_new_contour_filled, overlapped_contours, -1, 255, cv2.FILLED)
        object_roi_mask_new_contour_filled = (cv2.bitwise_and(object_roi_mask_new_contour_filled, object_roi_mask_new_contour_filled, mask = object_roi_mask))
        
        # get the final list of contours in case some of the droplets were separated during this process
        #inverted_object_roi = cv2.bitwise_not(object_roi_mask_new_contours_filled)
        final_contours, _ = cv2.findContours(object_roi_mask_new_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        list_of_masks_filled = []
        list_of_masks_contour = []
    
        for contour in final_contours:
            aux_max1 = np.zeros_like(object_roi_img_new_contours)
            aux_max2 = np.zeros_like(object_roi_img_new_contours)

            cv2.drawContours(aux_max1, [contour], -1, 255, cv2.FILLED)
            cv2.drawContours(aux_max2, [contour], -1, 255, 1)
            
            list_of_masks_filled.append(aux_max1)
            list_of_masks_contour.append(aux_max2)

          
        #Util.plotTwoImages(object_roi_mask, object_roi_mask_new_contours)
        return list_of_masks_contour, list_of_masks_filled, object_roi_img_new_contours, object_roi_img_new_contours_color, x_border, y_border, h_border, w_border, final_contours


    def save_overlapped_droplets(self, index, contour, circles, x_roi, y_roi, isOnEdge):   
        circle_ids = list(range(index, index + len(circles)))
        j = 0

        if isOnEdge: cv2.drawContours(self.separate_image, [contour], -1, detected_colors[3], 2)
        else: cv2.drawContours(self.separate_image, [contour], -1, detected_colors[2], 2)
        
        # save the ids from all the circles in the list that do not include its own index
        for circle in circles:
            overlapped_ids = []
            overlapped_ids = circle_ids[:j] + circle_ids[j+1:]

            radius = int(circle[2])
            center_x = int(circle[0] + x_roi)
            center_y = int(circle[1] + y_roi)
            circle_area = np.pi * radius ** 2

            self.droplets_data.append(Droplet(center_x, center_y, circle_area, int(index + j), overlapped_ids, radius))
            cv2.circle(self.detected_image, (center_x, center_y), radius, detected_colors[2], 1)
            if self.create_masks: cv2.circle(self.mask_overlapped, (center_x, center_y), radius, 255, cv2.FILLED)
           
            self.final_no_droplets += 1        
            
            j+=1

    def generate_circle_points(radius, center, num_points=100):
        angles = np.linspace(0, 2 * np.pi, num_points)
        
        points = []
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            points.append((x, y))
        
        return points

    def save_single_droplet(self, contour, id, center_x, center_y, area, overlappedIDs, shape):
       
        self.droplets_data.append(Droplet(center_x, center_y, area, id, overlappedIDs))
        cv2.drawContours(self.detected_image, [contour], -1, detected_colors[shape], 1)
        if self.create_masks: cv2.drawContours(self.mask_single, [contour], -1, 255, cv2.FILLED)

        cv2.drawContours(self.separate_image, [contour], -1, detected_colors[shape], 2)

        self.droplet_shapes[id] = contour
        
        self.final_no_droplets += 1
        

    

    def check_for_contour_shape(self, contour, contour_area, x_rect, y_rect, w_rect, h_rect):
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        is_single = cv2.isContourConvex(approx)
        
        no_droplets = 1
        isOnEdge = False

        # if the contour touches the sides, hough circles can be used to find the center of the shape even if it is outside the image
        if x_rect < 2 or y_rect < 2 or x_rect + w_rect > self.width - 2 or y_rect + h_rect > self.height - 2:
            isOnEdge = True
    
        # check first if the contour is convex (overlapping of droplets) or non convex (single droplet)
        # to eliminate unnecessary calculations
        if not is_single:

            # find the number of convex points that should indicate the number of droplets in that contour
            # if the number of convex points is 1, then we can assume its only one droplet
            no_droplets = self.find_number_of_droplets(contour)

            # only return here if there is more than 1 droplet
            if no_droplets > 1:
                shape = 2
                return shape, no_droplets, isOnEdge
            
            if no_droplets == 1: no_droplets = 2
            if no_droplets == 0: no_droplets = 1

        # check for circularity and ratio of the axes to determine if it is a circle or an elipse
        contour_perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (contour_area / (contour_perimeter ** 2))

        if len(contour) < 5:
            shape = 0
            return shape, no_droplets, isOnEdge

        (_, (major, minor), _) = cv2.fitEllipse(contour)
        elipse_aspect_ratio = major / minor

        if circularity > config.CIRCULARITY_THRESHOLD:
            shape = 0
        elif elipse_aspect_ratio > config.ELIPSE_THRESHOLD:
            shape = 1
        else:
            shape = 2
        
        return shape, no_droplets, isOnEdge
        

    def find_number_of_droplets(self, contour):
        hull = cv2.convexHull(contour, returnPoints = False)

        # convexity defect is a deviation from the convex hull shape
        # it returns (  start_index, 
        #               end_index, 
        #               farthest point from hull, 
        #               fixed distance from farthest point to the hull)
        # to get the value of the depth, divide the distance by 256.0.
        defects = cv2.convexityDefects(contour, hull)
        no_convex_points = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                _, _, f, d = defects[i, 0]
                far = tuple(contour[f][0])
                depth = d / 256.0 

                # 1 is the value that was found to best threshold
                if depth > 1:
                    no_convex_points +=1

        # if the number of convex point is not even, we assume the next value to be the number of circles
        if (no_convex_points % 2 == 1):
            no_convex_points += 1

        if no_convex_points == 0:
            no_convex_points = 1

        return no_convex_points


    def initialize_variables(self, image_gray, image_color):
        # save each step in a different image
        self.roi_image = copy.copy(image_gray)
        self.roi_image_color = copy.copy(image_color)
        self.hull_image = copy.copy(image_color)
        self.separate_image = copy.copy(image_color)
        self.image_gray = copy.copy(image_gray)
        self.image_color = copy.copy(image_color)
        self.detected_image = copy.copy(image_color)

        self.height, self.width, _ = image_color.shape

        # create masks
        if self.create_masks:
            mask = np.zeros_like(self.image_color)
            self.mask_overlapped = copy.copy(mask)
            self.mask_single = copy.copy(mask)

        # initialize variables
        self.droplets_data:list[Droplet]=[]
        self.contour_area = 0
        self.droplet_shapes = {}

        self.final_no_droplets = 0

    def create_mask(self, category, contour):
        if category == 2:
            if self.create_masks: cv2.drawContours(self.mask_overlapped, [contour], -1, 255, thickness=cv2.FILLED)
        else:
            if self.create_masks: cv2.drawContours(self.mask_single, [contour], -1, 255, thickness=cv2.FILLED)

    def get_contours(self):        
        img = copy.copy(self.image_gray)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # findContours detects the white part of the image, therefore the threshold has to be inverted
        # values in the threshold dont affect anything?
        _, threshold = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
        if self.save_image_steps:
            cv2.imwrite(os.path.join(config.RESULTS_LATEX_PIP_DIR, self.filename + "threshold.png"), threshold)
            thre = cv2.imread(os.path.join(config.RESULTS_LATEX_PIP_DIR, self.filename + "threshold.png"))
            self.separate_image = copy.copy(thre)
            self.detected_image = copy.copy(thre)

        inverted_threshold = cv2.bitwise_not(threshold)
        self.contours, _ = cv2.findContours(inverted_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    def calculate_stats(self, real_width):
        self.droplet_area = [d.area for d in self.droplets_data]

        self.volume_list = sorted(Statistics.area_to_volume(self.droplet_area, self.width, real_width))

        vmd_value, coverage_percentage, rsf_value, cumulative_volume = Statistics.calculate_statistics(self.volume_list, self.image_color, self.contour_area)
        self.stats = Statistics(vmd_value, rsf_value, coverage_percentage, self.final_no_droplets, self.droplets_data)
