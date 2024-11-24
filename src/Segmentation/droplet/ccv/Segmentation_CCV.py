import cv2
import numpy as np
import copy
import os
import sys
from matplotlib import pyplot as plt 

sys.path.insert(0, 'src')
import Common.config as config
from Common.Droplet import Droplet
from Common.Statistics import Statistics
import Segmentation.droplet.ccv.HoughTransform as hough_transform


circle_color = (255, 0, 0)
elipse_color = (0, 255, 255)
overlapped_color = (0, 255, 0)
edge_color = (0, 0, 255)
detected_colors = [circle_color, elipse_color, overlapped_color, edge_color]


class Segmentation_CCV:
    def __init__(self, image_color, image_gray, filename, save_image_steps:bool, segmentation_method:int, dataset_results_folder):
        self.save_image_steps = save_image_steps
        self.filename = filename
        self.dataset_results_folder = dataset_results_folder
    
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

            if shape < 2:   # single circle or single elipse
                self.save_single_droplet(contour, i, center_x, center_y, contour_area, overlapped_ids, shape)
                continue
            
            else:  # overlapping droplets and on the edge
 
                match segmentation_method:
                    case 0: # hough circles with clustering
                        original_contour = copy.copy(contour)

                        # for hough, create a roi mask with only the shape contour we are trying to identify
                        # careful with the changes in coordinates of the roi     
                        object_roi_mask_filled, object_roi_mask_edges, object_roi_color, object_roi_gray, x_roi, y_roi, h_roi, w_roi = self.crop_roi(contour, x_rect, y_rect, w_rect, h_rect)
                     
                        circles, img1, img2, img3 = hough_transform.apply_hough_circles_with_kmeans(object_roi_mask_filled, object_roi_mask_edges, no_droplets, contour, contour_area, object_roi_color, isOnEdge, w_roi, h_roi)
            
                        if self.save_image_steps:
                            roi_img = cv2.cvtColor(object_roi_color, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(os.path.join(config.RESULTS_LATEX_PIP_ROI_DIR, filename + "_" + str(i) + "roi.png"), roi_img)
                            
                            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

                            cv2.imwrite(os.path.join(config.RESULTS_LATEX_PIP_ROI_DIR, filename + "_" + str(i) + "roi_hough1.png"), img1)
                            cv2.imwrite(os.path.join(config.RESULTS_LATEX_PIP_ROI_DIR, filename + "_" + str(i) + "roi_hough2.png"), img2)
                            cv2.imwrite(os.path.join(config.RESULTS_LATEX_PIP_ROI_DIR, filename + "_" + str(i) + "roi_hough3.png"), img3)


                        if circles is not None and len(circles) > 1:
                            self.save_overlapped_droplets(i, original_contour, circles, x_roi, y_roi, isOnEdge)
                        else:
                            # if no circles are found, it is assumed the contour is an elipse
                            self.save_single_droplet(original_contour, i, center_x, center_y, contour_area, overlapped_ids, 1)
                
                    case 1:

                        roi_mask_list, roi_mask_filled_list, roi_img, roi_img_color, x_roi, y_roi, h_roi, w_roi, final_contours = self.crop_roi(contour, x_rect, y_rect, w_rect, h_rect)
                        
                        for contour, roi_mask, roi_mask_filled in zip(final_contours, roi_mask_list, roi_mask_filled_list):
                           
                            circles = hough_transform.apply_hough_circles_with_skeletonization(roi_mask_filled, contour_area, w_roi, h_roi, roi_img)

                            if circles is not None and len(circles) > 1:
                                self.save_overlapped_droplets(i, contour, circles, x_roi, y_roi, isOnEdge)
                            else:
                                # if no circles are found, it is assumed the contour is an elipse
                                self.save_single_droplet(contour, i, center_x, center_y, contour_area, overlapped_ids, 1)
                            



        self.detected_image = cv2.cvtColor(self.detected_image, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(os.path.join(dataset_results_folder, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME, filename + ".png"), self.detected_image)

        if self.save_image_steps:
            #self.detected_image = cv2.cvtColor(self.detected_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(config.RESULTS_LATEX_PIP_DIR, filename + "detected.png"), self.detected_image)
            
        self.separate_image = cv2.cvtColor(self.separate_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dataset_results_folder, filename + "contours.png"), self.separate_image)

        #cv2.imwrite(os.path.join(config.RESULTS_CV_DROPLETCLASSIFICATION_DIR, filename + "_countour.png"), self.contour_image)

    def crop_roi(self, contour, x, y, w, h):
        filled_image_mask = np.zeros_like(self.image_color)
        edges_image_mask = np.zeros_like(self.image_color)
        cv2.drawContours(filled_image_mask, [contour], -1, 255, cv2.FILLED)
        cv2.drawContours(edges_image_mask, [contour], -1, 255, 1)
        
        # calculate the coordinates of the roi image to cut from the original image with a small border around the contour
        pixel_border_expand = config.BORDER_EXPAND
        x_border = max(0, x - pixel_border_expand)
        y_border = max(0, y - pixel_border_expand)
        w_border = min(self.width - x_border, w + 2*pixel_border_expand)
        h_border = min(self.height - y_border, h + 2*pixel_border_expand)

        object_roi_mask_filled = filled_image_mask[y_border:y_border + h_border, x_border:x_border + w_border]
        object_roi_mask_filled = cv2.bitwise_not(object_roi_mask_filled)
        object_roi_mask_filled = cv2.cvtColor(object_roi_mask_filled, cv2.COLOR_BGR2GRAY)

        object_roi_mask_edges = edges_image_mask[y_border:y_border + h_border, x_border:x_border + w_border]
        object_roi_mask_edges = cv2.bitwise_not(object_roi_mask_edges)
        object_roi_mask_edges = cv2.cvtColor(object_roi_mask_edges, cv2.COLOR_BGR2GRAY)

        object_roi_color = self.image_color[y_border:y_border + h_border, x_border:x_border + w_border]
        object_roi_gray = self.image_gray[y_border:y_border + h_border, x_border:x_border + w_border]

        return object_roi_mask_filled, object_roi_mask_edges, object_roi_color, object_roi_gray, x_border, y_border, h_border, w_border

    def save_overlapped_droplets(self, index, contour, circles, x_roi, y_roi, isOnEdge):   
        circle_ids = list(range(index, index + len(circles)))
        j = 0

        if isOnEdge: cv2.drawContours(self.separate_image, [contour], -1, detected_colors[3], 1)
        else: cv2.drawContours(self.separate_image, [contour], -1, detected_colors[2], 1)
        
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
        
        cv2.drawContours(self.separate_image, [contour], -1, detected_colors[shape], 1)

        self.droplet_shapes[id] = contour
        
        self.final_no_droplets += 1
        

    def check_for_contour_shape(self, contour, contour_area, x_rect, y_rect, w_rect, h_rect):
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        is_single = cv2.isContourConvex(approx)
        
        no_droplets = -1
        shape = -1
        isOnEdge = False

        # check first if the contour is convex (overlapping of droplets) or non convex (single droplet)
        # to eliminate unnecessary calculations
        if not is_single:
            shape = 2

            # if the contour touches the sides, hough circles can be used to find the center of the shape 
            # even if it is outside the image
            if x_rect < 2 or y_rect < 2 or x_rect + w_rect > self.width - 2 or y_rect + h_rect > self.height - 2:
                isOnEdge = True
                
            # find the number of convex points that should indicate the number of droplets in that contour
            # if the number of convex points is 1, then we can assume its only one droplet
            no_droplets = self.find_number_of_droplets(contour)

            # only return here if there is more than 1 droplet
            # add an additional circle to better the chances of finding the correct one
            no_droplets += 1
            
        # check for circularity and ratio of the axes to determine if it is a circle or an elipse
        contour_perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (contour_area / (contour_perimeter ** 2))

        (_, (major, minor), _) = cv2.fitEllipse(contour)
        elipse_aspect_ratio = major / minor

        if shape != 2 and circularity > config.CIRCULARITY_THRESHOLD:
            shape, no_droplets = 0, 1
        elif shape != 2 and elipse_aspect_ratio > config.ELIPSE_THRESHOLD:
            shape, no_droplets = 1, 1
        elif shape == 2 and circularity > config.CIRCULARITY_THRESHOLD + 10:
            shape, no_droplets = 0, 1
        else: shape = 2
        
        if no_droplets < 1: no_droplets = 1
        
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


        # initialize variables
        self.droplets_data:list[Droplet]=[]
        self.contour_area = 0
        self.droplet_shapes = {}

        self.final_no_droplets = 0

    def get_contours(self):        
        img = copy.copy(self.image_gray)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # findContours detects the white part of the image, therefore the threshold has to be inverted
        # values in the threshold dont affect anything?
        _, threshold = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
        #if self.save_image_steps:
        cv2.imwrite(os.path.join(self.dataset_results_folder, self.filename + "threshold.png"), threshold)
            # thre = cv2.imread(os.path.join(config.RESULTS_LATEX_PIP_DIR, self.filename + "threshold.png"))
            # self.separate_image = copy.copy(thre)
            # self.detected_image = copy.copy(thre)

        inverted_threshold = cv2.bitwise_not(threshold)
        self.contours, _ = cv2.findContours(inverted_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
# path = "data\\droplets\\synthetic_dataset_normal_droplets\\raw\\image\\0.png"
# # read image
# image_gray = cv2.imread(path ,cv2.IMREAD_GRAYSCALE)
# image_colors = cv2.imread(path)  

# image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)

# # calculate statistics
# calculated = Segmentation_CV(image_colors, image_gray, "0", False, True, 0, "results\\computer_vision_algorithm")
