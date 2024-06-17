import cv2
import numpy as np
import copy
import os
import config
import Util

from Droplet import Droplet
from Statistics import Statistics
from HoughTransform import HoughTransform


class Segmentation:
    def __init__(self, image_color, image_gray, filename, save_images:bool, create_masks:bool):
        self.save_images = save_images
        self.create_masks = create_masks

        self.initialize_variables(image_gray, image_color)
       
        # sort contours to catch the biggest ones first and remove the smaller ones inside before
        self.get_contours()
        self.contours = sorted(self.contours, key=lambda c: cv2.arcLength(c, True), reverse=True)
        len_contours_original = len(self.contours)

        i=0
        while(i < len(self.contours)):         
            contour = self.contours[i]
            contour_area = cv2.contourArea(contour)
            self.contour_area += contour_area
            overlapped_ids = []

            # treat small contours like a perfect circle
            if len(contour) < 5 and contour_area < 3 and i < len_contours_original: 
                self.save_draw_circle_droplet(contour, overlapped_ids, i, (200, 200, 200), 0)
                i += 1
                continue    

            elif len(contour) < 5 and i < len_contours_original:
                self.save_draw_circle_droplet(contour, overlapped_ids, i, (160, 160, 160), 0)
                i += 1
                continue
           
            # when contour is not closed there is a chance of wrongly detected contours inside
            if contour_area < cv2.arcLength(contour, True): 
                check, contour = self.remove_inside_contours(contour, i)
                
                if check == 0:
                    i += 1
                    continue
              
                # treat small contours like a perfect circle
                if len(contour) < 5: 
                    self.save_draw_circle_droplet(contour, overlapped_ids, i, (255, 0, 51), 0)
                    i += 1
                    continue    

           
            # check the shape to categorize it into 0: single, 1: elipse, 2: overlapped
            shape, no_convex_points = self.check_for_shape(contour)

            # not a real contour of a droplet so we skip or error of contour inside contour
            if (no_convex_points == -1) or (no_convex_points == -2):
                i += 1
                continue
            
            match shape:
                case 0: # circle single
                    self.save_draw_circle_droplet(contour, overlapped_ids, i, (255, 255, 255), 0)
                
                case 1: # elipse
                    self.save_draw_elipse_droplet(contour, i, overlapped_ids, (102, 0, 102), 1)
                
                case 2: # circles overlapped

                    # get ROI of the contour to analyze
                    roi_mask, roi_img, x_roi, y_roi, _, _ = self.crop_ROI(contour, config.BORDER_EXPAND)
               
                    # detect circles and only return the main ones by clustering
                    circles = HoughTransform(roi_mask, no_convex_points, contour, contour_area, roi_img).circles
            
                    # save each one of the overlapped circles
                    if circles is not None:                    
                        self.save_draw_overlapped_droplet(circles, x_roi, y_roi, i, contour)
                        

                    else:
                        self.save_draw_elipse_droplet(contour, i, overlapped_ids, (0, 102, 51), 1)
                                    
            i += 1

        # create the masks and calculate values for statistics
        if self.create_masks:
            cv2.imwrite(os.path.join(config.RESULTS_CV_MASK_OV_DIR, filename + '.png'), self.mask_overlapped)
            cv2.imwrite(os.path.join(config.RESULTS_CV_MASK_SIN_DIR, filename + '.png'), self.mask_single)

        cv2.imwrite(os.path.join(config.RESULTS_CV_DROPLETCLASSIFICATION_DIR, filename + ".png"), self.detected_image)
        cv2.imwrite(os.path.join(config.RESULTS_CV_DROPLETCLASSIFICATION_DIR, filename + "_countour.png"), self.contour_image)

        self.calculate_stats()

    def initialize_variables(self, image_gray, image_color):
        # save each step in a different image
        self.roi_image = copy.copy(image_gray)
        self.roi_image_color = copy.copy(image_color)
        self.contour_image = copy.copy(image_color)
        self.detected_image = copy.copy(image_color)
        self.hull_image = copy.copy(image_color)
        self.separate_image = copy.copy(image_color)
        self.image = copy.copy(image_gray)
        self.color_image = copy.copy(image_color)

        self.height, self.width, _ = image_color.shape


        
        # create masks
        if self.create_masks:
            mask = np.zeros_like(self.image)
            self.mask_overlapped = copy.copy(mask)
            self.mask_single = copy.copy(mask)

        # initialize variables
        self.droplets_data:list[Droplet]=[]
        self.contour_area = 0

        self.ignore_ids = []
    def is_circle_within_image(self, height, width, x, y, radius):
        # check if the circle is within the image boundaries
        if (x - radius >= 0) and (x + radius <= width) and (y - radius >= 0) and (y + radius <= height):
            return True
        return False
    
    def save_draw_overlapped_droplet(self, circles, x_roi, y_roi, i, contour):
        if self.create_masks: self.create_mask(2, contour) 
        circle_ids = list(range(i, i + len(circles)))
        j = 0
        for circle in circles:
            overlapped_ids = []
            overlapped_ids = circle_ids[:j] + circle_ids[j+1:]
            
            self.droplets_data.append(Droplet(True, int(circle[0] + x_roi), int(circle[1] + y_roi), float(circle[2]*2), int(i + j), overlapped_ids))
            cv2.circle(self.detected_image, (int(circle[0] + x_roi), int(circle[1] + y_roi)), int(circle[2]), (0, 255, 0), 2)
            j+=1
            
    def save_draw_circle_droplet(self, contour, overlapped_ids, i, color_array, shape):
        if self.create_masks: self.create_mask(shape, contour) 
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)

        if (self.is_circle_within_image(self.height, self.width, center_x, center_y, radius)):
            #diameter = 0.95*(np.sqrt((4*contour_area)/np.pi))**0.91
            self.droplets_data.append(Droplet(False, int(center_x), int(center_y), float(radius * 2), int(i), overlapped_ids))
            cv2.circle(self.detected_image, (int(center_x), int(center_y)), int(radius), color_array, 2)
        


    def save_draw_elipse_droplet(self, contour, i, overlapped_ids, color_array, shape):
        if self.create_masks: self.create_mask(shape, contour)  
        (x, y), (major, minor), angle = cv2.fitEllipse(contour)
        elipse = cv2.fitEllipse(contour)
        self.droplets_data.append(Droplet(True, int(x), int(y) , float(minor), int(i), overlapped_ids))
        cv2.ellipse(self.detected_image, elipse, color = color_array, thickness=2)
       
        
    def remove_inside_contours(self, contour, index):
        # get new contour with a new threshold
        for i in range(3, config.BORDER_EXPAND):
            new_contours, object_roi_img, x, y, w, h = self.get_new_contours(contour, i)
            if len(new_contours) > 1:
                contour_aux = cv2.convexHull(new_contours[0])
                area = cv2.contourArea(contour_aux)
                
                if len(new_contours) > 0 and area > 0.4 * w * h:
                    break

        
        if len(new_contours) < 1:
            return 0, new_contours

        new_contour = new_contours[0]
        new_contour =cv2.convexHull(new_contour)

        shifted_contour = copy.copy(new_contour)
        shifted_contour[:, :, 0] += x
        shifted_contour[:, :, 1] += y

        #shifted_contour = np.array([(point[0][0] + x, point[0][1] + y) for point in new_contour])
       
        cv2.drawContours(object_roi_img, [new_contour], -1, (255, 0, 0), thickness=1)
        cv2.drawContours(self.contour_image, [shifted_contour], -1, (255, 0, 255), thickness=2)
        
        # plt.imshow(object_roi_img)
        # plt.show()

        # plt.imshow(self.detected_image)
        # plt.show()
        
        # remove previous contours in that section of the image
        self.contours = [contour for contour in self.contours if not self.is_contour_in_roi(contour, x, y, h, w)]
        #TODO check if correct
        self.contours.append(shifted_contour)

        return 1, shifted_contour
    
    def get_new_contours(self, contour, pixel_border_expand):
        _, object_roi_img, x, y, h, w = self.crop_ROI(contour, pixel_border_expand)
        middle_pixel = object_roi_img[h // 2, w // 2]
        # assume middle pixel is of interest
        threshold = 50
        lower_bound = np.array([middle_pixel[0] - threshold, middle_pixel[1] - threshold, middle_pixel[2] - threshold])
        upper_bound = np.array([middle_pixel[0] + threshold, middle_pixel[1] + threshold, middle_pixel[2] + threshold])
        mask = cv2.inRange(object_roi_img, lower_bound, upper_bound)
        result = cv2.bitwise_and(object_roi_img, object_roi_img, mask=mask)
        
        # plt.imshow(result)
        # plt.show()

        img = copy.copy(object_roi_img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        # _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        edges = cv2.Canny(result, 170, 200)
        new_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_contours = sorted(new_contours, key=cv2.contourArea, reverse=True)

        cv2.drawContours(img, new_contours, -1, (0, 0, 255), thickness=1)
        
        # plt.imshow(img)
        # plt.show()
        
        return new_contours, object_roi_img, x, y, w, h

    def is_contour_in_roi(self, contour, x, y, h, w):
        for point in contour:
            px, py = point[0]

            if (x <= px <= x + w and y <= py <= y + h):
                return True

        return False
    
    def find_number_of_circles(self, contour):
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

            return no_convex_points
        
        except Exception as e:
            print(e) 
            return -2

    def create_mask(self, category, contour):
        if category == 2:
            cv2.drawContours(self.mask_overlapped, [contour], -1, 255, thickness=cv2.FILLED)
        else:
            cv2.drawContours(self.mask_single, [contour], -1, 255, thickness=cv2.FILLED)

    def get_contours(self):        
    
        img = copy.copy(self.image)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)   
        edges = cv2.Canny(threshold, 170, 200)
       
        self.contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.contour_image, self.contours, -1, (255, 255, 255), 1)

        # save number of contours
        self.final_no_droplets = len(self.contours)
    
    def crop_ROI(self, contour, pixel_border_expand):
        output_image = np.zeros_like(self.roi_image)
        cv2.drawContours(output_image, [contour], -1, 255, 1)

        x, y, w, h = cv2.boundingRect(contour)

        x_border = max(0, x - pixel_border_expand)
        y_border = max(0, y - pixel_border_expand)
        w_border = min(self.contour_image.shape[1] - x_border, w + 2 * pixel_border_expand)
        h_border = min(self.contour_image.shape[0] - y_border, h + 2 * pixel_border_expand)

        object_roi_mask = output_image[y_border:y_border + h_border, x_border:x_border + w_border]
        object_roi_img = self.roi_image_color[y_border:y_border + h_border, x_border:x_border + w_border]
        object_roi_img = cv2.cvtColor(object_roi_img, cv2.COLOR_RGB2BGR)
              
        return object_roi_mask, object_roi_img, x_border, y_border, h_border, w_border

    def check_for_shape(self, contour):
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        is_single = cv2.isContourConvex(approx)
        
        # if the contour is convex, we assume there is more than one droplet
        if not is_single:
            # find how many circles are in the shape
            # if the no
            no_convex_points = self.find_number_of_circles(contour)
        else:
            # try to distinguish between elipse and circle, based on axes ratio, circularity and area
            no_convex_points = 0
        
        #TODO makes this better
        ellipse = cv2.fitEllipse(contour)
        (_, axes, _) = ellipse

        aspect_ratio = axes[0] / axes[1]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        area_elipse = np.pi * axes[0]/2 * axes[1]/2
        
        # categorize the contour given the values from before
        if aspect_ratio < 0.25:
            return 0, -1
        
        elif no_convex_points == 0: 
            if (aspect_ratio > config.ELIPSE_THRESHOLD and circularity > config.CIRCULARITY_THRESHOLD):
                cv2.drawContours(self.separate_image, [contour], -1, (102, 0, 204), 2)
                shape = 0   # circle
            else:
                cv2.drawContours(self.separate_image, [contour], -1, (204, 51, 153), 2)
                shape = 1   # elipse
        else:
            cv2.drawContours(self.separate_image, [contour], -1, (44, 156, 63), 2)
            shape = 2       # overlapped
        
        return shape, no_convex_points
       
    def calculate_stats(self):
        self.droplet_diameter = [d.diameter for d in self.droplets_data]

        self.volume_list = sorted(Statistics.diameter_to_volume(self.droplet_diameter, self.width))

        cumulative_fraction = Statistics.calculate_cumulative_fraction(self.volume_list)
        vmd_value = Statistics.calculate_vmd(cumulative_fraction, self.volume_list)
        rsf_value = Statistics.calculate_rsf(cumulative_fraction, self.volume_list, vmd_value)
        coverage_percentage = Statistics.calculate_coverage_percentage_c(self.image, self.height, self.width, self.contour_area)

        self.stats = Statistics(vmd_value, rsf_value, coverage_percentage, self.final_no_droplets, self.droplets_data)
