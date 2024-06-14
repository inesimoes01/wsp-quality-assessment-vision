import cv2
import numpy as np
import copy
import os
import config
import Util

from Droplet import Droplet
from Statistics import Statistics
from Distortion import Distortion

from HoughTransform import HoughTransform

#TODO better overlapping count
#TODO better coverage
#TODO remove contour inside contour in real images
#TODO quando uso

class Calculated_Statistics:
    def __init__(self, color_image, filename, save_images:bool, create_masks:bool):
        self.save_images = save_images
        self.create_masks = create_masks

        # save each step in a different image
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        self.roi_image = copy.copy(gray)
        self.roi_image_color = copy.copy(color_image)
        self.contour_image = copy.copy(color_image)
        self.detected_image = copy.copy(color_image)
        self.hull_image = copy.copy(color_image)
        self.separate_image = copy.copy(color_image)

        height, width, _ = color_image.shape
        image_area = height * width

        # get objects from image
        self.image = copy.copy(gray)
        self.color_image = copy.copy(color_image)
        self.get_contours()

        # create masks
        if self.create_masks:
            mask = np.zeros_like(self.image)
            self.mask_overlapped = copy.copy(mask)
            self.mask_single = copy.copy(mask)

        # initialize variables
        self.droplets_data:list[Droplet]=[]
        self.contour_area = 0
        i=0
        
        # sort contours to catch the biggest ones first and remove the smaller ones inside before
        self.contours = sorted(self.contours, key=lambda c: cv2.arcLength(c, True), reverse=True)
        self.ignore_ids = []

        len_contours_original = len(self.contours)

        while(i < len(self.contours)):
            if i in self.ignore_ids: 
                if self.create_masks: self.create_mask(shape, contour)  
                self.save_draw_circle_droplet(contour, overlapped_ids, i, contour_area, (0, 0, 0), height, width)
                  
                i += 1
                continue
         
            contour = self.contours[i]
            contour_area = cv2.contourArea(contour)
            self.contour_area += contour_area
            overlapped_ids = []

            # treat small contours like a perfect circle
            if len(contour) < 5 and contour_area < 3 and i < len_contours_original: 
                if self.create_masks: self.create_mask(shape, contour)  
                self.save_draw_circle_droplet(contour, overlapped_ids, i, contour_area, (200, 200, 200), height, width)
                i += 1
                continue    

            elif len(contour) < 5 and i < len_contours_original:
                if self.create_masks: self.create_mask(shape, contour)  
                self.save_draw_circle_droplet(contour, overlapped_ids, i, contour_area, (160, 160, 160), height, width)
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
                    if self.create_masks: self.create_mask(shape, contour)  
                    self.save_draw_circle_droplet(contour, overlapped_ids, i, contour_area, (255, 0, 51), height, width)
                  
                    i += 1
                    continue    

           
            # check the shape to categorize it into 0: single, 1: elipse, 2: overlapped
            shape, no_convex_points = self.check_for_shape(contour, width, height)

            # not a real contour of a droplet so we skip
            if (no_convex_points == -1):

                roi_mask, roi_img, x_roi, y_roi, _, _ = self.crop_ROI(contour, config.BORDER_EXPAND)
     
                i += 1
                continue
            
            # error of contour inside contour
            if (no_convex_points == -2):

                #TODO add treatment
                i += 1
                continue
            
            match shape:
                # circle single
                case 0:
                    self.save_draw_circle_droplet(contour, overlapped_ids, i, contour_area, (255, 255, 255), height, width)
                # elipse
                case 1:
                    self.save_draw_elipse_droplet(contour, i, overlapped_ids, (102, 0, 102))
                    
                # circles overlapped
                case 2:
                    # get ROI of the contour to analyze
                    roi_mask, roi_img, x_roi, y_roi, _, _ = self.crop_ROI(contour, config.BORDER_EXPAND)
               
                    # detect circles and only return the main ones by clustering
                    circles = HoughTransform(roi_mask, no_convex_points, contour, contour_area, roi_img).circles
            
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
                            cv2.circle(roi_img, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 255, 0), 2)
                    else:
                        self.save_draw_elipse_droplet(contour, i, overlapped_ids, (0, 102, 51))
                                    
            
            if self.create_masks: self.create_mask(shape, contour)  


            i += 1

        # create the masks and calculate values for statistics
        if self.create_masks:
            cv2.imwrite(os.path.join(config.RESULTS_CV_MASK_OV_DIR, filename + '.png'), self.mask_overlapped)
            cv2.imwrite(os.path.join(config.RESULTS_CV_MASK_SIN_DIR, filename + '.png'), self.mask_single)

        cv2.imwrite(os.path.join(config.RESULTS_CV_DROPLETCLASSIFICATION_DIR, filename + ".png"), self.detected_image)
        cv2.imwrite(os.path.join(config.RESULTS_CV_DROPLETCLASSIFICATION_DIR, filename + "_countour.png"), self.contour_image)
        cv2.imwrite(os.path.join(config.RESULTS_CV_DROPLETCLASSIFICATION_DIR, filename + "_canny.png"), self.canny) 

        self.calculate_stats()
    
    def is_circle_within_image(self, height, width, x, y, radius):
        # check if the circle is within the image boundaries
        if (x - radius >= 0) and (x + radius <= width) and (y - radius >= 0) and (y + radius <= height):
            return True
        return False
        
    def save_draw_circle_droplet(self, contour, overlapped_ids, i, contour_area, color_array, height, width):
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)

        if (self.is_circle_within_image(height, width, center_x, center_y, radius)):
            #diameter = 0.95*(np.sqrt((4*contour_area)/np.pi))**0.91
            self.droplets_data.append(Droplet(False, int(center_x), int(center_y), float(radius * 2), int(i), overlapped_ids))
            cv2.circle(self.detected_image, (int(center_x), int(center_y)), int(radius), color_array, 2)

    def save_draw_elipse_droplet(self, contour, i, overlapped_ids, color_array):
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
    
    def normalize_point(self, point):
        point = np.array(point)
        if point.ndim == 1:
            return point
        elif point.ndim == 2:
            return point[0]
        else:
            raise ValueError("Invalid point format")
        
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
        # thesholding
        img = copy.copy(self.image)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # plt.imshow(img)
        # plt.show()

        # th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
        # th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
        # th4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 3)
        th5 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

        
        # #_, otsu_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   
        edges = cv2.Canny(th5, 170, 200)
        # kernel = np.ones((1,1),np.uint8)
        #edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
        # edges = cv2.dilate(edges, kernel)
        # edges = cv2.erode(edges, kernel)
        #plotFourImages(th2, edges, th4, th5)

        self.canny = copy.copy(edges)
       
        self.contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = self.image.shape
        #self.contour_image = np.full((height, width, 3), (0), dtype=np.uint8)
        cv2.drawContours(self.contour_image, self.contours, -1, (255, 255, 255), 1)

        # plt.imshow(self.contour_image)
        # plt.show()

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

    def check_for_shape(self, contour, image_width, image_height):
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