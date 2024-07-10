import cv2
import copy
import numpy as np
import sys
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.morphology import skeletonize
from skimage.draw import disk


sys.path.insert(0, 'src/common')
import config

import Util

show_plots = False


def apply_hough_circles_with_kmeans(roi, roi_filled, no_convex_points, contour, area, roi_img, isOnEdge,  w_roi, h_roi):
    circles = cv2.HoughCircles(roi_filled, 
                                    cv2.HOUGH_GRADIENT, 
                                    dp=1,              # Inverse ratio of accumulator resolution to image resolution. Higher values mean lower resolution/precision but potentially faster processing.
                                    minDist=1,           # Minimum distance between centers of detected circles.
                                    param1=200,          # Higher threshold of Canny edge detector.
                                    param2=5,           # Accumulator threshold for circle centers at the detection stage. Smaller values may lead to more false detections.
                                    minRadius=0,         # Minimum radius of circles to be detected.
                                    maxRadius=int(area))         # Maximum radius of circles to be detected. If negative, it defaults to the maximum image dimension.

    roi_initial_list = copy.copy(roi_img)
    roi_kmeans = copy.copy(roi_img)
    roi_iou_check = copy.copy(roi_img)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = circles[0,:]

        for circle in circles:                                
            r = int(circle[2])
            x = int(circle[0])
            y = int(circle[1])
            cv2.circle(roi_initial_list, (x, y), r, 255, thickness=1)

        if len(circles) > 1:
            circles = remove_circles_based_on_size(circles, w_roi, h_roi)
        
        # kmeans approximation with the final number of circles
        circles = k_means_approximation(circles, no_convex_points, roi_kmeans)

        if len(circles) > 1:
            circles = remove_circles_based_on_iou_contribution(roi_filled, roi_iou_check, circles)
    
    return circles

def apply_hough_circles_with_skeletonization(roi_filled, area, w_roi, h_roi, roi_img):

    skeleton = skeletonize(roi_filled)

    # plt.imshow(skeleton)
    # plt.show()
    circles = cv2.HoughCircles(roi_filled, 
                                cv2.HOUGH_GRADIENT, 
                                dp=1,              # Inverse ratio of accumulator resolution to image resolution. Higher values mean lower resolution/precision but potentially faster processing.
                                minDist=1,           # Minimum distance between centers of detected circles.
                                param1=200,          # Higher threshold of Canny edge detector.
                                param2=5,           # Accumulator threshold for circle centers at the detection stage. Smaller values may lead to more false detections.
                                minRadius=0,         # Minimum radius of circles to be detected.
                                maxRadius = int(area))         # Maximum radius of circles to be detected. If negative, it defaults to the maximum image dimension.
    
    roi_initial_list = copy.copy(roi_img)
    roi_size_list = copy.copy(roi_img)
    roi_outside_list = copy.copy(roi_img)
    roi_skeleton = copy.copy(roi_img)

    final_circle_list = []
    if circles is not None:

        circles = np.uint16(np.around(circles))
        circles = circles[0,:]

        for circle in circles:                                
            r = int(circle[2])
            x = int(circle[0])
            y = int(circle[1])
            cv2.circle(roi_initial_list, (x, y), r, 255, thickness=1)


        if len(circles) > 1:
            circles = remove_circles_based_on_size(circles, w_roi, h_roi)
            
            for circle in circles:                                
                r = int(circle[2])
                x = int(circle[0])
                y = int(circle[1])
                cv2.circle(roi_outside_list, (x, y), r, 255, thickness=1)

            if len(circles) > 1:

                circles = sorted(circles, key=lambda x: x[2], reverse=True)
            
                # if the first circle area is too close to the whole contour area, dont keep it
                circle_area = int(int(circles[0][2]) ** 2 * np.pi)
                if circle_area > (area * 0.8):
                    circles = circles[1:]

                selected_circles = []
                selected_circles.append(circles[0])
                
                cv2.circle(roi_size_list, (int(circles[0][0]), int(circles[0][1])), int(circles[0][2]), (255, 0, 0), thickness=1)

                # save all the circles with the biggest radius that do not intersect
                for circle in circles[1:]:
                    x1 = int(circle[0])
                    y1 = int(circle[1])
                    r1 = int(circle[2])
                    overlap = False

                    for selected_circle in selected_circles:
                        if circle_overlap(selected_circle, circle):
                            overlap = True

                    if not overlap:
                        selected_circles.append(circle)
                        cv2.circle(roi_size_list, (int(circle[0]), int(circle[1])), int(circle[2]), (255, 0, 0), thickness=1)


                circles = copy.copy(selected_circles)


                if len(circles) > 1: 

                    for circle in circles: 
                        (x, y, r) = circle
                        circle_mask = np.zeros_like(skeleton, dtype=bool)
                        rr, cc = disk((y, x), r, shape=skeleton.shape)
                        circle_mask[rr, cc] = True
                        
                        if np.any(skeleton & circle_mask):
                            final_circle_list.append(circle)
                            cv2.circle(roi_skeleton, (x, y), r, (255, 0, 0), thickness=1)


    #Util.plotFourImages(roi_initial_list, roi_outside_list, roi_size_list, roi_skeleton)
    return final_circle_list


def circle_overlap(selected_circle, other):
    x1 = int(selected_circle[0])
    y1 = int(selected_circle[1])
    r1 = int(selected_circle[2])
    x2 = int(other[0])
    y2 = int(other[1])
    r2 = int(other[2])

    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return distance + 2 < (r1 + r2)

def remove_circles_based_on_size(circles, width, height):
    final_list = []
    for circle in circles:
        circle_x, circle_y = int(circle[0]), int(circle[1])
        circle_radius = int(circle[2])
        
        # Check if the circle is within the bounds of the square
        if (circle_x - circle_radius >= 0 and    # Left bound
            circle_x + circle_radius <= width and  # Right bound
            circle_y - circle_radius >= 0 and    # Top bound
            circle_y + circle_radius <= height):  # Bottom bound
        
            final_list.append(circle)
    return final_list


def remove_circles_based_on_iou_contribution(roi_filled, roi_final, circles):
    # check iou values with and without some circles to make sure that we have the best result
    # this avoids to have big circles that do not actually detect any circle

    mask_check_calculated = np.zeros_like(roi_filled)
    for circle in circles:
        r = int(circle[2])
        x = int(circle[0])
        y = int(circle[1])
        cv2.circle(mask_check_calculated, (x, y), r, 255, thickness=cv2.FILLED)
        #cv2.circle(roi_img, (x, y), r, 255, thickness=1)

    mask_check_groundtruth = np.zeros_like(roi_filled)
    original_contour, _ = cv2.findContours(roi_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in original_contour:
        cv2.drawContours(mask_check_groundtruth, [contour], -1, 255, thickness=cv2.FILLED)
    
    iou_original = np.sum(np.logical_and(mask_check_calculated, mask_check_groundtruth)) / np.sum(np.logical_or(mask_check_calculated, mask_check_groundtruth))

    # only do the check if the iou value is small already
    if iou_original < 0.8:
        max_iou = iou_original + 1
        while (max_iou > iou_original):
            j_count = 0
            
            iou_array = []
            for i_count in range(len(circles)):
                mask_check_calculated = np.zeros_like(roi_filled)

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
                
        mask_check_calculated = np.zeros_like(roi_filled)
        
        for circle in circles:
            r = int(circle[2])
            x = int(circle[0])
            y = int(circle[1])
            cv2.circle(mask_check_calculated, (x, y), r, 255, thickness=cv2.FILLED)
            cv2.circle(roi_final, (x, y), r, 255, thickness=1)
                    
    return circles

        
def k_means_approximation(circles, no_convex_points, roi):
    results = []

    circles = np.unique(circles, axis=0)

    if (len(circles) < no_convex_points): 
        return circles
    
    model = KMeans(n_clusters=no_convex_points, random_state=0, n_init='auto').fit(circles)
    
    for circle in model.cluster_centers_:
        results.append(circle)
        cv2.circle(roi, (int(circle[0]), int(circle[1])), int(circle[2]), 255, thickness=1)

    return results



# def remove_bad_circles( circles, area, roi, gray):
#     i_count = 0
#     j_count = 0


#     while(len(circles) > 1 and i_count < len(circles) - 2):
#         circle1 = circles[i_count]
#         j_count = 0

#         while(len(circles) > 1 and j_count < len(circles)): 
#             if (j_count != i_count):
#                 circle2 = circles[j_count]
#                 r1 = int(circle1[2])
#                 r2 = int(circle2[2])
#                 x1 = int(circle1[0])
#                 x2 = int(circle2[0])
#                 y1 = int(circle1[1])
#                 y2 = int(circle2[1])

#                 # calculate distance between centers
#                 dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

#                 distance_threshold_related_to_area = config.DISTANCE_THRESHOLD * area  

#                 # if dist is bigger than the sum of the radius, circles dont overlap
#                 if dist >= r1 + r2:
#                     overlapping_area = 0
#                     j_count += 1
                
#                 # if one circle is almost completely within the other
#                 elif dist <= r2 or dist <= r1 or (dist <= distance_threshold_related_to_area):
#                     overlapping_area = math.pi * min(r1, r2) ** 2

#                     iou1, iou2 = check_iou(x1, x2, y1, y2, r1, r2, roi, gray)

#                     if iou1 >= iou2:
#                         circles = np.delete(np.array(circles), j_count, axis = 0)
#                     elif iou1 < iou2:
#                         circles = np.delete(np.array(circles), i_count, axis = 0)

#                 # calculate overlapping area
#                 else: 
#                     #TODO change this to be masks and overlap 
#                     part1 = r1**2 * math.acos((dist**2 + r1**2 - r2**2) / (2 * dist * r1))
#                     part2 = r2**2 * math.acos((dist**2 + r2**2 - r1**2) / (2 * dist * r2))
#                     part3 = 0.5 * math.sqrt((-dist + r1 + r2) * (dist + r1 - r2) * (dist - r1 + r2) * (dist + r1 + r2))
#                     overlapping_area = part1 + part2 + part3

#                     # if circles are too overlapped they loose meaning
#                     if overlapping_area > area * 0.60:
#                         circles = np.delete(np.array(circles), j_count, axis = 0)
#                     else: j_count += 1
                
#                 if i_count >= len(circles): 
#                     break
            
#             else: j_count += 1
        
#         i_count += 1
#     return circles

# def check_iou(cx, cy, radii, mask_original, image):
#     mask_circle = np.zeros_like(mask_original)
#     for center_y, center_x, radius in zip(cy, cx, radii):
#         cv2.circle(mask_circle, (center_x, center_y), radius, 255, thickness=cv2.FILLED)
#         cv2.circle(image, (center_x, center_y), radius, (255, 0, 0), thickness=cv2.FILLED)
            
#     iou = np.sum(np.logical_and(mask_circle, mask_original)) / np.sum(np.logical_or(mask_circle, mask_original))

#     #Util.plotTwoImages(mask_circle, mask_original)
#     return iou, image




# # def check_iou(x1, x2, y1, y2, r1, r2, roi, gray):
# #     mask_check1 = np.zeros_like(roi)
# #     mask_check2 = np.zeros_like(roi)
# #     mask_check_original = np.zeros_like(roi)

# #     original_contour, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     for contour in original_contour:
# #         cv2.drawContours(mask_check_original, [contour], -1, 255, thickness=cv2.FILLED)
    
# #     # check which circle hits more pixels from the original image
# #     cv2.circle(mask_check1, (x1, y1), r1, 255, thickness=cv2.FILLED)
# #     cv2.circle(mask_check2, (x2, y2), r2, 255, thickness=cv2.FILLED)
    
# #     iou1 = np.sum(np.logical_and(mask_check1, mask_check_original)) / np.sum(np.logical_or(mask_check1, mask_check_original))
# #     iou2 = np.sum(np.logical_and(mask_check2, mask_check_original)) / np.sum(np.logical_or(mask_check2, mask_check_original))
    
# #     return iou1, iou2

# def check_overlapped( x1, x2, y1, y2, r1, r2, roi, gray):
#     # create mask1 and mask2
#     mask_check1 = np.zeros_like(roi)
#     mask_check2 = np.zeros_like(roi)
#     total_mask = np.zeros_like(roi)        
#     cv2.circle(mask_check1, (x1, y1), r1, 255, thickness=cv2.FILLED)
#     cv2.circle(mask_check2, (x2, y2), r2, 255, thickness=cv2.FILLED)
#     cv2.circle(total_mask, (x1, y1), r1, 255, thickness=cv2.FILLED)
#     cv2.circle(total_mask, (x2, y2), r2, 255, thickness=cv2.FILLED)
    
#     return np.sum(np.logical_and(mask_check1, mask_check2)), np.sum(total_mask == 255)

# def safe_acos( value):
#     # Ensure value is within the valid range for acos
#     return math.acos(min(1, max(-1, value)))
#     # mask_check1 = np.zeros_like(roi)
#     # mask_check2 = np.zeros_like(roi)
    
#     # # check which circle hits more pixels from the original image
#     # cv2.circle(mask_check1, (x1, y1), r1, 255, thickness=cv2.FILLED)
#     # cv2.circle(mask_check2, (x2, y2), r2, 255, thickness=cv2.FILLED)

#     # iou = np.sum(np.logical_and(mask_check1, mask_check2)) / np.sum(np.logical_or(mask_check1, mask_check2))
    
#     # return iou     
# def calculate_no_pixels( x, y, r, roi):
#     mask = np.zeros_like(roi)
#     cv2.circle(mask, (x, y), r, 255, thickness=cv2.FILLED)
#     return np.sum(mask == 255)

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


    # circles = cv2.HoughCircles(roi_img, 
    #                             cv2.HOUGH_GRADIENT, 
    #                             dp=1.3,                # Inverse ratio of accumulator resolution to image resolution. Higher values mean lower resolution/precision but potentially faster processing.
    #                             minDist=1,           # Minimum distance between centers of detected circles.
    #                             param1=200,          # Higher threshold of Canny edge detector.
    #                             param2=10,           # Accumulator threshold for circle centers at the detection stage. Smaller values may lead to more false detections.
    #                             minRadius=0,         # Minimum radius of circles to be detected.
    #                             maxRadius=0)         # Maximum radius of circles to be detected. If negative, it defaults to the maximum image dimension.
    
    # roi_final1 = copy.copy(roi)
    # roi_final2 = copy.copy(roi)
    # roi_final3 = copy.copy(roi_img)

    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     circles = circles[0,:]

    #     for circle in circles:                                
    #         r = int(circle[2])
    #         x = int(circle[0])
    #         y = int(circle[1])
    #         cv2.circle(roi_img, (x, y), r, 255, thickness=1)

    #     plt.imshow(roi_img)
    #     plt.show
    #     # kmeans approximation with the final number of circles
    #     circles = self.k_means_approximation(circles, roi_final1, no_convex_points)

    #     if len(circles) > 1:
    #         # create masks
    #         mask_check_calculated = np.zeros_like(roi)
    #         for circle in circles:
    #             r = int(circle[2])
    #             x = int(circle[0])
    #             y = int(circle[1])
    #             cv2.circle(mask_check_calculated, (x, y), r, 255, thickness=cv2.FILLED)
    #             cv2.circle(roi_img, (x, y), r, 255, thickness=1)

    #         mask_check_groundtruth = np.zeros_like(roi)
    #         original_contour, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         for contour in original_contour:
    #             cv2.drawContours(mask_check_groundtruth, [contour], -1, 255, thickness=cv2.FILLED)
            
    #         iou_original = np.sum(np.logical_and(mask_check_calculated, mask_check_groundtruth)) / np.sum(np.logical_or(mask_check_calculated, mask_check_groundtruth))

    #         plt.imshow(roi_img)
    #         plt.show()
            
    #         if iou_original < 0.8:
    #             j_count = 0
                
    #             iou_array = []
    #             for i_count in range(len(circles)):
    #                 mask_check_calculated = np.zeros_like(roi)

    #                 for j_count, circle in enumerate(circles):
    #                     if (j_count == i_count):
    #                         continue

    #                     r = int(circle[2])
    #                     x = int(circle[0])
    #                     y = int(circle[1])
    #                     cv2.circle(mask_check_calculated, (x, y), r, 255, thickness=cv2.FILLED)

    #                 iou_aux = np.sum(np.logical_and(mask_check_calculated, mask_check_groundtruth)) / np.sum(np.logical_or(mask_check_calculated, mask_check_groundtruth))
    #                 iou_array.append(iou_aux)
                

    #             max_iou = max(iou_array)
    #             if max_iou > iou_original:
    #                 circles = np.delete(np.array(circles), iou_array.index(max_iou), axis = 0)
                        
    #             mask_check_calculated = np.zeros_like(roi)
    #             for circle in circles:
    #                 r = int(circle[2])
    #                 x = int(circle[0])
    #                 y = int(circle[1])
    #                 cv2.circle(mask_check_calculated, (x, y), r, 255, thickness=cv2.FILLED)
    #                 cv2.circle(roi_final3, (x, y), r, 255, thickness=2)
                

    #             # plt.imshow(roi_final3)
    #             # plt.show()

    #     circles = remove_bad_circles(circles, area, roi, roi_final2)
# import os
# image_gray = cv2.imread(os.path.join(config.DATA_ARTIFICIAL_WSP_IMAGE_DIR, "1000.png"), cv2.IMREAD_GRAYSCALE)
# #gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

# edges = canny(image_gray, low_threshold=100, high_threshold=250)
# hough_radii = np.arange(1, 20, 1)
# hough_res = hough_circle(edges, hough_radii)
# accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1000)

# for center_y, center_x, radius in zip(cy, cx, radii):
#     cv2.circle(image_gray, (center_x, center_y), radius, 255, thickness=1)
#     # circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
#     # image[circy, circx] = (220, 20, 20)

# plt.imshow(image_gray)
# plt.show()











    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
#     #edges = canny(roi)
#    # Util.plotTwoImages(edges, roi)
#     circles = []
#     iou = 0

#     min_radius = 0
#     if not isOnEdge:

#         # while iou < 0.7 and min_radius < 10:
#         #     min_radius += 1
#         hough_radii = np.arange(1, 20, 1)
#         hough_res = hough_circle(roi, hough_radii)
#         accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks = no_convex_points + 1)
        
#         iou, roi_img = check_iou(cy, cx, radii, roi_filled, roi_img)
            
#             # plt.imshow(roi_img)
#             # plt.show()

        
#         for center_y, center_x, radius in zip(cy, cx, radii):
#             cv2.circle(roi_img, (center_x, center_y), radius, (255, 0, 0), thickness=1)
#             circles.append([center_x, center_y, radius])

#         # plt.imshow(roi_img)
#         # plt.show()