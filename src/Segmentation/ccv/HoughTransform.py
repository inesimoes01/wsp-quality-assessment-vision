import cv2
import copy
import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.morphology import skeletonize
from skimage.draw import disk

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

        if len(circles) > 1:
            circles = remove_circles_outside_mask(circles, roi_filled)

        # kmeans approximation with the final number of circles
        circles = k_means_approximation(circles, no_convex_points, roi_kmeans)

        if len(circles) > 1:
            circles = remove_circles_based_on_iou_contribution(roi_filled, roi_iou_check, circles)

    
    return circles, roi_initial_list, roi_kmeans, roi_iou_check

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


def remove_circles_outside_mask(circles, roi_mask):
    final_list_circles = []

    for circle in circles:
        circle_mask = np.zeros_like(roi_mask)

        cv2.circle(circle_mask, (int(circle[0]), int(circle[1])), int(circle[2]), 255, cv2.FILLED)

        intersection = cv2.bitwise_and(roi_mask, circle_mask)
        intersection_exists = np.any(intersection)

        if intersection_exists:
            final_list_circles.append(circle)

    return final_list_circles

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

    if (len(circles) < no_convex_points + 2): 
        return circles
    
    model = KMeans(n_clusters=no_convex_points + 2, random_state=0, n_init='auto').fit(circles)
    
    for circle in model.cluster_centers_:
        results.append(circle)
        cv2.circle(roi, (int(circle[0]), int(circle[1])), int(circle[2]), 255, thickness=1)

    return results

