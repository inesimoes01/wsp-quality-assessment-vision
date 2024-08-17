import cv2
import numpy as np
import copy
import os
from matplotlib import pyplot as plt 
import sys 
sys.path.insert(0, 'src/common')
import config as config
import Util


def detect_rectangle(image, image_color):
    image = copy.copy(image)

    # detect the contour of the rectangle
    largest_contour = detect_rectangle_alternative(image_color)
    cv2.drawContours(image, [largest_contour], -1, (0, 0, 0), 5)
    return largest_contour


def remove_distortion(image, contour, filename, save_steps=False):
    maxWidth, maxHeight, pts_src, pts_dst = calculate_points(contour)

    h, status = cv2.findHomography(pts_src, pts_dst)
    #matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
    undistorted_image = cv2.warpPerspective(image, h, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    
    if save_steps:
        cv2.imwrite("results\\latex\\rectangle_cv\\" + filename + "undistorted.png", undistorted_image)
    return undistorted_image

    
def calculate_points(contour):
    approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), closed=True) 
    
    if len(contour) > 5: 
        noPaper = True
        
    # order corners
    approx = sorted(approx, key=lambda x: x[0][0] + x[0][1])
    top_left = approx[0][0]
    top_right = approx[1][0]
    bottom_left = approx[2][0]
    bottom_right = approx[3][0]

    # L2 norm
    width_AD = np.sqrt(((top_left[0] - top_right[0]) ** 2) + ((top_left[1] - top_right[1]) ** 2))
    width_BC = np.sqrt(((bottom_left[0] - bottom_right[0]) ** 2) + ((bottom_left[1] - bottom_right[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
    
    height_AB = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    height_CD = np.sqrt(((bottom_right[0] - top_right[0]) ** 2) + ((bottom_right[1] - top_right[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))
    
    input_pts = np.float32([top_left, bottom_left, bottom_right, top_right])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight + 1],
                            [maxWidth +  1, maxHeight + 1],
                            [maxWidth + 1, 0]])
  
    return maxWidth, maxHeight, input_pts, output_pts

def detect_rectangle( image, image_color):
    
    blur = cv2.GaussianBlur(image, (9, 9), 0)
    #_, threshold_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, threshold_image = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)
    kernel = np.ones((10, 10), np.uint8)
    dilation = cv2.dilate(threshold_image, kernel, iterations = 1)
    edges = cv2.Canny(dilation, 127, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    cv2.drawContours(image_color, contours, -1, (255, 0, 0), 4)
    cv2.drawContours(image_color, [contours[0]], -1, (255, 255, 0), 4)


    plt.imshow(image_color)
    plt.show()


def group_lines_by_position( lines, axis, threshold=5):
    grouped_lines = []
    lines = sorted(lines, key=lambda x: x[0][axis])
    current_group = [lines[0]]

    for line in lines[1:]:
        if abs(line[0][axis] - current_group[-1][0][axis]) < threshold:
            current_group.append(line)
        else:
            grouped_lines.append(current_group)
            current_group = [line]

    grouped_lines.append(current_group)
    return grouped_lines

def draw_grouped_lines( image, grouped_lines, axis=0):
    for group in grouped_lines:
        coords = [line[0] for line in group]
        coords = sorted(coords, key=lambda x: x[1 - axis])
        start_point = tuple(coords[0][0:2])
        end_point = tuple(coords[-1][0:2])
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)

def detect_rectangle_alternative(image, filename, save_steps = False):
    edges = cv2.GaussianBlur(image, (5, 5), 3, 3)
    
    if save_steps:
        cv2.imwrite("results\\latex\\rectangle_cv\\" + filename + "blur.png", edges)

    # find the most present color
    histogram = cv2.calcHist([edges], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    max_count = np.amax(histogram)
    most_present_color = np.unravel_index(np.argmax(histogram), histogram.shape)
    most_present_color = tuple(int(c) for c in most_present_color)
    
    # remove colors based on upper and lower bounds
    tolerance = 100 
    lower_bound = np.array([max(0, c - tolerance) for c in most_present_color])
    upper_bound = np.array([min(255, c + tolerance) for c in most_present_color])
    mask = cv2.inRange(edges, lower_bound, upper_bound)
    result = cv2.bitwise_and(edges, edges, mask=cv2.bitwise_not(mask))

    if save_steps:
        cv2.imwrite("results\\latex\\rectangle_cv\\" + filename + "_mask.png", result)

    # threshold image
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    if save_steps:
        cv2.imwrite("results\\latex\\rectangle_cv\\" + filename + "_threshold.png", thresh)

    # find contours        
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    hull = []
   
    for contour in contours:
        hull.append(cv2.convexHull(contour, False))

    cv2.drawContours(image, [hull[0]], -1, (255, 0, 0), 2)

    if save_steps:
        cv2.imwrite("results\\latex\\rectangle_cv\\" + filename + "_final.png", image)

    return hull[0]


# im = "data\\real_rectangle_dataset\\test\\image\\2_V1_A3_jpg.rf.3ff6c061dd2d3f7239d33e83352914b1.jpg"
# im = cv2.imread(im)
# im_to_destroi = copy.copy(im)
# contour = detect_rectangle_alternative(im_to_destroi, "2_V1_A3_jpg.rf.3ff6c061dd2d3f7239d33e83352914b1.jpg", True)
# remove_distortion(im, contour, "2_V1_A3_jpg.rf.3ff6c061dd2d3f7239d33e83352914b1.jpg", True)
# maxHeight = 2000
# maxWidth = 300
# output_pts = np.float32([[0, 0],
#                             [0, maxHeight + 1],
#                             [maxWidth +  1, maxHeight + 1],
#                             [maxWidth + 1, 0]])

# final = remove_distortion(im, contour)

# plt.imshow(final)
# plt.show()