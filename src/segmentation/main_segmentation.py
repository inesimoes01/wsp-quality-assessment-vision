import sys
import cv2
from matplotlib import pyplot as plt 
import os
import copy
import numpy as np

sys.path.insert(0, 'src/others')
from util import *
from paths import *

def read_files(filename):
    in_image = cv2.imread(os.path.join(path_to_images_folder, filename + ".png"))
    in_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
    out_image = copy.copy(in_image)

    stats_file_path = (os.path.join(path_to_statistics_folder, filename + ".txt"))
    return in_image, out_image, stats_file_path

def get_ground_truth(stats_file_path):
    with open(stats_file_path, 'r') as f:
        for line in f:
            if "Number of droplets: " in line:
                number_of_droplets = int(line.split(":")[1].strip())
    return number_of_droplets

def get_contours(image):
    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = copy.copy(gray)

    # canny edge detectio + thresholding + contours
    edges = cv2.Canny(gray, 50, 150)
    ret, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    contour_image = copy.copy(image)

    # number of contours
    object_count = len(contours)

    return contours, contour_image, object_count

def measure_diameter(contour, no_droplets, diameter_image, separate_image):   
    # bool variable to check if the algorithm sees the object as overlapped or single droplets
    isOverlapped = 0

    # find minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(contour)
    diameter = radius * 2

    # annotate image for diameter values
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(diameter_image, center, radius, (255, 0, 0), 2)
    cv2.putText(diameter_image, f'{diameter:.2f}', (int(x-radius), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # perform shape analysis
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # droplets of only 1 pixel
    if(radius <= 1):    
        cv2.drawContours(separate_image, [contour], -1, (102, 0, 204), 2)
        return separate_image, diameter_image, no_droplets, isOverlapped

    # calculate parameter to evaluate the circularity of the droplet
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # classify based on properties
    if circularity > 0.8: 
        cv2.drawContours(separate_image, [contour], -1, (102, 0, 204), 2)
    else:
        cv2.drawContours(separate_image, [contour], -1, (44, 156, 63), 2)
        isOverlapped = 1
        no_droplets += 1

    return separate_image, diameter_image, no_droplets, isOverlapped
        
def crop_ROI(index, contour, enumerate_image, roi_image, isOverlapped, filename):
    # crop region of interest
    x, y, w, h = cv2.boundingRect(contour)
    expansion_factor = 2
    expanded_w = int(w * expansion_factor)
    expanded_h = int(h * expansion_factor)
    x -= int((expanded_w - w) / 2)
    y -= int((expanded_h - h) / 2)
    x = max(x, 0)
    y = max(y, 0)
    object_roi = roi_image[y:y+expanded_h, x:x+expanded_w]
    (x, y), radius = cv2.minEnclosingCircle(contour)
    cv2.putText(enumerate_image, f'{index}', (int(x-radius), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # save outputs
    object_roi = cv2.cvtColor(object_roi, cv2.COLOR_RGB2BGR)

    if (isOverlapped): cv2.imwrite(path_to_save_contours_overlapped + '\\' + str(index) + '.png', object_roi)
    else: cv2.imwrite(path_to_save_contours_single + '\\' + str(index) + '.png', object_roi)

def display_results(number_of_droplets, object_count, transformed_image, original_image):
    fig = plt.figure(figsize = (10, 7))
    fig.add_subplot(2, 1, 1)
    plt.imshow(original_image)
    plt.title("IN Number of droplets: {}".format(number_of_droplets))
    fig.add_subplot(2, 1, 2)
    plt.imshow(transformed_image)
    plt.title("OUT Number of droplets: {}".format(object_count))

    plt.show()





# delete old outputs
delete_folder_contents(path_to_separation_folder)
delete_folder_contents(path_to_outputs_folder)

# for each one of the images of the dataset
for file in os.listdir(path_to_images_folder):
    # get name of the file
    parts = file.split(".")
    filename = parts[0]
    path_to_save_contours_single = os.path.join(path_to_outputs_folder, "single", filename)
    path_to_save_contours_overlapped = os.path.join(path_to_outputs_folder, "overlapped", filename)

    # create folder to save outputs
    create_folders(path_to_save_contours_overlapped)
    create_folders(path_to_save_contours_single)

    # read files
    in_image, out_image, stats_file_path = read_files(filename)

    # get all the contours
    contours, contour_image, object_count = get_contours(out_image)

    # ground truth of number of contours
    number_of_droplets = get_ground_truth(stats_file_path)

    # save each step in a different variable
    roi_image = copy.copy(in_image)
    enumerate_image = copy.copy(in_image)
    diameter_image = copy.copy(in_image)
    separate_image = copy.copy(in_image)

    # calculate diameter + save each contour
    for i, contour in enumerate(contours):    
        separate_image, diameter_image, final_no_droplets, isOverlapped = measure_diameter(contour, object_count, diameter_image, separate_image)
    
        crop_ROI(i, contour, enumerate_image, roi_image, isOverlapped, filename)

        separate_image = cv2.cvtColor(separate_image, cv2.COLOR_RGB2BGR)
    
    # save final image
    cv2.imwrite(f'images\\artificial_dataset\\separation\\result_image_' + filename + '.png', separate_image)

    print("Number of droplets detected: ", final_no_droplets)
    print("Number of droplets before counting double: ", object_count)
    print("Number of droplets real: ", number_of_droplets)
    print("")

    #display_results(number_of_droplets, object_count, separate_image, in_image)










   



