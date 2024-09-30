import os
import sys 
import cv2
import numpy as np
import csv
import time
import copy
import gc
import pandas as pd
import skimage
from matplotlib import pyplot as plt 
from shapely import Polygon, MultiPolygon, unary_union
from pathlib import Path
import shutil

sys.path.insert(0, 'src')
import Common.Util as FoldersUtil
import evaluate_algorithms_config
import Common.config as config
import Segmentation.dataset.DatasetUtil as dataset_util

from urllib.parse import urlparse
from cellpose import models, core, io

def are_polygons_equal(poly1, poly2, tolerance=1e-6):
    """Custom function to check if two polygons are the same within a given tolerance."""
    return poly1.equals_exact(poly2, tolerance)

def handle_edge_cases(label_path, predicted_droplets, width, height, distance_threshold):
    joined_droplets = []
    with open(label_path, "w") as file:
        # for each normal droplet, write it 
        for (droplet, i, isEdge) in predicted_droplets:
            if i not in joined_droplets:
                # if the droplet is on the edge try to find the rest of the droplet
                if isEdge:
                    for (droplet2, j, isEdge) in predicted_droplets:
                        if isEdge and i != j and j not in joined_droplets:
            
                            # check if the polygons are close enough to be considered the same
                            poly1 = Polygon(droplet)
                            poly2 = Polygon(droplet2)

                            if poly1.is_valid and poly2.is_valid:

                                if poly1.intersects(poly2) or poly1.distance(poly2) < distance_threshold:
                                    # merge polygons
                                    merged_polygon = unary_union([poly1, poly2])

                                    if isinstance(merged_polygon, MultiPolygon): continue
                                
                                    exterior_coords = list(merged_polygon.exterior.coords)
                                    
                                    yolo_coords = []
                                    for x, y in exterior_coords:
                                        normalized_x = x / width
                                        normalized_y = y / height
                                        yolo_coords.append((normalized_x, normalized_y))
                                

                                    yolo_line = f"0"
                                    for (x_norm, y_norm) in yolo_coords:
                                        yolo_line += f" {x_norm:.10f} {y_norm:.10f}"
                                    file.write(yolo_line + "\n")

                                    joined_droplets.append(i)
                                    joined_droplets.append(j)
                                    break  
                            else: 
                                break

                    normalized_points = []
                    for point in droplet:
                        x, y = point
                    
                        x_normalized = x / width
                        y_normalized = y / height
                        normalized_points.append((x_normalized, y_normalized))

                    yolo_line = f"0"
                    for (x_norm, y_norm) in normalized_points:
                        yolo_line += f" {x_norm:.10f} {y_norm:.10f}"
                    file.write(yolo_line + "\n")

                            
                
                # if not save it as is
                else:
                    # write each one of the points in the label file
                    normalized_points = []
                    for point in droplet:
                        x, y = point
                    
                        x_normalized = x / width
                        y_normalized = y / height
                        normalized_points.append((x_normalized, y_normalized))

                    yolo_line = f"0"
                    for (x_norm, y_norm) in normalized_points:
                        yolo_line += f" {x_norm:.10f} {y_norm:.10f}"
                    file.write(yolo_line + "\n")

def load_and_convert_image(file_path):
    img = skimage.io.imread(file_path)
    if img.shape[-1] == 4:  # If the image has an alpha channel
        img = skimage.color.rgba2rgb(img)  # Convert RGBA to RGB
    return skimage.color.rgb2gray(img) 


def save_shapes_to_yolo_label(label_path, droplets_detected, width, height):
    with open(label_path, "w") as file:
        for droplet in droplets_detected:
            # write each one of the points in the label file
            normalized_points = []
            for point in droplet:
                x, y = point
                # Normalize coordinates by dividing by the image dimensions and converting to percentages
                x_normalized = x / width
                y_normalized = y / height
                normalized_points.append((x_normalized, y_normalized))

            yolo_line = f"0"
            for (x_norm, y_norm) in normalized_points:
                yolo_line += f" {x_norm:.10f} {y_norm:.10f}"
            file.write(yolo_line + "\n")

def compute_cellpose_segmentation_full(filename, image_path, preannotation_path, last_index, x_offset, y_offset):
    height, width = cv2.imread(image_path).shape[:2]
    
    use_GPU = core.use_gpu()

    files = [image_path, image_path]
    
    imgs = [load_and_convert_image(f) for f in files]
    nimg = len(imgs)
    imgs_2D = imgs[:-1]

    model = models.Cellpose(gpu = use_GPU, model_type = 'cyto3')

    channels = [[0, 0], [0, 0]]
    diameter = 10
    masks, flows, styles, diams = model.eval(imgs_2D, diameter=diameter, flow_threshold=None, channels=channels)
    io.save_masks(imgs_2D, masks, flows, files, png=True, savedir = preannotation_path, save_txt = True)

    #shutil.copy(os.path.join(preannotation_path, "temp_cp_outlines.txt"), os.path.join(preannotation_path, filename + "_cp_outlines.txt"),)
    with open(os.path.join(preannotation_path, "temp_cp_outlines.txt"), 'r') as file:
        lines = file.readlines()

    polygons = []

    for i, line in enumerate(lines):
        points = list(map(int, line.strip().split(',')))
        coordinates = [(points[i] + x_offset, points[i+1] + y_offset) for i in range(0, len(points), 2)]
        contour = np.array(coordinates, dtype=np.int32)

        polygons.append(contour)
    

    predicted_droplets_adjusted_with_edges = []
    edge_zone_width = 5
    for i, polygon in enumerate(polygons):
        isEdge = False
        for point in polygon:
            if (point[0] < edge_zone_width or 
                point[0] > width - edge_zone_width or
                point[1] < edge_zone_width or 
                point[1] > height - edge_zone_width):

                isEdge = True

        predicted_droplets_adjusted_with_edges.append((polygon, i + last_index, isEdge))

    return predicted_droplets_adjusted_with_edges, last_index

def compute_cellpose_segmentation(filename, image_path, preannotation_path, diameter_median):
    use_GPU = core.use_gpu()

    files = [image_path, image_path]
    
    imgs = [load_and_convert_image(f) for f in files]
    nimg = len(imgs)
    imgs_2D = imgs[:-1]

    model = models.Cellpose(gpu = use_GPU, model_type = 'cyto3')

    channels = [[0, 0], [0, 0]]
    diameter = diameter_median
    masks, flows, styles, diams = model.eval(imgs_2D, diameter=diameter, flow_threshold=None, channels=channels)
    io.save_masks(imgs_2D, masks, flows, files, png=True, savedir = preannotation_path, save_txt = True)

    with open(os.path.join(preannotation_path, filename + "_cp_outlines.txt"), 'r') as file:
        lines = file.readlines()

    polygons = []

    for i, line in enumerate(lines):
        points = list(map(int, line.strip().split(',')))
        coordinates = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
        contour = np.array(coordinates, dtype=np.int32)

        polygons.append(contour)

    return polygons
            

def save_shapes_to_yolo_label(label_path, droplets_detected, width, height):
    with open(label_path, "w") as file:
        for droplet in droplets_detected:
            # write each one of the points in the label file
            normalized_points = []
            for point in droplet:
                x, y = point
                # Normalize coordinates by dividing by the image dimensions and converting to percentages
                x_normalized = x / width
                y_normalized = y / height
                normalized_points.append((x_normalized, y_normalized))

            yolo_line = f"0"
            for (x_norm, y_norm) in normalized_points:
                yolo_line += f" {x_norm:.10f} {y_norm:.10f}"
            file.write(yolo_line + "\n")
               
def manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics):
    directory_image = os.path.join(path_dataset, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
    directory_label = os.path.join(path_dataset, config.DATA_GENERAL_LABEL_FOLDER_NAME)
    directory_stats = os.path.join(path_dataset, config.DATA_GENERAL_STATS_FOLDER_NAME)

    # manage folders to store the results of the segmentation
    list_folders = []
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_STATS_FOLDER_NAME))
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME))
    list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME))

    FoldersUtil.manage_folders(list_folders)


    return directory_image, directory_label, directory_stats


def main_cellpose_full(fieldnames_segmentation, fieldnames_statistics, fieldnames_time, path_csv_segmentation, path_csv_statistics, path_dataset, path_results):

    directory_image, directory_label, directory_stats = manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics)
 
    segmentation_time_csv_path = os.path.join(path_results, config.RESULTS_GENERAL_SEGMENTATIONTIME_FOLDER_NAME + ".csv")
    
    with open(segmentation_time_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(fieldnames_time)
        
        # apply the segmentation in each one of the images and then calculate the accuracy and save it
        for i, file in enumerate(os.listdir(directory_image)): 
        
            start_time = time.time()
            filename = file.split(".")[0]
        
            full_image_path = os.path.join(directory_image, file)
            image_colors = cv2.imread(os.path.join(directory_image, file))  
            image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
            height, width = image_colors.shape[:2]

            segmentation_time = 0
            total_predicted_droplets = []
           
            label_path = os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME, filename + ".txt")
            preannotation_path = os.path.join(path_results, "preannotation")

            print("Evaluating image", filename + "..." )

            try:
                # divide into squares
                # squares = dataset_util.divide_image_into_squares_simple(image_colors)
                # last_index = 0
                
                # for (square, x_offset, y_offset, (x, y, _)) in squares:
                #     start_time = time.time()
                    
                #     # save temporarly for cellpose segmentation and apply the segmentation
                #     image_path = os.path.join(path_results, "temp.png")
                #     cv2.imwrite(image_path, square)
                #     predicted_droplets, last_index = compute_cellpose_segmentation_full(filename, image_path, preannotation_path, last_index, x_offset, y_offset)
                    
                #     seg_time = time.time()
                #     segmentation_time += seg_time - start_time

                #     total_predicted_droplets.extend(predicted_droplets)
                
                print(segmentation_time)
                # save segmentation time to a file
                csv_writer.writerow([filename, segmentation_time])

                handle_edge_cases(label_path, total_predicted_droplets, width, height, 5)

                #display_results(full_image_path, label_path)
                
                #save_shapes_to_yolo_label(label_path, predicted_droplets, width, height)

            except np.core._exceptions._ArrayMemoryError as e:
                print(f"Memory error encountered while processing {filename}: {e}")

def main_cellpose_square(fieldnames_segmentation, fieldnames_statistics, fieldnames_time, path_csv_segmentation, path_csv_statistics, path_dataset, path_results):
    directory_image, directory_label, directory_stats = manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics)
    
    segmentation_time_csv_path = os.path.join(path_results, config.RESULTS_GENERAL_SEGMENTATIONTIME_FOLDER_NAME + ".csv")
    
    with open(segmentation_time_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(fieldnames_time)
        

        for i, file in enumerate(os.listdir(directory_image)): 
            start_time = time.time()
            filename = file.split(".")[0]
        
            image_path = os.path.join(directory_image, file)
            image_colors = skimage.io.imread(image_path)
            height, width = image_colors.shape[:2]
            image_area = width * height

            # if height >= 710:
            #     diameter_median = 15
            # elif height >= 630:
            #     diameter_median = 12
            # elif height >= 505:
            #     diameter_median = 10
            # else: 
            diameter_median = 10

            label_path = os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME, filename + ".txt")
            preannotation_path = os.path.join(path_results, "preannotation")

            print("Evaluating image", filename + "..." )

            try:
                # apply segmentation
                predicted_droplets = compute_cellpose_segmentation(filename, image_path, preannotation_path, diameter_median)
                
                seg_time = time.time()
                segmentation_time = seg_time - start_time

                # save segmentation time to a file
                csv_writer.writerow([filename, segmentation_time])

                save_shapes_to_yolo_label(label_path, predicted_droplets, width, height)

            except np.core._exceptions._ArrayMemoryError as e:
                print(f"Memory error encountered while processing {filename}: {e}")


# SYNTHETIC DATASET
# main_cellpose_square(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
#            fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
#            fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
#            path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_CELLPOSE, 
#            path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_CELLPOSE, 
#            path_dataset=config.DATA_SYNTHETIC_WSP_TESTING_DIR, 
#            path_results=config.RESULTS_SYNTHETIC_CELLPOSE_DIR)

# REAL DATASET
# main_cellpose_square(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
#            fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
#            fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
#            path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_CELLPOSE, 
#            path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_CELLPOSE, 
#            path_dataset=config.DATA_REAL_WSP_TESTING_DIR2, 
#            path_results=config.RESULTS_REAL_CELLPOSE_DIR )

main_cellpose_square(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
           fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
           fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
           path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_FULL_DATASET_CELLPOSE, 
           path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_FULL_DATASET_CELLPOSE, 
           path_dataset=config.DATA_SYNTHETIC_FULL_WSP_TESTING_DIR, 
           path_results=config.RESULTS_SYNTHETIC_FULL_CELLPOSE_DIR)

main_cellpose_square(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
           fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
           fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
           path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_FULL_DATASET_CELLPOSE, 
           path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_FULL_DATASET_CELLPOSE, 
           path_dataset=config.DATA_REAL_FULL_WSP_TESTING_DIR, 
           path_results=config.RESULTS_REAL_FULL_CELLPOSE_DIR)



