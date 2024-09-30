import os
import sys 
import cv2
import numpy as np
import csv
import time
import copy
import gc
import pandas as pd

from matplotlib import pyplot as plt 
from pathlib import Path
from ultralytics import YOLO
from shapely import Polygon, MultiPolygon, unary_union

sys.path.insert(0, 'src')
import Common.Util as FoldersUtil
import evaluate_algorithms_config
import Common.config as config
import Segmentation.dataset.DatasetUtil as dataset_util

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
 
def compute_yolo_segmentation_normal(image_path, yolo_model):
    predicted_droplets_adjusted = []
    # predict image results
    
    results = yolo_model(image_path, conf=0.1)

    if results[0].masks:
        segmentation_result = results[0].masks.xy
       
        detected_pts = []

        for polygon in segmentation_result:
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            detected_pts.append(pts)

        for coords in detected_pts:
            adjusted_coords = []
            for point in coords:
                x, y = point[0]
                adjusted_coords.append([x, y])
            if adjusted_coords != []:
                predicted_droplets_adjusted.append(np.array(adjusted_coords, dtype=np.int32))

    return predicted_droplets_adjusted

def compute_yolo_segmentation_full(image_path, full_image_path, yolo_model, last_index, x_offset, y_offset):
    predicted_droplets_adjusted = []
    predicted_droplets_adjusted_with_edges = []

    # predict image results
    image = cv2.imread(full_image_path)
    height, width = image.shape[:2]

    results = yolo_model(image_path, conf=0.3)


    if results[0].masks:
        segmentation_result = results[0].masks.xy
       
        detected_pts = []

        for polygon in segmentation_result:
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            detected_pts.append(pts)

        for coords in detected_pts:
            adjusted_coords = []
            for point in coords:
                x, y = point[0]
                adjusted_coords.append([x + x_offset, y + y_offset])
            if adjusted_coords != [] and len(adjusted_coords) >= 4:
                predicted_droplets_adjusted.append(np.array(adjusted_coords, dtype=np.int32))
            


        # check which droplets are on the edge
        edge_zone_width = 5
        for i, polygon in enumerate(predicted_droplets_adjusted):
            isEdge = False
            cv2.drawContours(image, [polygon], -1, (255, 0, 0), 1)

            for point in polygon:
                if (point[0] < edge_zone_width or 
                    point[0] > width - edge_zone_width or
                    point[1] < edge_zone_width or 
                    point[1] > height - edge_zone_width):

                    isEdge = True
        
            predicted_droplets_adjusted_with_edges.append((polygon, i + last_index, isEdge))

        
 
        #     if isEdge:
        #         cv2.drawContours(image, [polygon], -1, (255, 0, 0), 1)
        #     else:
        #         cv2.drawContours(image, [polygon], -1, (255, 255, 0), 1)
            

        # plt.imshow(image)
        # plt.show()              
            
    return predicted_droplets_adjusted_with_edges, len(predicted_droplets_adjusted_with_edges) + last_index

def save_shapes_to_yolo_label(label_path, droplets_detected, width, height):
    with open(label_path, "w") as file:
        for droplet in droplets_detected:
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

    with open(path_csv_segmentation, mode='w', newline='') as file:
        csv.DictWriter(file, fieldnames=fieldnames_segmentation).writeheader()

    with open(path_csv_statistics, mode='w', newline='') as file:
        csv.DictWriter(file, fieldnames=fieldnames_statistics).writeheader()

    return directory_image, directory_label, directory_stats

def display_results(image_path, label_path):
    image = cv2.imread(image_path)
    #image = cv2.colorChange(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]

    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()            
            coordinates = list(map(float, parts[1:]))
            
            if len(coordinates) >= 8:
                polygon = [(coordinates[i] * width, coordinates[i+1] * height) for i in range(0, len(coordinates), 2)]
                contour = np.array(polygon, dtype=np.int32)

                cv2.drawContours(image, [contour], -1, (255, 0, 0), 1)

    # plt.imshow(image)
    # plt.show()
    return
                
def main_yolo_squares(fieldnames_segmentation, fieldnames_statistics, fieldnames_time, path_csv_segmentation, path_csv_statistics, path_dataset, path_results, yolo_model_path):
    directory_image, directory_label, directory_stats = manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics)
 
    segmentation_time_csv_path = os.path.join(path_results, config.RESULTS_GENERAL_SEGMENTATIONTIME_FOLDER_NAME + ".csv")
    with open(segmentation_time_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(fieldnames_time)
        
        # apply the segmentation in each one of the images and then calculate the accuracy and save it
        for i, file in enumerate(os.listdir(directory_image)): 
        
            start_time = time.time()
            yolo_model = YOLO(yolo_model_path)
            filename = file.split(".")[0]
        
            image_path = os.path.join(directory_image, file)
            image_colors = cv2.imread(os.path.join(directory_image, file))  
            image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
            height, width = image_colors.shape[:2]
           
            label_path = os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME, filename + ".txt")

            print("Evaluating image", filename + "..." )

            try:
                predicted_droplets = compute_yolo_segmentation_normal(image_path, yolo_model)
                
                seg_time = time.time()
                segmentation_time = seg_time - start_time
                
                # save segmentation time to a file
                csv_writer.writerow([filename, segmentation_time])

                save_shapes_to_yolo_label(label_path, predicted_droplets, width, height)

            except np.core._exceptions._ArrayMemoryError as e:
                print(f"Memory error encountered while processing {filename}: {e}")

def main_yolo_full(fieldnames_segmentation, fieldnames_statistics, fieldnames_time, path_csv_segmentation, path_csv_statistics, path_dataset, path_results, yolo_model_path):

    directory_image, directory_label, directory_stats = manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics)
 
    segmentation_time_csv_path = os.path.join(path_results, config.RESULTS_GENERAL_SEGMENTATIONTIME_FOLDER_NAME + ".csv")
    with open(segmentation_time_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(fieldnames_time)
        
        # apply the segmentation in each one of the images and then calculate the accuracy and save it
        for i, file in enumerate(os.listdir(directory_image)): 
        
            start_time = time.time()
            yolo_model = YOLO(yolo_model_path)
            filename = file.split(".")[0]
        
            full_image_path = os.path.join(directory_image, file)
            image_colors = cv2.imread(os.path.join(directory_image, file))  
            image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
            height, width = image_colors.shape[:2]

            segmentation_time = 0
            total_predicted_droplets = []
           
            label_path = os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME, filename + ".txt")

            print("Evaluating image", filename + "..." )

            try:
                # divide into squares
                squares = dataset_util.divide_image_into_squares_simple(image_colors)
                last_index = 0
                for (square, x_offset, y_offset, (x, y, _)) in squares:
                    start_time = time.time()
                    
                    # save temporarly for yolo segmentation and apply the segmentation
                    image_path = os.path.join(path_results, "temp.png")
                    cv2.imwrite(image_path, square)
                    predicted_droplets, last_index = compute_yolo_segmentation_full(image_path, full_image_path, yolo_model, last_index, x_offset, y_offset)
                    
                    seg_time = time.time()
                    segmentation_time += seg_time - start_time

                    total_predicted_droplets.extend(predicted_droplets)

                # save segmentation time to a file
                csv_writer.writerow([filename, segmentation_time])

                handle_edge_cases(label_path, total_predicted_droplets, width, height, 5)

                #display_results(full_image_path, label_path)
                
                #save_shapes_to_yolo_label(label_path, predicted_droplets, width, height)

            except np.core._exceptions._ArrayMemoryError as e:
                print(f"Memory error encountered while processing {filename}: {e}")


yolo_model_path = evaluate_algorithms_config.DROPLET_YOLO_MODEL


#### SQUARE IMAGE
# SYNTHETIC DATASET
# main_yolo_squares(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
#           evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
#           evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
#           evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO, 
#           evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO, 
#           config.DATA_SYNTHETIC_WSP_TESTING_DIR,
#           config.RESULTS_SYNTHETIC_YOLO_DIR, 
#           yolo_model_path)

# # REAL DATASET
# main_yolo_squares(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
#           evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
#           evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
#           evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_YOLO, 
#           evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_YOLO, 
#           config.DATA_REAL_WSP_TESTING_DIR2, 
#           config.RESULTS_REAL_YOLO_DIR, 
#           yolo_model_path)

#### FULL IMAGE
# SYNTHETIC DATASET
# main_yolo_full(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
#           evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
#           evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
#           evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_FULL_DATASET_YOLO, 
#           evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_FULL_DATASET_YOLO, 
#           config.DATA_SYNTHETIC_FULL_WSP_TESTING_DIR, 
#           config.RESULTS_SYNTHETIC_FULL_YOLO_DIR, 
#           yolo_model_path)

# # REAL DATASET
main_yolo_full(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
          evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
          evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
          evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_FULL_DATASET_YOLO, 
          evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_FULL_DATASET_YOLO, 
          config.DATA_REAL_FULL_WSP_TESTING_DIR, 
          config.RESULTS_REAL_FULL_YOLO_DIR, 
          yolo_model_path)