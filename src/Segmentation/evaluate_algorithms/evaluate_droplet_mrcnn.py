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
from pathlib import Path
import tensorflow as tf
from shapely import Polygon
from tensorflow.keras import backend as K


sys.path.insert(0, 'src')
import Common.Util as FoldersUtil
import evaluate_algorithms_config
import Common.config as config
from Segmentation.droplet.cnn.MaskRCNN.mrcnn import model as modellib, utils 
import Segmentation.droplet.cnn.MaskRCNN.custom_mrcnn_classes as custom_mrcnn_classes
import Segmentation.dataset.DatasetUtil as dataset_util

# sys.path.insert(0, 'src\\Segmentation\\droplet\\cnn\\MaskRCNN')

# from mrcnn import model as modellib, utils
# import custom_mrcnn_classes as custom_mrcnn_classes

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

def compute_mrcnn_segmentation_normal(image, mrcnn_model_path):
    inference_config = custom_mrcnn_classes.InferenceConfig()

    # recreate the model in inference mode
    mrcnn_model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config, model_dir=mrcnn_model_path)
    mrcnn_model.load_weights(mrcnn_model_path, by_name=True)

    img_arr = np.array(image)
    results = mrcnn_model.detect([img_arr], verbose=0)
    r = results[0]

    droplets_detected = []

    for i in range(r['masks'].shape[-1]):
        mask = r['masks'][:, :, i]
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            points = contour.reshape(-1, 2)
            droplets_detected.append(points)

    return droplets_detected
            
def compute_mrcnn_segmentation_full(image, full_image_path, mrcnn_model_path, last_index, x_offset, y_offset, width, height):
    predicted_droplets_adjusted = []
    predicted_droplets_adjusted_with_edges = []
    
    inference_config = custom_mrcnn_classes.InferenceConfig()
    
    # recreate the model in inference mode
    mrcnn_model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config, model_dir=mrcnn_model_path)
    mrcnn_model.load_weights(mrcnn_model_path, by_name=True)

    img_arr = np.array(image)
    results = mrcnn_model.detect([img_arr], verbose=0)
    r = results[0]

    droplets_detected = []

    for i in range(r['masks'].shape[-1]):
        mask = r['masks'][:, :, i]
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            points = contour.reshape(-1, 2)
            droplets_detected.append(points)

    for coords in droplets_detected:
        adjusted_coords = []
        for point in coords:
            x, y = point
            adjusted_coords.append([x + x_offset, y  + y_offset])
        if adjusted_coords != [] and len(adjusted_coords) >= 4:
            predicted_droplets_adjusted.append(np.array(adjusted_coords, dtype=np.int32))

    # check which droplets are on the edge
    edge_zone_width = 5
    for i, polygon in enumerate(predicted_droplets_adjusted):
        isEdge = False
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



def main_mrcnn_square(fieldnames_segmentation, fieldnames_statistics, fieldnames_time, path_csv_segmentation, path_csv_statistics, path_dataset, path_results, model_path):
    directory_image, directory_label, directory_stats = manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics)
    #list_names = ["0_1","0_4","0_6","0_8","100_12","100_2","100_3","100_5","101_1","101_15", ]
    list_names = []
    segmentation_time_csv_path = os.path.join(path_results, config.RESULTS_GENERAL_SEGMENTATIONTIME_FOLDER_NAME + ".csv")
    with open(segmentation_time_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(fieldnames_time)

        # apply the segmentation in each one of the images and then calculate the accuracy and save it
        for i, file in enumerate(os.listdir(directory_image)): 

            start_time = time.time()
            filename = file.split(".")[0]

            if filename not in list_names:

                image_path = os.path.join(directory_image, file)
                image_colors = skimage.io.imread(image_path)
                height, width = image_colors.shape[:2]
                image_area = width * height

                label_path = os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME, filename + ".txt")

                print("Evaluating image", filename + "..." )

                try:
                    # apply segmentation
                    predicted_droplets = compute_mrcnn_segmentation_normal(image_colors, model_path)
                    
                    seg_time = time.time()
                    segmentation_time = seg_time - start_time

                    # save segmentation time to a file
                    csv_writer.writerow([filename, segmentation_time])

                    save_shapes_to_yolo_label(label_path, predicted_droplets, width, height)

                    gc.collect()

                except tf.errors.ResourceExhaustedError:
                    print(f"ResourceExhaustedError: Skipping {image_path} due to memory issues")
                    K.clear_session()

                except np.core._exceptions._ArrayMemoryError as e:
                    print(f"Memory error encountered while processing {filename}: {e}")

def main_mrcnn_full(fieldnames_segmentation, fieldnames_statistics, fieldnames_time, path_csv_segmentation, path_csv_statistics, path_dataset, path_results, model_path):
    directory_image, directory_label, directory_stats = manage_folder(path_dataset, path_results, path_csv_segmentation, fieldnames_segmentation, path_csv_statistics, fieldnames_statistics)

    segmentation_time_csv_path = os.path.join(path_results, config.RESULTS_GENERAL_SEGMENTATIONTIME_FOLDER_NAME + ".csv")
    with open(segmentation_time_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(fieldnames_time)

        # apply the segmentation in each one of the images and then calculate the accuracy and save it
        for i, file in enumerate(os.listdir(directory_image)): 

            filename = file.split(".")[0]

            full_image_path = os.path.join(directory_image, file)
            image_colors = skimage.io.imread(full_image_path)
            height, width = image_colors.shape[:2]
            image_area = width * height

            segmentation_time = 0
            total_predicted_droplets = []

            label_path = os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME, filename + ".txt")

            print("Evaluating image", filename + "..." )

            try:
                squares = dataset_util.divide_image_into_squares_simple(image_colors)
                last_index = 0
                for (square, x_offset, y_offset, (x, y, _)) in squares:
                    start_time = time.time()

                    # apply segmentation
                    image_path = os.path.join(path_results, "temp.png")
                    cv2.imwrite(image_path, square)
                    image_colors = skimage.io.imread(image_path)
                    predicted_droplets = compute_mrcnn_segmentation_full(image_colors, full_image_path, model_path, last_index, x_offset, y_offset, width, height)
                    
                    seg_time = time.time()
                    segmentation_time += seg_time - start_time
                    
                    total_predicted_droplets.extend(predicted_droplets)
                
                # save segmentation time to a file
                csv_writer.writerow([filename, segmentation_time])

                handle_edge_cases(label_path, total_predicted_droplets, width, height, 5)

                #save_shapes_to_yolo_label(label_path, predicted_droplets, width, height)

                gc.collect()

            except tf.errors.ResourceExhaustedError:
                print(f"ResourceExhaustedError: Skipping {image_path} due to memory issues")
                K.clear_session()

            except np.core._exceptions._ArrayMemoryError as e:
                print(f"Memory error encountered while processing {filename}: {e}")

# SYNTHETIC DATASET
# main_mrcnn_square(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
#            fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
#            fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
#            path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_MRCNN, 
#            path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_MRCNN, 
#            path_dataset=config.DATA_SYNTHETIC_NORMAL_WSP_TESTING_DIR, 
#            path_results=config.RESULTS_SYNTHETIC_MRCNN_DIR, 
#            model_path=evaluate_algorithms_config.DROPLET_MRCNN_MODEL)

# # REAL DATASET
main_mrcnn_square(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
           fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
           fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
           path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_MRCNN, 
           path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_MRCNN, 
           path_dataset=config.DATA_REAL_WSP_TESTING_DIR2, 
           path_results=config.RESULTS_REAL_MRCNN_DIR, 
           model_path=evaluate_algorithms_config.DROPLET_MRCNN_MODEL)

# synthetic full
# main_mrcnn_full(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
#            fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
#            fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
           
#            path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_FULL_DATASET_MRCNN, 
#            path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_FULL_DATASET_MRCNN, 
#            path_dataset=config.DATA_SYNTHETIC_NORMAL_FULL_WSP_TESTING_DIR, 
#            path_results=config.RESULTS_SYNTHETIC_FULL_MRCNN_DIR, 
#            model_path=evaluate_algorithms_config.DROPLET_MRCNN_MODEL)

# real full
# main_mrcnn_full(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
#            fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, 
#            fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
           
#            path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_FULL_DATASET_MRCNN, 
#            path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_FULL_DATASET_MRCNN, 
#            path_dataset=config.DATA_REAL_WSP_TESTING_DIR2, 
#            path_results=config.RESULTS_REAL_FULL_MRCNN_DIR, 
#            model_path=evaluate_algorithms_config.DROPLET_MRCNN_MODEL)
