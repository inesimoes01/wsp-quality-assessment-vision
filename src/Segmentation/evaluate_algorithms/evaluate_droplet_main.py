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
from joblib import Parallel, delayed
from matplotlib import pyplot as plt 
from shapely import Polygon

sys.path.insert(0, 'src')
import Common.config as config 
import evaluate_algorithms_config
from Common.Statistics import Statistics as stats

isDropletCCV_Full, isDropletCCV_Square = False, False
isDropletYOLO_Full, isDropletYOLO_Square = False, False
isDropletMRCNN_Full, isDropletMRCNN_Square = True, True
isDropletCELLPOSE_Full, isDropletCELLPOSE_Square = False, False

IOU_THRESHOLD, DISTANCE_THRESHOLD = 0.5, 20

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

def match_predicted_to_groundtruth(predicted_polygons, ground_truths, distance_threshold, image):
    matched_indices = []
    results = []

    for index_pred, (predicted_polygon, pred_center) in enumerate(predicted_polygons):
        best_iou, best_match_index = 0, 0
        mask_predicted = np.zeros_like(image)
        cv2.drawContours(mask_predicted, [predicted_polygon], -1, 255, cv2.FILLED)
    
        for index_gt, (ground_truth_polygon, gt_center) in enumerate(ground_truths):
            if index_gt in matched_indices:
                continue

            distance = np.linalg.norm(np.array(pred_center) - np.array(gt_center))

            if distance < distance_threshold:
                mask_groundtruth = np.zeros_like(image)
                cv2.drawContours(mask_groundtruth, [ground_truth_polygon], -1, 255, cv2.FILLED)

                iou = calculate_iou(mask_predicted, mask_groundtruth)

                if iou > best_iou:
                    best_iou = iou
                    best_match_index = index_gt
                    

        if best_iou > 0:
            results.append((predicted_polygon, ground_truth_polygon, best_match_index, best_iou))
            matched_indices.append(best_match_index)
           

    return results

def calculate_avg_precision(precision, recall):
    precisions = []
    recalls = []
    
    precisions.append(precision)
    recalls.append(recall)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
  
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    return avg_prec

def calculate_iou(mask1, mask2):
    intersection_sum = np.sum(np.logical_and(mask1, mask2))
    union_sum = np.sum(np.logical_or(mask1, mask2))

    if union_sum == 0: iou = 0  
    else: iou = intersection_sum / union_sum

    return iou

def calculate_metrics(ground_truth_masks, results, image_correct_predictions, iou_threshold):
    tp = 0
    fp = 0
    fn = 0
    
    matches_indices = []
    
    # calculate confusion matrix values and draw it in final image
    for predicted_polygon, ground_truth_polygon, best_match_index, best_iou in results:
        if best_match_index is not None:
            matches_indices.append(best_match_index)

            if best_iou > iou_threshold:
                tp += 1
                cv2.drawContours(image_correct_predictions, [predicted_polygon], -1, (0, 255, 0), 1)
    
            else: fp += 1

    unmatched_ground_truths = [ground_truth_masks[i] for i in range(len(ground_truth_masks)) if i not in matches_indices]
    fn = len(unmatched_ground_truths)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return image_correct_predictions, precision, recall, f1_score, tp, fp, fn

def evaluate_matches(ground_truth_masks, results, image_correct_predictions, iou_threshold = 0.5):
    # calculate precision for the agreed iou threshold
    image_correct_predictions, precision, recall, f1_score, tp, fp, fn = calculate_metrics(ground_truth_masks, results, image_correct_predictions, iou_threshold)

    # calculate map05
    _, p05, r05, _, _, _, _ = calculate_metrics(ground_truth_masks, results, image_correct_predictions, 0.5)
    map05 = calculate_avg_precision(p05, r05)

    # calculate map05095
    avg_precs = []
    iou_thrs = []

    for iou_thr in np.linspace(0.5, 0.95, 10):
        _, p0595, r0595, _, _, _, _ = calculate_metrics(ground_truth_masks, results, image_correct_predictions, iou_thr)
        avg_precision = calculate_avg_precision(p0595, r0595)
        
        avg_precs.append(avg_precision)
        iou_thrs.append(iou_thr)

    map0595 = np.mean(avg_precs)

    return image_correct_predictions, precision, recall, f1_score, map05, map0595, tp, fp, fn

def read_groundtruth_statistics_file(statistics_file_path):
    data = pd.read_csv(statistics_file_path)

    if 'GroundTruth' in data.columns:
        groundtruth_col = 'GroundTruth'
    elif 'Groundtruth' in data.columns:
        groundtruth_col = 'Groundtruth'
    else:
        raise KeyError("Neither 'GroundTruth' nor 'Groundtruth' found in the columns")

    # Now, use this column name to extract the values
    vmd_value = data.at[0, groundtruth_col]
    rsf_value = data.at[1, groundtruth_col]
    coverage_percentage = data.at[2, groundtruth_col]
    no_total_droplets = data.at[3, groundtruth_col]
    overlapped_percentage = data.at[4, groundtruth_col]
    no_overlapped_droplets = data.at[5, groundtruth_col]

            
    stats_groundtruth = stats(vmd_value, rsf_value, coverage_percentage, no_total_droplets, no_overlapped_droplets, overlapped_percentage, None)
    return stats_groundtruth

def calculate_predicted_statistics(predicted_shapes, width_px, height_px, width_mm):
    total_droplet_area = 0
    total_no_droplets = 0
    list_droplet_area = []
    list_polygons = []
    
    # get general information from the polygons
    predicted_shapes = [item[0] for item in predicted_shapes]
    for pred in predicted_shapes:
        polygon = Polygon(pred)
        if polygon.area < 10000:
            list_polygons.append(polygon)
            polygon_area = polygon.area

            total_droplet_area += polygon_area
            list_droplet_area.append(polygon_area)
            total_no_droplets += 1

    # get list of the droplet diameters
    diameter_list = sorted(stats.area_to_diameter_micro(list_droplet_area, width_px, width_mm))
    
    # find the overlapping polygons
    no_droplets_overlapped = 0
    overlapping_polygons = []
    for i, polygon in enumerate(list_polygons):
        if not polygon.is_valid:
            polygon = polygon.buffer(0)  # Attempt to fix invalid polygon
        for j, other_polygon in enumerate(list_polygons):
            if i != j:
                if not other_polygon.is_valid:
                    other_polygon = other_polygon.buffer(0)  # Attempt to fix invalid other polygon
                if polygon.intersects(other_polygon):
                    overlapping_polygons.append(i)
                    overlapping_polygons.append(j)

    no_droplets_overlapped = len(overlapping_polygons)
    overlaped_percentage = no_droplets_overlapped / total_no_droplets * 100 if total_no_droplets > 0 else 0

    # calculate statistics
    vmd_value, coverage_percentage, rsf_value, _ = stats.calculate_statistics(diameter_list, height_px * width_px, total_droplet_area)    
    predicted_stats = stats(vmd_value, rsf_value, coverage_percentage, total_no_droplets, no_droplets_overlapped, overlaped_percentage)
    
    return predicted_stats

def save_predicted_statistics_file(predicted_statistics_file, predicted_stats, predicted_statistics_fieldnames):
    data = {
        '': predicted_statistics_fieldnames,
        'Predicted': [predicted_stats.vmd_value, predicted_stats.rsf_value, predicted_stats.coverage_percentage, predicted_stats.no_droplets, predicted_stats.overlaped_percentage, predicted_stats.no_droplets_overlapped], 
    }
    df = pd.DataFrame(data)
    df.to_csv(predicted_statistics_file, index=False, float_format='%.2f')

def write_segmentation_csv_file(metric, path_csv_segmentation, fieldnames_segmentation):
    with open(path_csv_segmentation, mode='a', newline='') as file:
        new_row = {
                fieldnames_segmentation[0]: metric[0], 
                fieldnames_segmentation[1]: metric[1], 
                fieldnames_segmentation[2]: metric[2], 
                fieldnames_segmentation[3]: metric[3], 
                fieldnames_segmentation[4]: metric[4], 
                fieldnames_segmentation[5]: metric[5], 
                fieldnames_segmentation[6]: metric[6], 
                fieldnames_segmentation[7]: metric[7], 
                fieldnames_segmentation[8]: metric[8], 
                fieldnames_segmentation[9]: metric[9]
            }
        writer = csv.DictWriter(file, fieldnames=fieldnames_segmentation)
        writer.writerow(new_row)

def write_statistics_csv_file(filename, predicted_stats:stats, groundtruth_stats:stats, path_statistics, fieldname_statistics):
    
    with open(path_statistics, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldname_statistics)
        
        new_row = {
            fieldname_statistics[0]: filename, 

            # VMD
            fieldname_statistics[1]: predicted_stats.vmd_value, 
            fieldname_statistics[2]: groundtruth_stats.vmd_value, 
            fieldname_statistics[3]: abs((predicted_stats.vmd_value - groundtruth_stats.vmd_value) / groundtruth_stats.vmd_value), 
            
            # RSF
            fieldname_statistics[4]: predicted_stats.rsf_value, 
            fieldname_statistics[5]: groundtruth_stats.rsf_value, 
            fieldname_statistics[6]: abs((predicted_stats.rsf_value - groundtruth_stats.rsf_value) / groundtruth_stats.rsf_value), 
            
            # COVERAGE PERCENTAGE
            fieldname_statistics[7]: predicted_stats.coverage_percentage, 
            fieldname_statistics[8]: groundtruth_stats.coverage_percentage, 
            fieldname_statistics[9]:abs((predicted_stats.coverage_percentage - groundtruth_stats.coverage_percentage) / groundtruth_stats.coverage_percentage), 
            
            # TOTAL NO DROPLETS
            fieldname_statistics[10]: predicted_stats.no_droplets, 
            fieldname_statistics[11]: groundtruth_stats.no_droplets, 
            fieldname_statistics[12]: abs((predicted_stats.no_droplets - groundtruth_stats.no_droplets) / groundtruth_stats.no_droplets), 
            
            # NO OVERLAPPED DROPLETS
            fieldname_statistics[13]: predicted_stats.no_droplets_overlapped, 
            fieldname_statistics[14]: groundtruth_stats.no_droplets_overlapped, 
            fieldname_statistics[15]: abs((predicted_stats.no_droplets_overlapped - groundtruth_stats.no_droplets_overlapped) / groundtruth_stats.no_droplets_overlapped) if groundtruth_stats.no_droplets_overlapped > 0 else 0,
            
            # PERCENTAGE OVERLAPPED DROPLETS
            fieldname_statistics[16]: predicted_stats.overlaped_percentage, 
            fieldname_statistics[17]: groundtruth_stats.overlaped_percentage, 
            fieldname_statistics[18]: abs((predicted_stats.overlaped_percentage - groundtruth_stats.overlaped_percentage) / groundtruth_stats.overlaped_percentage) if groundtruth_stats.overlaped_percentage > 0 else 0
        }
        writer.writerow(new_row)

def calculate_centroid(polygon):
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    centroid_x = sum(x_coords) / len(polygon)
    centroid_y = sum(y_coords) / len(polygon)
    return centroid_x, centroid_y

def get_polygon_from_yolo_label(label_file_path, width, height, x_offset = 0, y_offset = 0):
    polygons = []
    
    with open(label_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()            
            coordinates = list(map(float, parts[1:]))
            
            if len(coordinates) >= 8:
                polygon = [(coordinates[i] * width + x_offset, coordinates[i+1] * height + y_offset) for i in range(0, len(coordinates), 2)]
                contour = np.array(polygon, dtype=np.int32)

                polygons.append(contour)
    

    # sort the polygons
    polygons_with_centroids = [(polygon, calculate_centroid(polygon)) for polygon in polygons]
    sorted_polygons = sorted(polygons_with_centroids, key=lambda item: (item[1][0], item[1][1]))
        
    return sorted_polygons

def main_evaluation(fieldnames_segmentation, fieldnames_statistics, fieldnames_time, 
                    fieldnames_predicted_statistics, path_csv_segmentation, path_csv_statistics, 
                    path_dataset, path_results, iou_threshold, distance_threshold, width_mm = 0):
    directory_image = os.path.join(path_dataset, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
    directory_label = os.path.join(path_dataset, config.DATA_GENERAL_LABEL_FOLDER_NAME)
    directory_stats = os.path.join(path_dataset, config.DATA_GENERAL_STATS_FOLDER_NAME)

    # create segmentation and statistics evaluation csv files
    with open(path_csv_segmentation, mode='w', newline='') as file:
        csv.DictWriter(file, fieldnames=fieldnames_segmentation).writeheader()
    with open(path_csv_statistics, mode='w', newline='') as file:
        csv.DictWriter(file, fieldnames=fieldnames_statistics).writeheader()

    alternative_path = os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME)

    for file in os.listdir(directory_image): 
        start_time = time.time()
        filename_with_real_width = file.split(".")[0]


        if width_mm == 0:
            filename, numerical_part = filename_with_real_width.split('-')
            width_mm = float(numerical_part.replace("_", "."))
       
        image_path = os.path.join(directory_image, filename_with_real_width + ".png")
        predicted_label_path = os.path.join(path_results, config.DATA_GENERAL_LABEL_FOLDER_NAME, filename_with_real_width + ".txt")
        groundtruth_label_path = os.path.join(path_dataset, config.DATA_GENERAL_LABEL_FOLDER_NAME, filename_with_real_width + ".txt")
        time_csv_path = os.path.join(path_results, config.RESULTS_GENERAL_SEGMENTATIONTIME_FOLDER_NAME + ".csv")
        groundtruth_stats_path = (os.path.join(directory_stats, filename_with_real_width + ".csv"))
        predicted_stats_path = (os.path.join(path_results, config.DATA_GENERAL_STATS_FOLDER_NAME, filename_with_real_width + ".csv"))
        total_predicted_image = (os.path.join(path_results, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME, filename_with_real_width + "_predictions.png"))
        correct_predicted_image = (os.path.join(path_results, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME, filename_with_real_width + "_correct_predictions.png"))
        groundtruth_image = (os.path.join(path_results, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME, filename_with_real_width + ".png"))
        
        # get images
        image_colors = cv2.imread(os.path.join(directory_image, filename_with_real_width + ".png"))  
        image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
        height, width = image_colors.shape[:2]

        # image to show the correct predictions 
        image_correct_predictions = copy.copy(image_colors)
        image_predictions = copy.copy(image_colors)
        image_groundtruth = copy.copy(image_colors)

        df = pd.read_csv(time_csv_path)


        print("Evaluating image", filename_with_real_width + "..." )

        try:
            ####### EVALUATE SEGMENTATION
            # get predicted shapes given the yolo label file
            if os.path.exists(predicted_label_path):
                predicted_shapes = get_polygon_from_yolo_label(predicted_label_path, width, height)
                
                # get groundtruth shapes given the yolo label file
                groundtruth_shapes = get_polygon_from_yolo_label(groundtruth_label_path, width, height)

                # match the groundtruth and the predicted shapes to find correct predictions given the iou threshold
                results = match_predicted_to_groundtruth(predicted_shapes, groundtruth_shapes, distance_threshold, image_colors)
                image_correct_predictions, precision, recall, f1_score, map5, map595, tp, fp, fn = evaluate_matches(groundtruth_shapes, results, image_correct_predictions, iou_threshold)

                # get segmentation time value from the csv file created previously
                if filename_with_real_width in df[fieldnames_time[0]].astype(str).values:
                    filtered_df = df[df[fieldnames_time[0]].astype(str) == str(filename_with_real_width)]
                    segmentation_time = filtered_df[fieldnames_time[1]].values[0]
                else:
                    segmentation_time = 0

                # update metrics to the segmentation file
                write_segmentation_csv_file((filename_with_real_width, precision, recall, f1_score, map5, map595, tp, fp, fn, segmentation_time), path_csv_segmentation, fieldnames_segmentation)
                
                ####### EVALUATE STATISTICS

                # calculate predicted statistics 
                predicted_statistics = calculate_predicted_statistics(predicted_shapes, width, height, width_mm)
                
                # get groundtruth statistics (ATTENTION !!!! SYNTHETIC DATASET WAS CREATED WITH OVERLAPPED PERCENTAGE AND OVERLAPPED NUMBER SWITCHED)
                groundtruth_statistics = read_groundtruth_statistics_file(groundtruth_stats_path)

                # update metrics to the statistics file
                write_statistics_csv_file(filename_with_real_width, predicted_statistics, groundtruth_statistics, path_csv_statistics, fieldnames_statistics)

                ####### SAVE PREDICTED RESULTS FILES

                # save predicted statistics file
                save_predicted_statistics_file(predicted_stats_path, predicted_statistics, fieldnames_predicted_statistics)

                # save predicted shapes image
                image_predictions = cv2.cvtColor(image_predictions, cv2.COLOR_BGR2RGB)
                image_correct_predictions = cv2.cvtColor(image_correct_predictions, cv2.COLOR_BGR2RGB)
                image_groundtruth = cv2.cvtColor(image_groundtruth, cv2.COLOR_BGR2RGB)
                #cv2.imwrite(total_predicted_image, image_predictions)
                cv2.imwrite(correct_predicted_image, image_correct_predictions)
                #cv2.imwrite(groundtruth_image, image_groundtruth)

                end_time = time.time()
                elapsed_time = end_time - start_time
                
                print("Time taken:", elapsed_time, "seconds")
    
                #cv2.imwrite(os.path.join(path_results, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME, filename + "_correct_predictions.png"), image_correct_predictions)

        except np.core._exceptions._ArrayMemoryError as e:
            print(f"Memory error encountered while processing {filename}: {e}")
    
    return image_correct_predictions

def update_general_evaluation_droplet_stats(path_general_evaluation, path_individual_evaluation, method, fieldnames_general, fieldnames):
    df = pd.read_csv(path_individual_evaluation)

    # for field in [fieldnames[3], fieldnames[6], fieldnames[9], fieldnames[12]]:
    #     if df[field] == "nan":
    #         df[field] = 0

    average_df = pd.DataFrame([{
        fieldnames_general[0]: method,

        # VMD, RSF, COVERAGE_PERCENTAGE, NO_DROPLETS
        fieldnames_general[1]: df[fieldnames[3]].mean(),
        fieldnames_general[2]: df[fieldnames[6]].mean(),
        fieldnames_general[3]: df[fieldnames[9]].mean(),
        fieldnames_general[4]: df[fieldnames[12]].mean(),
        #'OtherCoveragePercentage_error': df['OtherCoveragePercentage_error'].mean(),
    
        fieldnames_general[5]: df[fieldnames[3]].median(),
        fieldnames_general[6]: df[fieldnames[6]].median(),
        fieldnames_general[7]: df[fieldnames[9]].median(),
        fieldnames_general[8]: df[fieldnames[12]].median(),
        #'OtherCoveragePercentage_median': df['OtherCoveragePercentage_error'].median(),
        
        fieldnames_general[9]: df[fieldnames[3]].std(),
        fieldnames_general[10]: df[fieldnames[6]].std(),
        fieldnames_general[11]: df[fieldnames[9]].std(),
        fieldnames_general[12]: df[fieldnames[12]].std(),
        #'OtherCoveragePercentage_std': df['OtherCoveragePercentage_error'].std(),

        fieldnames_general[13]: df[fieldnames[3]].max(),
        fieldnames_general[14]: df[fieldnames[6]].max(),
        fieldnames_general[15]: df[fieldnames[9]].max(),
        fieldnames_general[16]: df[fieldnames[12]].max(),
        #OtherCoveragePercentage_max': df['OtherCoveragePercentage_error'].max(),
    }])

    df_gen = pd.read_csv(path_general_evaluation)

    # # check if there's a row with the same 'method' (assumed to be in the first column)
    # if method in df_gen[fieldnames_general[0]].values:
    #     # if method already exists, update the existing row
    #     df_gen.loc[df_gen[fieldnames_general[0]] == method, :] = average_df.iloc
    # else:
        # if method does not exist, append the new row
    df_gen = pd.concat([df_gen, average_df], ignore_index=True)

    df_gen.to_csv(path_general_evaluation, index=False)

def update_general_evaluation_droplet_segm(path_general_evaluation, path_individual_evaluation, method, fieldnames_general, fieldnames):
    df = pd.read_csv(path_individual_evaluation)

    tp = df[fieldnames[6]].sum()
    fp = df[fieldnames[7]].sum()
    fn = df[fieldnames[8]].sum()

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    average_df = pd.DataFrame([{
        fieldnames_general[0]: method,
        fieldnames_general[1]: precision,
        fieldnames_general[2]: recall,
        fieldnames_general[3]: f1_score,
        fieldnames_general[4]: df[fieldnames[4]].mean(),
        fieldnames_general[5]: df[fieldnames[5]].mean(),
        fieldnames_general[6]: tp,
        fieldnames_general[7]: fp,
        fieldnames_general[8]: fn,
        fieldnames_general[9]: df[fieldnames[9]].mean()
    }])

    df_gen = pd.read_csv(path_general_evaluation)

    # check if there's a row with the same 'method' (assumed to be in the first column)
    # if method in df_gen[fieldnames_general[0]].values:
    #     # if method already exists, update the existing row
    #     df_gen.loc[df_gen[fieldnames_general[0]] == method, :] = average_df.iloc
    # else:
        # if method does not exist, append the new row
    df_gen = pd.concat([df_gen, average_df], ignore_index=True)

    df_gen.to_csv(path_general_evaluation, index=False)

def update_general_evaluation_paper(path_general_evaluation, path_individual_evaluation, method, fieldnames_general, fieldnames):
    df = pd.read_csv(path_individual_evaluation)

    average_df = pd.DataFrame([{
        fieldnames[0]: method,
        fieldnames[1]: df[fieldnames[1]].median(),
        fieldnames[2]: df[fieldnames[2]].mean()
    }])

    df_gen = pd.read_csv(path_general_evaluation)

    # check if there's a row with the same 'method' (assumed to be in the first column)
    if method in df_gen[fieldnames_general[0]].values:
        # if method already exists, update the existing row
        df_gen.loc[df_gen[fieldnames_general[0]] == method, :] = average_df.iloc[0]
    else:
        # if method does not exist, append the new row
        df_gen = pd.concat([df_gen, average_df], ignore_index=True)

    df_gen.to_csv(path_general_evaluation, index=False)

def new_csv_file(path_to_new_csv, new_csv_fieldnames):
    with open(path_to_new_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=new_csv_fieldnames)
        writer.writeheader()

def calculate_metrics_for_resolution(df, method, fieldnames, fieldnames_general, path_general_evaluation):
    # Sum the values for tp, fp, fn
    tp = df[fieldnames[6]].sum()
    fp = df[fieldnames[7]].sum()
    fn = df[fieldnames[8]].sum()

    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Create a DataFrame for the results
    average_df = pd.DataFrame([{
        fieldnames_general[0]: method,
        fieldnames_general[1]: precision,
        fieldnames_general[2]: recall,
        fieldnames_general[3]: f1_score,
        fieldnames_general[4]: df[fieldnames[4]].mean(),
        fieldnames_general[5]: df[fieldnames[5]].mean(),
        fieldnames_general[6]: tp,
        fieldnames_general[7]: fp,
        fieldnames_general[8]: fn,
        fieldnames_general[9]: df[fieldnames[9]].mean()
    }])

    # Load the general evaluation file
    df_gen = pd.read_csv(path_general_evaluation)

    # Concatenate the new results to the general evaluation DataFrame
    df_gen = pd.concat([df_gen, average_df], ignore_index=True)

    # Save the updated general evaluation to CSV
    df_gen.to_csv(path_general_evaluation, index=False)
def save_shapes_to_yolo_label(label_path, droplets_detected, width, height):
    with open(label_path, "w") as file:
        for droplet, centroid in droplets_detected:
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

def divide_evaluations_by_image_resolution(path_individual_evaluation, path_general_evaluation, fieldnames_individual, fieldnames_general, method):
    df = pd.read_csv(path_individual_evaluation)

    # Create filters for low, medium, and high resolution
    df[fieldnames_individual[0]] = df[fieldnames_individual[0]].apply(lambda x: int(x.split('-')[0].split('_')[0]))  

    # Filter for low resolution images (0 <= num <= 49)
    low_res_df = (df[(int(df[fieldnames_individual[0]]) >= 0) & (int(df[fieldnames_individual[0]]) <= 49)]) | (df[(int(df[fieldnames_individual[0]]) >= 150) & (int(df[fieldnames_individual[0]]) <= 199)])
    if not low_res_df.empty:
        calculate_metrics_for_resolution(low_res_df, method + "_lowresolution", fieldnames_individual, fieldnames_general, path_general_evaluation)

    # Filter for medium resolution images (50 <= num <= 99)
    medium_res_df = (df[(df[fieldnames_individual[0]] >= 50) & (df[fieldnames_individual[0]] <= 99)]) | (df[(df[fieldnames_individual[0]] >= 200) & (df[fieldnames_individual[0]] <= 249)])
    if not medium_res_df.empty:
        calculate_metrics_for_resolution(medium_res_df, method + "_medium_resolution", fieldnames_individual, fieldnames_general, path_general_evaluation)

    # Filter for high resolution images (100 <= num <= 149)
    high_res_df = (df[(df[fieldnames_individual[0]] >= 100) & (df[fieldnames_individual[0]] <= 149)]) | (df[(df[fieldnames_individual[0]] >= 250) & (df[fieldnames_individual[0]] <= 299)])
    if not high_res_df.empty:
        calculate_metrics_for_resolution(high_res_df, method + "_highresolution", fieldnames_individual, fieldnames_general, path_general_evaluation)

def group_images_by_prefix(directory_image):
    image_groups = {}
    
    for file in os.listdir(directory_image):
        filename = file.split(".")[0]
        prefix = filename.split("-")[0]
        full_image_name = "_".join(prefix.split("_")[0:-2])
        x_offset, y_offset = int(prefix.split("_")[-2]), int(prefix.split("_")[-1])
        
        if full_image_name not in image_groups:
            image_groups[full_image_name] = []
        
        image_groups[full_image_name].append((filename, x_offset, y_offset))
    
    return image_groups

def main_evaluation_full_image(fieldnames_segmentation, fieldnames_statistics, fieldnames_time, 
                    fieldnames_predicted_statistics, path_csv_segmentation, path_csv_statistics, 
                    path_dataset, path_results, iou_threshold, distance_threshold, square_image_path, path_real_results, width_mm = 0):
    directory_image = os.path.join(path_dataset, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
    directory_label = os.path.join(path_dataset, config.DATA_GENERAL_LABEL_FOLDER_NAME)
    directory_stats = os.path.join(path_dataset, config.DATA_GENERAL_STATS_FOLDER_NAME)

    directory_label_square = os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME)
    directory_image_square = os.path.join(square_image_path, config.DATA_GENERAL_IMAGE_FOLDER_NAME)

    # create segmentation and statistics evaluation csv files
    with open(path_csv_segmentation, mode='w', newline='') as file:
        csv.DictWriter(file, fieldnames=fieldnames_segmentation).writeheader()
    with open(path_csv_statistics, mode='w', newline='') as file:
        csv.DictWriter(file, fieldnames=fieldnames_statistics).writeheader()

    image_groups = group_images_by_prefix(directory_image_square)

    for full_image_name, image_group in image_groups.items(): 
        
        start_time = time.time()
        #filename = file.split(".")[0]

        full_image_path = os.path.join(directory_image, full_image_name + ".png")
        image_colors = cv2.imread(full_image_path)  
        #image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
        height, width = image_colors.shape[:2]

        image_path = os.path.join(directory_image, full_image_name + ".png")
        predicted_label_path_full = os.path.join(path_real_results, config.DATA_GENERAL_LABEL_FOLDER_NAME, full_image_name + ".txt")
        groundtruth_label_path = os.path.join(path_dataset, config.DATA_GENERAL_LABEL_FOLDER_NAME, full_image_name + ".txt")

        time_csv_path = os.path.join(path_results, config.RESULTS_GENERAL_SEGMENTATIONTIME_FOLDER_NAME + ".csv")
        
        groundtruth_stats_path = (os.path.join(directory_stats, full_image_name + ".csv"))
        predicted_stats_path = (os.path.join(path_results, config.DATA_GENERAL_STATS_FOLDER_NAME, full_image_name + ".csv"))
        
        #total_predicted_image = (os.path.join(path_results, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME, filename + "_predictions.png"))
        correct_predicted_image = (os.path.join(path_results, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME, full_image_name + "_correct_predictions.png"))
        #groundtruth_image = (os.path.join(path_results, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME, filename + ".png"))
        
        # get images
        image_colors = cv2.imread(os.path.join(directory_image, full_image_name + ".png"))  
        image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
        height, width = image_colors.shape[:2]

        # image to show the correct predictions 
        image_correct_predictions = copy.copy(image_colors)
        image_predictions = copy.copy(image_colors)
        image_groundtruth = copy.copy(image_colors)

        df = pd.read_csv(time_csv_path)

        print("Evaluating image", full_image_name + "..." )

        predicted_shapes = []
        segmentation_time = 0

        for (square_filename, x_offset, y_offset) in image_group: 
        
            predicted_label_path = os.path.join(path_results, config.DATA_GENERAL_LABEL_FOLDER_NAME, square_filename + ".txt")
            predicted_image_path = os.path.join(square_image_path, config.DATA_GENERAL_IMAGE_FOLDER_NAME, square_filename + ".png")
            
            if os.path.exists(predicted_label_path):
                height_square, width_square = cv2.imread(predicted_image_path).shape[:2]
                predicted_shapes.extend(get_polygon_from_yolo_label(predicted_label_path, width_square, height_square, x_offset, y_offset))

                if square_filename in df[fieldnames_time[0]].astype(str).values:
                    filtered_df = df[df[fieldnames_time[0]].astype(str) == str(square_filename)]
                    segmentation_time += filtered_df[fieldnames_time[1]].values[0]
                else:
                    segmentation_time = 0

        save_shapes_to_yolo_label(predicted_label_path_full, predicted_shapes, width, height)

        groundtruth_shapes = get_polygon_from_yolo_label(groundtruth_label_path, width, height)

        # match the groundtruth and the predicted shapes to find correct predictions given the iou threshold
        results = match_predicted_to_groundtruth(predicted_shapes, groundtruth_shapes, distance_threshold, image_colors)
        image_correct_predictions, precision, recall, f1_score, map5, map595, tp, fp, fn = evaluate_matches(groundtruth_shapes, results, image_correct_predictions, iou_threshold)


        # update metrics to the segmentation file
        write_segmentation_csv_file((full_image_name, precision, recall, f1_score, map5, map595, tp, fp, fn, segmentation_time), path_csv_segmentation, fieldnames_segmentation)
        
        ####### EVALUATE STATISTICS

        # calculate predicted statistics 
        predicted_statistics = calculate_predicted_statistics(predicted_shapes, width, height, width_mm)
        
        # get groundtruth statistics (ATTENTION !!!! SYNTHETIC DATASET WAS CREATED WITH OVERLAPPED PERCENTAGE AND OVERLAPPED NUMBER SWITCHED)
        groundtruth_statistics = read_groundtruth_statistics_file(groundtruth_stats_path)

        # update metrics to the statistics file
        write_statistics_csv_file(full_image_name, predicted_statistics, groundtruth_statistics, path_csv_statistics, fieldnames_statistics)

        ####### SAVE PREDICTED RESULTS FILES

        # save predicted statistics file
        save_predicted_statistics_file(predicted_stats_path, predicted_statistics, fieldnames_predicted_statistics)

        # save predicted shapes image
        image_predictions = cv2.cvtColor(image_predictions, cv2.COLOR_BGR2RGB)
        image_correct_predictions = cv2.cvtColor(image_correct_predictions, cv2.COLOR_BGR2RGB)
        image_groundtruth = cv2.cvtColor(image_groundtruth, cv2.COLOR_BGR2RGB)
        
        cv2.imwrite(correct_predicted_image, image_correct_predictions)
       

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("Time taken:", elapsed_time, "seconds")
        
    return image_correct_predictions

def compute_evaluations():
    if isDropletCCV_Square:
        main_evaluation(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION,
                        fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
                        fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
                        fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

                        path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_CV,
                        path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_CV,
                        path_dataset=config.DATA_SYNTHETIC_WSP_TESTING_DIR, 
                        path_results=config.RESULTS_SYNTHETIC_CCV_DIR,
                        iou_threshold=IOU_THRESHOLD,
                        distance_threshold=DISTANCE_THRESHOLD)
        

        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
                                               evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_CV, 
                                               "droplet_synthetic_square_dataset_ccv",
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
                                                evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_CV, 
                                                "droplet_synthetic_square_dataset_ccv",
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS) 
    
    
        main_evaluation(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION,
                        fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
                        fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
                        fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

                        path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_CV,
                        path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_CV,
                        path_dataset=config.DATA_REAL_WSP_TESTING_DIR, 
                        path_results=config.RESULTS_REAL_CCV_DIR,
                        iou_threshold=IOU_THRESHOLD,
                        distance_threshold=DISTANCE_THRESHOLD)
    
        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
                                               evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_CV, 
                                               "droplet_real_square_dataset_ccv", 
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
                                                evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_CV, 
                                                "droplet_real_square_dataset_ccv",
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS)
        
    if isDropletCCV_Full:
        # main_evaluation(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
        #                 fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
        #                 fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
        #                 fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

        #                 path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_FULL_DATASET_CV,
        #                 path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_FULL_DATASET_CV,
        #                 path_dataset=config.DATA_SYNTHETIC_FULL_WSP_TESTING_DIR, 
        #                 path_results=config.RESULTS_SYNTHETIC_FULL_CCV_DIR,
        #                 iou_threshold=IOU_THRESHOLD,
        #                 distance_threshold=DISTANCE_THRESHOLD,
        #                 width_mm=76)
        

        # update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
        #                                        evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_FULL_DATASET_CV, 
        #                                        "droplet_synthetic_full_dataset_ccv",
        #                                        evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
        #                                        evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        
        # update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
        #                                         evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_FULL_DATASET_CV, 
        #                                         "droplet_synthetic_full_dataset_ccv",
        #                                         evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
        #                                         evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS) 
    
    
        main_evaluation(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                        fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
                        fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
                        fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

                        path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_FULL_DATASET_CV,
                        path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_FULL_DATASET_CV,
                        path_dataset=config.DATA_REAL_FULL_WSP_TESTING_DIR, 
                        path_results=config.RESULTS_REAL_FULL_CCV_DIR,
                        iou_threshold=IOU_THRESHOLD,
                        distance_threshold=DISTANCE_THRESHOLD,
                        width_mm=26)
    
        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
                                               evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_FULL_DATASET_CV, 
                                               "droplet_real_full_dataset_ccv", 
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
                                                evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_FULL_DATASET_CV, 
                                                "droplet_real_full_dataset_ccv",
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS)
    
    if isDropletMRCNN_Square:
        # REAL
        main_evaluation(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                        fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
                        fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
                        fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

                        path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_MRCNN,
                        path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_MRCNN,
                        path_dataset=config.DATA_REAL_WSP_TESTING_DIR2, 
                        path_results=config.RESULTS_REAL_MRCNN_DIR,
                        iou_threshold=IOU_THRESHOLD,
                        distance_threshold=DISTANCE_THRESHOLD)
        
        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
                                               evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_MRCNN, 
                                               "droplet_real_square_dataset_mrcnn", 
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
                                                evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_MRCNN, 
                                                "droplet_real_square_dataset_mrcnn",
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS)

        # SYNTHETIC
        main_evaluation(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                        fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
                        fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
                        fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

                        path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_MRCNN,
                        path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_MRCNN,
                        path_dataset=config.DATA_SYNTHETIC_WSP_TESTING_DIR, 
                        path_results=config.RESULTS_SYNTHETIC_MRCNN_DIR,
                        iou_threshold=IOU_THRESHOLD,
                        distance_threshold=DISTANCE_THRESHOLD)
        
        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
                                               evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_MRCNN, 
                                               "droplet_synthetic_square_dataset_mrcnn",
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
                                                evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_MRCNN, 
                                                "droplet_synthetic_square_dataset_mrcnn",
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS) 
    
    if isDropletMRCNN_Full:
        # REAL
        main_evaluation_full_image(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                        fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
                        fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
                        fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

                        path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_FULL_DATASET_MRCNN,
                        path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_FULL_DATASET_MRCNN,
                        path_dataset=config.DATA_REAL_FULL_WSP_TESTING_DIR, 
                        path_results=config.RESULTS_REAL_MRCNN_DIR,
                        iou_threshold=IOU_THRESHOLD,
                        distance_threshold=DISTANCE_THRESHOLD, 
                        square_image_path=config.DATA_REAL_WSP_TESTING_DIR2,
                        path_real_results = config.RESULTS_REAL_FULL_MRCNN_DIR,
                        width_mm=26)
        
        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
                                               evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_FULL_DATASET_MRCNN, 
                                               "droplet_real_full_dataset_mrcnn", 
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
                                                evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_FULL_DATASET_MRCNN, 
                                                "droplet_real_full_dataset_mrcnn",
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS)

        #SYNTHETIC
        main_evaluation_full_image(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                        fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
                        fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
                        fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

                        path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_FULL_DATASET_MRCNN,
                        path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_FULL_DATASET_MRCNN,
                        path_dataset=config.DATA_SYNTHETIC_FULL_WSP_TESTING_DIR, 
                        path_results=config.RESULTS_SYNTHETIC_MRCNN_DIR,
                        iou_threshold=IOU_THRESHOLD,
                        distance_threshold=DISTANCE_THRESHOLD, 
                        square_image_path=config.DATA_SYNTHETIC_WSP_TESTING_DIR,
                        path_real_results = config.RESULTS_SYNTHETIC_FULL_MRCNN_DIR,
                        width_mm=76)
        
        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
                                               evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_FULL_DATASET_MRCNN, 
                                               "droplet_synthetic_full_dataset_mrcnn",
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
                                                evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_FULL_DATASET_MRCNN, 
                                                "droplet_synthetic_full_dataset_mrcnn",
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS) 
    
    if isDropletYOLO_Square:

        # main_evaluation(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
        #                 fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
        #                 fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
        #                 fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

        #                 path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO,
        #                 path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO,
        #                 path_dataset=config.DATA_SYNTHETIC_WSP_TESTING_DIR, 
        #                 path_results=config.RESULTS_SYNTHETIC_YOLO_DIR,
        #                 iou_threshold=IOU_THRESHOLD,
        #                 distance_threshold=DISTANCE_THRESHOLD)
        
        # update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
        #                                        evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO, 
        #                                        "droplet_synthetic_square_dataset_yolo",
        #                                        evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
        #                                        evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        # update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
        #                                         evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO, 
        #                                         "droplet_synthetic_square_dataset_yolo",
        #                                         evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
        #                                         evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS) 
        
        main_evaluation(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                        fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
                        fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
                        fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

                        path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_YOLO,
                        path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_YOLO,
                        path_dataset=config.DATA_REAL_WSP_TESTING_DIR2, 
                        path_results=config.RESULTS_REAL_YOLO_DIR,
                        iou_threshold=IOU_THRESHOLD,
                        distance_threshold=DISTANCE_THRESHOLD)

        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
                                               evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_YOLO, 
                                               "droplet_real_square_dataset_yolo", 
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
                                                evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_YOLO, 
                                                "droplet_real_square_dataset_yolo",
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS)

    if isDropletYOLO_Full:

        main_evaluation_full_image(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                        fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
                        fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
                        fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

                        path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_FULL_DATASET_YOLO,
                        path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_FULL_DATASET_YOLO,
                        path_dataset=config.DATA_SYNTHETIC_FULL_WSP_TESTING_DIR, 
                        path_results=config.RESULTS_SYNTHETIC_YOLO_DIR,
                        square_image_path=config.DATA_SYNTHETIC_WSP_TESTING_DIR,
                        path_real_results=config.RESULTS_SYNTHETIC_FULL_YOLO_DIR,
                        iou_threshold=IOU_THRESHOLD,
                        distance_threshold=DISTANCE_THRESHOLD, width_mm=76)
        
        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
                                               evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_FULL_DATASET_YOLO, 
                                               "droplet_synthetic_full_dataset_yolo",
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
                                                evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_FULL_DATASET_YOLO, 
                                                "droplet_synthetic_full_dataset_yolo",
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS) 
        
        # main_evaluation_full_image(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
        #                 fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
        #                 fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
        #                 fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

        #                 path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_FULL_DATASET_YOLO,
        #                 path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_FULL_DATASET_YOLO,
        #                 path_dataset=config.DATA_REAL_FULL_WSP_TESTING_DIR, 
        #                 path_results=config.RESULTS_REAL_YOLO_DIR,
        #                 square_image_path=config.DATA_REAL_WSP_TESTING_DIR2,
        #                 path_real_results=config.RESULTS_REAL_FULL_YOLO_DIR,
        #                 iou_threshold=IOU_THRESHOLD,
        #                 distance_threshold=DISTANCE_THRESHOLD, width_mm=26)

        # update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
        #                                        evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_FULL_DATASET_YOLO, 
        #                                        "droplet_real_full_dataset_yolo", 
        #                                        evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
        #                                        evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        
        # update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
        #                                         evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_FULL_DATASET_YOLO, 
        #                                         "droplet_real_full_dataset_yolo",
        #                                         evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
        #                                         evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS)
    
    if isDropletCELLPOSE_Square:

        # main_evaluation(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
        #                 fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
        #                 fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
        #                 fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

        #                 path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_CELLPOSE,
        #                 path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_CELLPOSE,
        #                 path_dataset=config.DATA_SYNTHETIC_WSP_TESTING_DIR, 
        #                 path_results=config.RESULTS_SYNTHETIC_CELLPOSE_DIR,
        #                 iou_threshold=IOU_THRESHOLD,
        #                 distance_threshold=DISTANCE_THRESHOLD)
        
        # update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
        #                                        evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_CELLPOSE, 
        #                                        "droplet_synthetic_square_dataset_cellpose",
        #                                        evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
        #                                        evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        # update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
        #                                         evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_CELLPOSE, 
        #                                         "droplet_synthetic_square_dataset_cellpose",
        #                                         evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
        #                                         evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS) 
        
        main_evaluation(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                        fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
                        fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
                        fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

                        path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_CELLPOSE,
                        path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_CELLPOSE,
                        path_dataset=config.DATA_REAL_WSP_TESTING_DIR2, 
                        path_results=config.RESULTS_REAL_CELLPOSE_DIR,
                        iou_threshold=IOU_THRESHOLD,
                        distance_threshold=DISTANCE_THRESHOLD)

        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
                                               evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_CELLPOSE, 
                                               "droplet_real_square_dataset_cellpose", 
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
                                                evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_CELLPOSE, 
                                                "droplet_real_square_dataset_cellpose",
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS)

    if isDropletCELLPOSE_Full:

        # main_evaluation(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
        #                 fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
        #                 fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
        #                 fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

        #                 path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_FULL_DATASET_CELLPOSE,
        #                 path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_FULL_DATASET_CELLPOSE,
        #                 path_dataset=config.DATA_SYNTHETIC_FULL_WSP_TESTING_DIR, 
        #                 path_results=config.RESULTS_SYNTHETIC_FULL_CELLPOSE_DIR,
        #                 iou_threshold=IOU_THRESHOLD,
        #                 distance_threshold=DISTANCE_THRESHOLD, width_mm=76)
        
        # update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
        #                                        evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_FULL_DATASET_CELLPOSE, 
        #                                        "droplet_synthetic_full_dataset_cellpose",
        #                                        evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
        #                                        evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        
        # update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
        #                                         evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_FULL_DATASET_CELLPOSE, 
        #                                         "droplet_synthetic_full_dataset_cellpose",
        #                                         evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
        #                                         evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS) 
        
        main_evaluation(fieldnames_segmentation=evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                        fieldnames_statistics=evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS,
                        fieldnames_time=evaluate_algorithms_config.FIELDNAMES_SEGMENTATION_TIME,
                        fieldnames_predicted_statistics=evaluate_algorithms_config.FIELDNAMES_PREDICTED_STATISTICS,

                        path_csv_segmentation=evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_FULL_DATASET_CELLPOSE,
                        path_csv_statistics=evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_FULL_DATASET_CELLPOSE,
                        path_dataset=config.DATA_REAL_FULL_WSP_TESTING_DIR, 
                        path_results=config.RESULTS_REAL_FULL_CELLPOSE_DIR,
                        iou_threshold=IOU_THRESHOLD,
                        distance_threshold=DISTANCE_THRESHOLD, width_mm=26)

        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, 
                                               evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_FULL_DATASET_CELLPOSE, 
                                               "droplet_real_full_dataset_cellpose", 
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                                               evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION)
        
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, 
                                                evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_FULL_DATASET_CELLPOSE, 
                                                "droplet_real_full_dataset_cellpose",
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS,
                                                evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS)

compute_evaluations()

# new_csv_file(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_STATISTICS)
# new_csv_file(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION)

#divide_evaluations_by_image_resolution(evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_CV,
                                    #    evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL,
                                    #     evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, 
                                    #     evaluate_algorithms_config.FIELDNAMES_DROPLET_GENERAL_SEGMENTATION,
                                    #     "droplet_synthetic_square_dataset_ccv"
                                       
                                    #    )