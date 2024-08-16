import os
import sys 
import cv2
import numpy as np
import csv
import time
import gc
import pandas as pd
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from joblib import Parallel, delayed
from matplotlib import pyplot as plt 

import droplet.ccv.Segmentation_CV as seg

sys.path.insert(0, 'src')
import Common.config as config 
import Common.Util as Util
from Common.Statistics import Statistics as stats

TP, FN, TN, FP = 0, 0, 0, 0
iou_threshold = 0.5

directory_image = os.path.join(config.DATA_SYNTHETIC_NORMAL_WSP_DIR, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
directory_label = os.path.join(config.DATA_SYNTHETIC_NORMAL_WSP_DIR, config.DATA_GENERAL_LABEL_FOLDER_NAME)
directory_stats = os.path.join(config.DATA_SYNTHETIC_NORMAL_WSP_DIR, config.DATA_GENERAL_STATS_FOLDER_NAME)

file_count = len([entry for entry in os.listdir(directory_image) if os.path.isfile(os.path.join(directory_image, entry))])
gt_matched = [False] * file_count

# manage folder
list_folders = []
list_folders.append(os.path.join(config.RESULTS_CV_DIR , config.RESULTS_GENERAL_STATS_FOLDER_NAME))
list_folders.append(os.path.join(config.RESULTS_CV_DIR , config.RESULTS_GENERAL_ACC_FOLDER_NAME))
list_folders.append(os.path.join(config.RESULTS_CV_DIR , config.RESULTS_GENERAL_LABEL_FOLDER_NAME))
list_folders.append(os.path.join(config.RESULTS_CV_DIR , config.RESULTS_GENERAL_INFO_FOLDER_NAME))
list_folders.append(os.path.join(config.RESULTS_CV_DIR , config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME))
list_folders.append(os.path.join(config.RESULTS_CV_DIR , config.RESULTS_GENERAL_UNDISTORTED_FOLDER_NAME))
list_folders.append(os.path.join(config.RESULTS_CV_DIR , config.RESULTS_GENERAL_MASK_SIN_FOLDER_NAME))
list_folders.append(os.path.join(config.RESULTS_CV_DIR , config.RESULTS_GENERAL_MASK_OV_FOLDER_NAME))

Util.manage_folders(list_folders)

def write_stats_csv(filename, predicted_stats:stats, groundtruth_stats:stats):
    
    with open(os.path.join(config.RESULTS_ACCURACY_DIR, "droplet_stats_evaluation_cv.csv"), mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file", "VMD_pred", "VMD_gt", "VMD_error", 
                                                  "RSF_pred", "RSF_gt", "RSF_error", 
                                                  "CoveragePercentage_pred", "CoveragePercentage_gt", "CoveragePercentage_error", 
                                                  "NoDroplets_pred", "NoDroplets_gt", "NoDroplets_error", 
                                                  "NoOverlappedDroplets_pred", "NoOverlappedDroplets_gt", "NoOverlappedDroplets_error",
                                                  "OverlappedDropletsPercentage_pred", "OverlappedDropletsPercentage_gt", "OverlappedDropletsPercentage_error"])
        
        new_row = {
            "file": filename, 
            "VMD_pred": predicted_stats.vmd_value, "VMD_gt": groundtruth_stats.vmd_value, "VMD_error": abs((predicted_stats.vmd_value - groundtruth_stats.vmd_value) / groundtruth_stats.vmd_value), 
            "RSF_pred": predicted_stats.rsf_value, "RSF_gt": groundtruth_stats.rsf_value, "RSF_error": abs((predicted_stats.rsf_value - groundtruth_stats.rsf_value) / groundtruth_stats.rsf_value), 
            "CoveragePercentage_pred": predicted_stats.coverage_percentage, "CoveragePercentage_gt": groundtruth_stats.coverage_percentage, "CoveragePercentage_error":abs((predicted_stats.coverage_percentage - groundtruth_stats.coverage_percentage) / groundtruth_stats.coverage_percentage), 
            "NoDroplets_pred": predicted_stats.no_droplets, "NoDroplets_gt": groundtruth_stats.no_droplets, "NoDroplets_error": abs((predicted_stats.no_droplets - groundtruth_stats.no_droplets) / groundtruth_stats.no_droplets), 
            "NoOverlappedDroplets_pred": predicted_stats.no_droplets_overlapped, "NoOverlappedDroplets_gt": groundtruth_stats.no_droplets_overlapped, "NoOverlappedDroplets_error": abs((predicted_stats.no_droplets_overlapped - groundtruth_stats.no_droplets_overlapped) / groundtruth_stats.no_droplets_overlapped),
            "OverlappedDropletsPercentage_pred": predicted_stats.overlaped_percentage, "OverlappedDropletsPercentage_gt": groundtruth_stats.overlaped_percentage, "OverlappedDropletsPercentage_error": abs((predicted_stats.overlaped_percentage - groundtruth_stats.overlaped_percentage) / groundtruth_stats.overlaped_percentage), 
        }
        writer.writerow(new_row)


def calculate_map_multiple(matches, matches_indices, ground_truth_masks):
    avg_precs = []
    iou_thrs = []

    for iou_thr in np.linspace(0.5, 0.95, 10):
        _, _, _, avg_precision, _, _, _ = calculate_map_single(matches, matches_indices, ground_truth_masks, iou_thr)
        
        avg_precs.append(avg_precision)
        iou_thrs.append(iou_thr)

    return np.mean(avg_precs)

def calculate_map_single(matches, matches_indices, ground_truth_masks, iou_threshold):
    tp = 0
    fp = 0
    fn = 0

    for _, _, iou in matches:
        if iou >= iou_threshold:
            tp += 1
        else:
            fp += 1

    unmatched_ground_truths = [ground_truth_masks[i] for i in range(len(ground_truth_masks)) if i not in matches_indices]
    fn = len(unmatched_ground_truths)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    avg_precision = calculate_avg_precision(precision, recall)
    
    return precision, recall, f1_score, avg_precision, tp, fp, fn
   
def calc_precision_recall(img_results):
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

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

def calculate_iou(im, mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0: return 0.0
    iou = intersection / union

    return iou

def create_yolo_mask(file_path, width, height, im):
    polygons = []
    mask_list = []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()            
            coordinates = list(map(float, parts[1:]))
            polygon = [(coordinates[i] * height, coordinates[i+1] * width) for i in range(0, len(coordinates), 2)]
            polygons.append(polygon)
        
    # sort the polygons
    polygons_with_centroids = [(polygon, calculate_centroid(polygon)) for polygon in polygons]
    sorted_polygons = sorted(polygons_with_centroids, key=lambda item: (item[1][0], item[1][1]))
        
    return sorted_polygons

def calculate_centroid(polygon):
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    centroid_x = sum(x_coords) / len(polygon)
    centroid_y = sum(y_coords) / len(polygon)
    return centroid_x, centroid_y

def create_mask(image_shape, polygon, is_circle=False, center=None, radius=None):
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    if is_circle:
        cv2.circle(mask, center, radius, 255, cv2.FILLED)
    else:
        cv2.drawContours(mask, [polygon], -1, 255, cv2.FILLED)
    return mask

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def match_predicted_to_groundtruth(predicted_polygon, ground_truths, droplet_shapes, distance_threshold, image_shape, matched_indices):
    best_iou = 0
    best_match = None
    best_match_index = None

    mask_predicted = np.zeros(image_shape, dtype=np.uint8)

    if predicted_polygon.overlappedIDs == []:
        cont = droplet_shapes.get(predicted_polygon.id)
        cv2.drawContours(mask_predicted, [cont], -1, 255, cv2.FILLED)
    else:
        cv2.circle(mask_predicted, (predicted_polygon.center_x, predicted_polygon.center_y), predicted_polygon.radius, 255, cv2.FILLED)
    predicted_center = (predicted_polygon.center_x, predicted_polygon.center_y)

    for index, (ground_truth_polygon, gt_center) in enumerate(ground_truths):
        if index in matched_indices:
            continue

        distance = np.linalg.norm(np.array(predicted_center) - np.array(gt_center))

        if distance < distance_threshold:
            mask_groundtruth =  np.zeros(image_shape, dtype=np.uint8)
            cont = ground_truth_polygon
            cv2.fillPoly(mask_groundtruth, np.array([cont], dtype=np.int32), 255)

            iou = calculate_iou(mask_predicted, mask_groundtruth)

            if iou > best_iou:
                best_iou = iou
                best_match = mask_groundtruth
                best_match_index = index

                if best_iou > 0.9:
                    break

    return mask_predicted, best_match, best_iou, best_match_index



def match_predictions_to_ground_truth(im, predicted_polygons, droplet_shapes, grounds_truths_polygons, distance_threshold):
    matches = []
    matched_indices = set() 


    for predicted_polygon in predicted_polygons:
        best_iou = 0
        best_match = None
        best_match_index = None
        
        mask_predicted = np.zeros_like(im)
        if predicted_polygon.overlappedIDs == []:
            cont = droplet_shapes.get(predicted_polygon.id)
            cv2.drawContours(mask_predicted, [cont], -1, 255, cv2.FILLED)
        else:
            cv2.circle(mask_predicted, (predicted_polygon.center_x, predicted_polygon.center_y), predicted_polygon.radius, 255, cv2.FILLED)
        
        for index, (ground_truth_polygon, gt_center) in enumerate(grounds_truths_polygons):
            # skip when groundtruth already matched
            if index in matched_indices:
                continue  

            mask_groundtruth = np.zeros_like(im)
            cont = ground_truth_polygon
            cv2.fillPoly(mask_groundtruth, np.array([cont], dtype=np.int32), 255)

            gt_x, gt_y = gt_center
            pr_x, pr_y = predicted_polygon.center_x, predicted_polygon.center_y

            distance = np.sqrt((pr_x - gt_x)**2 + (pr_y - gt_y)**2)

            if distance < distance_threshold:
                iou = calculate_iou(im, mask_predicted, mask_groundtruth)

                if iou > best_iou:
                    best_iou = iou
                    best_match = mask_groundtruth
                    best_match_index = index

                    if best_iou > 0.9:
                        break

        if best_match is not None:
            matched_indices.add(best_match_index)

        matches.append((mask_predicted, best_match, best_iou))

    return matches, matched_indices

def visualize_results(image_path, matches):
    image = cv2.imread(image_path)
    for pred_mask, true_coords, iou in matches:
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        if true_coords:
            cv2.rectangle(image, (true_coords[0], true_coords[1]), (true_coords[2], true_coords[3]), (0, 0, 255), 2)
    cv2.imshow('Segmentation Results', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def evaluate_matches(matches, matches_indices, ground_truth_masks, iou_threshold = 0.7):
    
    precision, recall, f1_score, map05, tp, fp, fn = calculate_map_single(matches, matches_indices, ground_truth_masks, iou_threshold)

    map0595 = calculate_map_multiple(matches, matches_indices, ground_truth_masks)
   
    tp = 0
    fp = 0
    fn = 0

    for _, _, iou in matches:
        if iou >= iou_threshold:
            tp += 1
        else:
            fp += 1

    unmatched_ground_truths = [ground_truth_masks[i] for i in range(len(ground_truth_masks)) if i not in matches_indices]
    fn = len(unmatched_ground_truths)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score, map05, map0595, tp, fp, fn

def write_final_csv_metrics(metric):
    with open(os.path.join(config.RESULTS_ACCURACY_DIR, "droplet_evaluation_cv.csv"), mode='a', newline='') as file:
        new_row = {
                "file": metric[0], "precision": metric[1], "recall": metric[2], "f1_score": metric[3], "map0.5": metric[4], "map0.5-0.95": metric[5], "tp": metric[6], "fp": metric[7], "fn": metric[8], "segmentation_time": metric[9]
            }
        writer = csv.DictWriter(file, fieldnames=["file", "precision", "recall", "f1_score", "map0.5", "map0.5-0.95", "tp", "fp", "fn", "segmentation_time"])
        writer.writerow(new_row)

def read_stats_file(filename):
    stats_file_path = (os.path.join(directory_stats, filename + ".csv"))
    data = pd.read_csv(stats_file_path)

    # Assign each value to a variable
    vmd_value = data.at[0, 'GroundTruth']
    rsf_value = data.at[1, 'GroundTruth']
    coverage_percentage = data.at[2, 'GroundTruth']
    no_total_droplets = data.at[3, 'GroundTruth']
    overlapped_percentage = data.at[4, 'GroundTruth']
    no_overlapped_droplets = data.at[5, 'GroundTruth']
            
    stats_groundtruth = stats(vmd_value, rsf_value, coverage_percentage, no_total_droplets, no_overlapped_droplets, overlapped_percentage, None)
    return stats_groundtruth


def compute_segmentation(file, filename):
    # read image
    image_gray = cv2.imread(os.path.join(directory_image, file), cv2.IMREAD_GRAYSCALE)
    image_colors = cv2.imread(os.path.join(directory_image, file))  
    image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
    width, height = image_colors.shape[:2]
    
    # get the predicted droplets with cv algorithm
    predicted_seg:seg.Segmentation_CV = seg.Segmentation_CV(image_colors, image_gray, filename, 
                                                save_image_steps = False, 
                                                create_masks = False, 
                                                segmentation_method = 0, 
                                                dataset_results_folder=config.DATA_SYNTHETIC_NORMAL_WSP_DIR)
    
    # calculate stats
    predicted_seg.droplet_area = [d.area for d in predicted_seg.droplets_data]

    predicted_seg.volume_list = sorted(stats.area_to_volume(predicted_seg.droplet_area, predicted_seg.width, config.WIDTH_MM))

    image_area = predicted_seg.width * predicted_seg.height
    vmd_value, coverage_percentage, rsf_value, _ = stats.calculate_statistics(predicted_seg.volume_list, image_area, predicted_seg.contour_area)
    
    no_droplets_overlapped = 0
    for drop in predicted_seg.droplets_data:
        if len(drop.overlappedIDs) > 0:
            no_droplets_overlapped += 1

    overlaped_percentage = no_droplets_overlapped /  predicted_seg.final_no_droplets * 100
    
    predicted_stats = stats(vmd_value, rsf_value, coverage_percentage, predicted_seg.final_no_droplets, no_droplets_overlapped, overlaped_percentage, predicted_seg.droplets_data)
    
    sorted_droplets = sorted(predicted_seg.droplets_data, key=lambda droplet: (droplet.center_x, droplet.center_y))

    return image_colors, sorted_droplets, predicted_seg.droplet_shapes, width, height, predicted_stats

def main():
    # start file
    with open(os.path.join(config.RESULTS_ACCURACY_DIR, "droplet_evaluation_cv.csv"), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file", "precision", "recall", "f1_score", "map0.5", "map0.5-0.95", "tp", "fp", "fn", "segmentation_time"])
        writer.writeheader()

    with open(os.path.join(config.RESULTS_ACCURACY_DIR, "droplet_stats_evaluation_cv.csv"), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file", "VMD_pred", "VMD_gt", "VMD_error", 
                                                  "RSF_pred", "RSF_gt", "RSF_error", 
                                                  "CoveragePercentage_pred", "CoveragePercentage_gt", "CoveragePercentage_error", 
                                                  "NoDroplets_pred", "NoDroplets_gt", "NoDroplets_error", 
                                                  "NoOverlappedDroplets_pred", "NoOverlappedDroplets_gt", "NoOverlappedDroplets_error",
                                                  "OverlappedDropletsPercentage_pred", "OverlappedDropletsPercentage_gt", "OverlappedDropletsPercentage_error"])
        writer.writeheader()

    # apply the segmentation in each one of the images and then calculate the accuracy and save it
    for i, file in enumerate(os.listdir(directory_image)): 
        start_time = time.time()

        parts = file.split(".")
        filename = parts[0]

        print("Evaluating image", filename + "..." )

        try:
            image_colors, predicted_droplets, droplet_shapes, width, height, predicted_stats = compute_segmentation(file, filename)
            seg_time = time.time()

            # get groundtruth
            groundtruth_polygons = create_yolo_mask(os.path.join(config.DATA_SYNTHETIC_NORMAL_WSP_DIR, config.DATA_GENERAL_LABEL_FOLDER_NAME, filename + ".txt"), width, height, image_colors)
            gt_stats = read_stats_file(filename)
            matches, matched_indices = match_masks(predicted_droplets, groundtruth_polygons, droplet_shapes, image_colors.shape, 10)

            precision, recall, f1_score, map5, map595, tp, fp, fn = evaluate_matches(matches, matched_indices, groundtruth_polygons, 0.5)
        
            segmentation_time = seg_time - start_time
            write_final_csv_metrics((filename, precision, recall, f1_score, map5, map595, tp, fp, fn, segmentation_time))
            
            write_stats_csv(filename, predicted_stats, gt_stats)

            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print("Time taken:", elapsed_time, "seconds")

        except np.core._exceptions._ArrayMemoryError as e:
            print(f"Memory error encountered while processing {filename}: {e}")
        
        # finally:
        #     # Cleanup to free memory
        #     del image_colors, predicted_droplets, droplet_shapes, groundtruth_polygons, matches, matched_indices, precision, recall, f1_score, map5, map595, tp, fp, fn 
        #     gc.collect()


main()