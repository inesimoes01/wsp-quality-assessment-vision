import os
import sys 
import cv2
import numpy as np
import csv
import time
import copy
import gc
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt 
from shapely import Polygon

import Segmentation.droplet.ccv.Segmentation_CCV as seg
import Common.config as config 
import Common.Util as Util
from Common.Statistics import Statistics as stats


def write_stats_csv(filename, predicted_stats:stats, groundtruth_stats:stats, path_statistics, fieldname_statistics):
    
    with open(path_statistics, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldname_statistics)
        
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

# def calculate_map_multiple(matches, matches_indices, ground_truth_masks):
#     avg_precs = []
#     iou_thrs = []

#     for iou_thr in np.linspace(0.5, 0.95, 10):
#         _, _, _, avg_precision, _, _, _ = calculate_map_single(matches, matches_indices, ground_truth_masks, iou_thr)
        
#         avg_precs.append(avg_precision)
#         iou_thrs.append(iou_thr)

#     return np.mean(avg_precs)

# def calculate_map_single(matches, matches_indices, ground_truth_masks, iou_threshold):
#     tp = 0
#     fp = 0
#     fn = 0

#     for _, _, iou in matches:
#         if iou >= iou_threshold:
#             tp += 1
#         else:
#             fp += 1

#     unmatched_ground_truths = [ground_truth_masks[i] for i in range(len(ground_truth_masks)) if i not in matches_indices]
#     fn = len(unmatched_ground_truths)

#     precision = tp / (tp + fp) if tp + fp > 0 else 0
#     recall = tp / (tp + fn) if tp + fn > 0 else 0
#     f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

#     avg_precision = calculate_avg_precision(precision, recall)
    
#     return precision, recall, f1_score, avg_precision, tp, fp, fn
   
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

def calculate_real_dataset_statistics(polygons, width_px, width_mm, image_area):
    area_list = []
    area_sum = 0
    num_pols = 0

    for pol in polygons:
        pol = Polygon(pol)
        area = pol.area
        area_sum += area
        area_list.append(area)
        num_pols += 1

    diameter_list = sorted(stats.area_to_diameter_micro(area_list, width_px, width_mm))

    vmd_value, coverage_percentage, rsf_value, _ = stats.calculate_statistics(diameter_list, image_area, area_sum)
        
    ground_truth_stats = stats(vmd_value, rsf_value, coverage_percentage, num_pols, 0, 0)

    return ground_truth_stats

def create_yolo_mask(file_path, width, height):
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

def create_mask(predicted_polygon, droplet_shape, mask_predicted):
    # mask = np.zeros(image_shape, dtype=np.uint8)
    
    # if is_circle:
    #     cv2.circle(mask, center, radius, 255, cv2.FILLED)
    # else:
    #     cv2.drawContours(mask, [polygon], -1, 255, cv2.FILLED)
    # return mask

    if predicted_polygon.overlappedIDs == []:
        cv2.drawContours(mask_predicted, [droplet_shape], -1, 255, cv2.FILLED)
    else:
        cv2.circle(mask_predicted, (predicted_polygon.center_x, predicted_polygon.center_y), predicted_polygon.radius, 255, cv2.FILLED)
    predicted_center = (predicted_polygon.center_x, predicted_polygon.center_y)

    return mask_predicted, predicted_center


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_single_precision_yolo(ground_truth_masks, results, image_correct_predictions, iou_threshold):
    tp = 0
    fp = 0
    fn = 0
    
    matches_indices = []
    
    # calculate confusion matrix values and draw it in final image
    for predicted_polygon, best_iou, best_match_index in results:
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

def calculate_single_precision_ccv(droplet_shapes, ground_truth_masks, results, image_correct_predictions, iou_threshold):
    tp = 0
    fp = 0
    fn = 0
    
    matches_indices = []
    
    # calculate confusion matrix values and draw it in final image
    for predicted_polygon, best_iou, best_match_index in results:
        if best_match_index is not None:
            matches_indices.append(best_match_index)

            if best_iou > iou_threshold:
                tp += 1
                if predicted_polygon.overlappedIDs == []:
                    cont = droplet_shapes.get(predicted_polygon.id)
                    cv2.drawContours(image_correct_predictions, [cont], -1, (0, 255, 0), 1)
                else:
                    cv2.circle(image_correct_predictions, (predicted_polygon.center_x, predicted_polygon.center_y), predicted_polygon.radius, (0, 255, 0), 1)
            else: fp += 1

    unmatched_ground_truths = [ground_truth_masks[i] for i in range(len(ground_truth_masks)) if i not in matches_indices]
    fn = len(unmatched_ground_truths)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score, tp, fp, fn


def evaluate_matches_ccv(droplet_shapes, ground_truth_masks, results, image_correct_predictions, iou_threshold = 0.5, ):
    # calculate precision for the agreed iou threshold
    precision, recall, f1_score, tp, fp, fn = calculate_single_precision_ccv(droplet_shapes, ground_truth_masks, results, image_correct_predictions, iou_threshold)

    # calculate map05
    p05, r05, _, _, _, _ = calculate_single_precision_ccv(droplet_shapes, ground_truth_masks, results, image_correct_predictions, 0.5)
    map05 = calculate_avg_precision(p05, r05)

    # calculate map05095
    avg_precs = []
    iou_thrs = []

    for iou_thr in np.linspace(0.5, 0.95, 10):
        p0595, r0595, _, _, _, _ = calculate_single_precision_ccv(droplet_shapes, ground_truth_masks, results, image_correct_predictions, iou_thr)
        avg_precision = calculate_avg_precision(p0595, r0595)
        
        avg_precs.append(avg_precision)
        iou_thrs.append(iou_thr)

    map0595 =  np.mean(avg_precs)

    return image_correct_predictions, precision, recall, f1_score, map05, map0595, tp, fp, fn


def evaluate_matches_yolo(ground_truth_masks, results, image_correct_predictions, iou_threshold = 0.5):
    # calculate precision for the agreed iou threshold
    image_correct_predictions, precision, recall, f1_score, tp, fp, fn = calculate_single_precision_yolo(ground_truth_masks, results, image_correct_predictions, iou_threshold)

    # calculate map05
    _, p05, r05, _, _, _, _ = calculate_single_precision_yolo(ground_truth_masks, results, image_correct_predictions, 0.5)
    map05 = calculate_avg_precision(p05, r05)

    # calculate map05095
    avg_precs = []
    iou_thrs = []

    for iou_thr in np.linspace(0.5, 0.95, 10):
        _, p0595, r0595, _, _, _, _ = calculate_single_precision_yolo(ground_truth_masks, results, image_correct_predictions, iou_thr)
        avg_precision = calculate_avg_precision(p0595, r0595)
        
        avg_precs.append(avg_precision)
        iou_thrs.append(iou_thr)

    map0595 = np.mean(avg_precs)

    return image_correct_predictions, precision, recall, f1_score, map05, map0595, tp, fp, fn

def write_final_csv_metrics(metric, path_csv_segmentation, fieldnames_segmentation):
    with open(path_csv_segmentation, mode='a', newline='') as file:
        new_row = {
                fieldnames_segmentation[0]: metric[0], fieldnames_segmentation[1]: metric[1], 
                fieldnames_segmentation[2]: metric[2], fieldnames_segmentation[3]: metric[3], 
                fieldnames_segmentation[4]: metric[4], fieldnames_segmentation[5]: metric[5], 
                fieldnames_segmentation[6]: metric[6], fieldnames_segmentation[7]: metric[7], 
                fieldnames_segmentation[8]: metric[8], fieldnames_segmentation[9]: metric[9]
            }
        writer = csv.DictWriter(file, fieldnames=fieldnames_segmentation)
        writer.writerow(new_row)

def read_stats_file(filename, directory_stats):
    stats_file_path = (os.path.join(directory_stats, filename + ".csv"))
    data = pd.read_csv(stats_file_path)

    # Assign each value to a variable
    vmd_value = data.at[0, 'GroundTruth']
    rsf_value = data.at[1, 'GroundTruth']
    coverage_percentage = data.at[2, 'GroundTruth']
    no_total_droplets = data.at[3, 'GroundTruth']
      # this is an error in the creation of the synthetic dataset
    overlapped_percentage = data.at[5, 'GroundTruth']
    no_overlapped_droplets = data.at[4, 'GroundTruth']
            
    stats_groundtruth = stats(vmd_value, rsf_value, coverage_percentage, no_total_droplets, no_overlapped_droplets, overlapped_percentage, None)
    return stats_groundtruth

def calculate_centroid_yolo(polygon):
    polygon = polygon.reshape(-1, 2)
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    centroid_x = sum(x_coords) / len(polygon)
    centroid_y = sum(y_coords) / len(polygon)
    return centroid_x, centroid_y

def calculate_yolo_stats(points, width, height, width_mm, image_path):
    area_list = []
    area_sum = 0
    num_pols = 0
    im = cv2.imread(image_path)
    list_points = []
    list_polygons = []
    
    # draw shapes in image
    for pts in points:
        flattened_array = pts.reshape(-1, 2)
        coordinates = [tuple(point) for point in flattened_array]
        
        pol = Polygon(coordinates)
        area = pol.area
        area_sum += area
        area_list.append(area)
        num_pols += 1

        list_points.append(pts)
        list_polygons.append(pol)

    # calculate center of each polygon
    polygons_with_centroids = [(polygon, calculate_centroid_yolo(polygon)) for polygon in list_points]
    sorted_polygons = sorted(polygons_with_centroids, key=lambda item: (item[1][0], item[1][1]))
    diameter_list = sorted(stats.area_to_diameter_micro(area_list, width, width_mm))
    
    # find the overlapping polygons
    no_droplets_overlapped = 0
    overlapping_polygons = []
    for i, polygon in enumerate(list_polygons):
        for j, other_polygon in enumerate(list_polygons):
            if i != j and polygon.intersects(other_polygon):
                overlapping_polygons.append(i)
                overlapping_polygons.append(j)

    no_droplets_overlapped = len(overlapping_polygons)
    overlaped_percentage = no_droplets_overlapped / num_pols * 100

    # calculate statistics
    vmd_value, coverage_percentage, rsf_value, _ = stats.calculate_statistics(diameter_list, height*width, area_sum)    
    predicted_stats = stats(vmd_value, rsf_value, coverage_percentage, num_pols, no_droplets_overlapped, overlaped_percentage)
    
    return sorted_polygons, predicted_stats

def match_predicted_to_groundtruth_yolo(predicted_polygons, ground_truths, distance_threshold, image):
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
                cont = ground_truth_polygon
                cv2.fillPoly(mask_groundtruth, np.array([cont], dtype=np.int32), 255)

                iou = calculate_iou(mask_predicted, mask_groundtruth)

                if iou > best_iou:
                    best_iou = iou
                    best_match = mask_groundtruth
                    best_match_index = index_gt
                    
                    if best_iou > 0.9:
                        break
        if best_iou > 0:
            results.append((predicted_polygon, best_match_index, best_iou))
            #matched_indices.append(best_match_index)

    return results

def match_predicted_to_groundtruth_ccv(predicted_polygons, droplet_shapes, ground_truths, distance_threshold, image):
    best_iou = 0
    results = []
    matched_indices = []


    for predicted_polygon in predicted_polygons:
        best_iou, best_match_index = 0, 0
        mask_predicted = np.zeros_like(image)

        mask_predicted, predicted_center = create_mask(predicted_polygon, droplet_shapes.get(predicted_polygon.id), mask_predicted)

        for index, (ground_truth_polygon, gt_center) in enumerate(ground_truths):
            if index in matched_indices:
                continue

            distance = np.linalg.norm(np.array(predicted_center) - np.array(gt_center))

            if distance < distance_threshold:
                mask_groundtruth = np.zeros_like(image)
                cont = ground_truth_polygon
                cv2.fillPoly(mask_groundtruth, np.array([cont], dtype=np.int32), 255)

                iou = calculate_iou(mask_predicted, mask_groundtruth)

                if iou > best_iou:
                    best_iou = iou
                    best_match_index = index
                    
                    if best_iou > 0.9:
                        break

        results.append((predicted_polygon, best_iou, best_match_index))
    return results
  

