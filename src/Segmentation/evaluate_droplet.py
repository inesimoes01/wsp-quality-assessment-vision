import os
import sys 
import cv2
import numpy as np
import csv
import time
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

file_count = len([entry for entry in os.listdir(directory_image) if os.path.isfile(os.path.join(directory_image, entry))])
gt_matched = [False] * file_count

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
    
    # save the masks groundtruth
    for pol in sorted_polygons:
        mask = np.zeros_like(im)
        cont = np.array(pol[0])

        cv2.fillPoly(mask, np.array([cont], dtype=np.int32), 255)

        mask_list.append((mask, pol[1]))

        
    return mask_list

def calculate_centroid(polygon):
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    centroid_x = sum(x_coords) / len(polygon)
    centroid_y = sum(y_coords) / len(polygon)
    return centroid_x, centroid_y

def match_predictions_to_ground_truth(im, predicted_masks, grounds_truths_masks, distance_threshold):
    matches = []
    matched_indices = set() 

    for predicted_mask, pred_center in predicted_masks:
        best_iou = 0
        best_match = None
        best_match_index = None

        for index, (ground_truth_mask, gt_center) in enumerate(grounds_truths_masks):
            # skip when groundtruth already matched
            if index in matched_indices:
                continue  

            gt_x, gt_y = gt_center
            pr_x, pr_y = pred_center

            distance = np.sqrt((pr_x - gt_x)**2 + (pr_y - gt_y)**2)

            if distance < distance_threshold:
                iou = calculate_iou(im, predicted_mask, ground_truth_mask)

                if iou > best_iou:
                    best_iou = iou
                    best_match = ground_truth_mask
                    best_match_index = index

                    if best_iou > 0.9:
                        break

        if best_match is not None:
            matched_indices.add(best_match_index)

        matches.append((predicted_mask, best_match, best_iou))

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

def evaluate_matches(matches, matches_indices, ground_truth_masks, iou_threshold = 0.5):
    tp = 0
    fp = 0
    fn = 0

    for _, _, iou in matches:
        if iou >= iou_threshold:
            tp += 1
        else:
            fp += 1
    
    matched_ground_truths = [match[1] for match in matches]

    unmatched_ground_truths = [ground_truth_masks[i] for i in range(len(ground_truth_masks)) if i not in matches_indices]
    fn = len(unmatched_ground_truths)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score, tp, fp, fn

def write_final_csv(metric):
    with open(os.path.join(config.RESULTS_ACCURACY_DIR, "droplet_evaluation_cv.csv"), mode='a', newline='') as file:
        new_row = {
                "file": metric[0], "precision": metric[1], "recall": metric[2], "f1_score": metric[3], "tp": metric[4], "fp": metric[5], "fn": metric[6], "segmentation_time": metric[7]
            }
        writer = csv.DictWriter(file, fieldnames=["file", "precision", "recall", "f1_score", "tp", "fp", "fn", "segmentation_time"])
        writer.writerow(new_row)


def compute_segmentation(file, filename):
    # read image
    image_gray = cv2.imread(os.path.join(directory_image, file), cv2.IMREAD_GRAYSCALE)
    image_colors = cv2.imread(os.path.join(directory_image, file))  
    image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
    width, height = image_colors.shape[:2]
    
    # get the predicted droplets with cv algorithm
    predicted_seg:seg.Segmentation_CV = seg.Segmentation_CV(image_colors, image_gray, filename, 
                                                save_image_steps = False, create_masks = False, 
                                                segmentation_method = 0, 
                                                dataset_results_folder=config.DATA_SYNTHETIC_NORMAL_WSP_DIR)
    
    # calculate stats
    predicted_seg.droplet_area = [d.area for d in predicted_seg.droplets_data]

    predicted_seg.volume_list = sorted(stats.area_to_volume(predicted_seg.droplet_area, predicted_seg.width, config.WIDTH_MM))

    image_area = predicted_seg.width *  predicted_seg.height
    vmd_value, coverage_percentage, rsf_value, _ = stats.calculate_statistics(predicted_seg.volume_list, image_area, predicted_seg.contour_area)
    predicted_stats = stats(vmd_value, rsf_value, coverage_percentage, predicted_seg.final_no_droplets, predicted_seg.droplets_data)

    
    predicted_masks = []
    sorted_droplets = sorted(predicted_seg.droplets_data, key=lambda droplet: (droplet.center_x, droplet.center_y))
        
    for drop in sorted_droplets:
        mask = np.zeros_like(image_colors)

        if drop.overlappedIDs == []:
            cont = predicted_seg.droplet_shapes.get(drop.id)
            cv2.drawContours(mask, [cont], -1, 255, cv2.FILLED)
        else:
            cv2.circle(mask, (drop.center_x, drop.center_y), drop.radius, 255, cv2.FILLED)


        predicted_masks.append((mask, (drop.center_x, drop.center_y)))


    return image_colors, predicted_masks, width, height

def main():
    # start file
    with open(os.path.join(config.RESULTS_ACCURACY_DIR, "droplet_evaluation_cv.csv"), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file", "precision", "recall", "f1_score", "tp", "fp", "fn", "segmentation_time"])
        writer.writeheader()

    # apply the segmentation in each one of the images and then calculate the accuracy and save it
    for i, file in enumerate(os.listdir(directory_image)): 
        start_time = time.time()

        parts = file.split(".")
        filename = parts[0]

        print("Evaluating image", filename + "..." )

        image_colors, predicted_masks, width, height = compute_segmentation(file, filename)
        seg_time = time.time()

        # get groundtruth
        groundtruth_masks = create_yolo_mask(os.path.join(config.DATA_SYNTHETIC_NORMAL_WSP_DIR, config.DATA_GENERAL_LABEL_FOLDER_NAME, filename + ".txt"), width, height, image_colors)

        matches, matched_indices = match_predictions_to_ground_truth(image_colors, predicted_masks, groundtruth_masks, 10)
        
        precision, recall, f1_score, tp, fp, fn = evaluate_matches(matches, matched_indices, groundtruth_masks,  0.5)
    
        segmentation_time = seg_time - start_time
        write_final_csv((filename, precision, recall, f1_score, tp, fp, fn, segmentation_time))

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("Time taken:", elapsed_time, "seconds")


main()