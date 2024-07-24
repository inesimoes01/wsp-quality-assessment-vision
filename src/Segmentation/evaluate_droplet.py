import os
import sys 
import cv2
import numpy as np
import csv
from matplotlib import pyplot as plt 

import droplet.ccv.Segmentation_CV as seg

sys.path.insert(0, 'src\\common')
import config
import Util



TP, FN, TN, FP = 0, 0, 0, 0
iou_threshold = 0.5

directory_image = os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
directory_label = os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_LABEL_FOLDER_NAME)

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
   
    for predicted_mask, pred_center in predicted_masks:
        best_iou = 0
        best_match = None
        
        for ground_truth_mask, gt_center in grounds_truths_masks:

            gt_x, gt_y = gt_center
            pr_x, pr_y = pred_center

            distance = np.sqrt((pr_x - gt_x)**2 + (pr_y - gt_y)**2 )

            if distance < distance_threshold:

                iou = calculate_iou(im, predicted_mask, ground_truth_mask)

                if iou > best_iou:
                    best_iou = iou
                    best_match = ground_truth_mask


                    if best_iou > 0.9:
                        break
        Util.plotTwoImages(predicted_mask, best_match)
        matches.append((predicted_mask, best_match, best_iou))

    return matches

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

def evaluate_matches(matches, ground_truth_masks, iou_threshold = 0.5):
    tp = 0
    fp = 0
    fn = 0

    for pred_mask, true_coords, iou in matches:
        if iou >= iou_threshold:
            tp += 1
        else:
            fp += 1
    
    matched_ground_truths = [match[1] for match in matches]

    fn = len(ground_truth_masks) - len(matched_ground_truths)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score

def write_final_csv(metrics):
    with open(os.path.join(config.RESULTS_ACCURACY_DIR, "droplet_evaluation_cv.csv"), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file", "precision", "recall", "f1_score"])
        writer.writeheader()
       
        for metric in metrics:
            new_row = {
                "file": metric[0], "precision": metric[1], "recall": metric[2], "f1_score": metric[3]
            }
            writer.writerow(new_row)


def evaluate_droplet_segmentation():
    metrics_to_save = []
    
    # apply the segmentation in each one of the images and then calculate the accuracy and save it
    for i, file in enumerate(os.listdir(directory_image)):
        parts = file.split(".")
        filename = parts[0]

        # read image
        image_gray = cv2.imread(os.path.join(directory_image, file), cv2.IMREAD_GRAYSCALE)
        image_colors = cv2.imread(os.path.join(directory_image, file))  
        image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
        width, height = image_colors.shape[:2]
        
        # get the predicted droplets with cv algorithm
        predicted_seg:seg.Segmentation_CV = seg.Segmentation_CV(image_colors, image_gray, filename, 
                                                  save_image_steps = False, create_masks = False, 
                                                  segmentation_method = 0, 
                                                  real_width = config.WIDTH_MM, 
                                                  real_height = config.HEIGHT_MM)
        
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
        
        # get groundtruth
        groundtruth_masks = create_yolo_mask(os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_LABEL_FOLDER_NAME, filename + ".txt"), width, height, image_colors)

        matches = match_predictions_to_ground_truth(image_colors, predicted_masks, groundtruth_masks, 10)
        
        precision, recall, f1_score = evaluate_matches(matches, groundtruth_masks,  0.5)

        metrics_to_save.append((filename, precision, recall, f1_score))

    return metrics_to_save

metrics = evaluate_droplet_segmentation()
write_final_csv(metrics)
