import os
import sys 
import cv2
import numpy as np
sys.path.insert(0, 'src\common')

import config
#sys.path.insert(0, 'src\Segmentation\ccv')
import Segmentation.ccv.Segmentation_CV as seg

import HoughTransform

TP, FN, TN, FP = 0, 0, 0, 0
iou_threshold = 0.5

directory_image = os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
directory_label = os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_LABEL_FOLDER_NAME)

file_count = len([entry for entry in os.listdir(directory_image) if os.path.isfile(os.path.join(directory_image, entry))])
gt_matched = [False] * file_count

def calculate_iou(im, mask1, mask2):
    mask1 = np.zeros_like(im)
    mask2 = np.zeros_like(im)

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
            polygon = [(coordinates[i] * width, coordinates[i+1] * height) for i in range(0, len(coordinates), 2)]
            polygons.append(polygon)

            mask = np.zeros_like(im)
            cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)

            mask_list.append(mask)
        
    return mask_list

def match_predictions_to_ground_truth(im, predicted_masks, grounds_truths_masks, calculated_droplets, groundtruth_droplets):
    save_pairs_id = []
    
    for pred_stat in calculated_droplets.values():
        for gt_stat in groundtruth_droplets.values():
            distance = np.sqrt((pred_stat.center_x - gt_stat.center_x)**2 + (pred_stat.center_y - gt_stat.center_y)**2 )
            
            if distance < config.ACCURACY_DISTANCE_THRESHOLD: #and abs(pred_stat.area - gt_stat.area) < config.ACCURACY_AREA_THRESHOLD:
                
                iou = calculate_iou(pred_stat, gt_stat)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = ground_truth

                    if best_iou > 0.9:
                        save_pairs_id.append((gt_stat.id, pred_stat.id))
                        break

                break
   
    # for predicted in predicted_masks:
    #     best_iou = 0
    #     best_match = None
        
    #     for ground_truth in grounds_truths_masks:
    #         iou = calculate_iou(predicted, ground_truth)

    #         if iou > best_iou:
    #             best_iou = iou
    #             best_match = ground_truth

    #             if best_iou > 0.9:
    #                 break

            # matches.append((predicted, best_match, best_iou))

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

def evaluate_matches(matches, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = 0

    for pred_mask, true_coords, iou in matches:
        if iou >= iou_threshold:
            tp += 1
        else:
            fp += 1
    
    matched_ground_truths = [match[1] for match in matches]

    fn = len(matches) - len(matched_ground_truths)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    print (precision, recall, f1_score)
    return precision, recall, f1_score

def evaluate_droplet_segmentation():
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
        for drop in predicted_seg.droplets_data:
            mask = np.zeros_like(image_colors)

            if drop.overlappedIDs == []:
                cont = predicted_seg.droplet_shapes.get(drop.id)
                cv2.drawContours(mask, [cont], -1, 255, cv2.FILLED)
            else:
                cv2.circle(mask, (drop.center_x, drop.center_y), drop.radius, 255, cv2.FILLED)
            predicted_masks.append(mask)
        
        # get groundtruth
        groundtruth_masks = create_yolo_mask(os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_LABEL_FOLDER_NAME, filename + ".txt"), width, height, image_colors)

        matches = match_predictions_to_ground_truth(image_colors, predicted_masks, groundtruth_masks, )
        
        evaluate_matches(matches, 0.5)

evaluate_droplet_segmentation()

