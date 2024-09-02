# import json 
# import os
# import sys
# from shapely import Polygon
# import cv2
# import numpy as np
# import time 
# import csv
# import gc

# from ultralytics import YOLO
# import copy

# from matplotlib import pyplot as plt 
# from PIL import Image

# sys.path.insert(0, 'src')
# import Segmentation.droplet.ccv.Segmentation_CCV as seg
# import Common.config as config 
# import Common.Util as FoldersUtil
# from Statistics import Statistics as stats

# import Segmentation.evaluate_algorithms.util_evaluate_droplet as util

# #from Segmentation.droplet.ccv import Segmentation_CV as seg
# # turn label-studio into yolo

# # folder_path = "data\\testing\\real_dataset_annotated"
# # folder_path_squares = "data\\testing\\real_dataset_annotated_squares"
# # image_path = "image"
# # label_path = "studio_label"
# # yolo_path = "label"


# # def calc_precision_recall(img_results):
# #     true_pos = 0; false_pos = 0; false_neg = 0
# #     for _, res in img_results.items():
# #         true_pos += res['true_pos']
# #         false_pos += res['false_pos']
# #         false_neg += res['false_neg']

# #     try:
# #         precision = true_pos/(true_pos + false_pos)
# #     except ZeroDivisionError:
# #         precision = 0.0
# #     try:
# #         recall = true_pos/(true_pos + false_neg)
# #     except ZeroDivisionError:
# #         recall = 0.0

# #     return (precision, recall)

# # def calculate_avg_precision(precision, recall):
# #     precisions = []
# #     recalls = []
    
# #     precisions.append(precision)
# #     recalls.append(recall)
# #     precisions = np.array(precisions)
# #     recalls = np.array(recalls)
  
# #     prec_at_rec = []
# #     for recall_level in np.linspace(0.0, 1.0, 11):
# #         try:
# #             args = np.argwhere(recalls >= recall_level).flatten()
# #             prec = max(precisions[args])
# #         except ValueError:
# #             prec = 0.0
# #         prec_at_rec.append(prec)
# #     avg_prec = np.mean(prec_at_rec)

# #     return avg_prec

# # def write_stats_csv(filename, predicted_stats:stats, groundtruth_stats:stats, fieldnames_statistics, path_statistics):
    
# #     with open(path_statistics, mode='a', newline='') as file:
# #         writer = csv.DictWriter(file, fieldnames=fieldnames_statistics)
        
# #         new_row = {
# #             "file": filename, 
# #             "VMD_pred": predicted_stats.vmd_value, "VMD_gt": groundtruth_stats.vmd_value, "VMD_error": abs((predicted_stats.vmd_value - groundtruth_stats.vmd_value) / groundtruth_stats.vmd_value), 
# #             "RSF_pred": predicted_stats.rsf_value, "RSF_gt": groundtruth_stats.rsf_value, "RSF_error": abs((predicted_stats.rsf_value - groundtruth_stats.rsf_value) / groundtruth_stats.rsf_value), 
# #             "CoveragePercentage_pred": predicted_stats.coverage_percentage, "CoveragePercentage_gt": groundtruth_stats.coverage_percentage, "CoveragePercentage_error":abs((predicted_stats.coverage_percentage - groundtruth_stats.coverage_percentage) / groundtruth_stats.coverage_percentage), 
# #             "NoDroplets_pred": predicted_stats.no_droplets, "NoDroplets_gt": groundtruth_stats.no_droplets, "NoDroplets_error": abs((predicted_stats.no_droplets - groundtruth_stats.no_droplets) / groundtruth_stats.no_droplets), 
# #             }
# #         writer.writerow(new_row)


# def get_droplets(polygons, width_px, width_mm):
#     area_list = []
#     area_sum = 0
#     num_pols = 0

#     for pol in polygons:
#         pol = Polygon(pol)
#         area = pol.area
#         area_sum += area
#         area_list.append(area)
#         num_pols += 1

#     diameter_list = sorted(stats.area_to_diameter_micro(area_list, width_px, width_mm))

#     return area_list, diameter_list, area_sum, num_pols

# # def calculate_iou(mask1, mask2):
# #     intersection = np.logical_and(mask1, mask2).sum()
# #     union = np.logical_or(mask1, mask2).sum()

# #     if union == 0: return 0.0
# #     iou = intersection / union

# #     return iou  

# def match_predicted_to_groundtruth_yolo(predicted_polygons, ground_truths, distance_threshold, image):
#     best_iou = 0
#     best_match = None

#     matched_indices = []
#     matches = []

#     for index_pred, (predicted_polygon, pred_center) in enumerate(predicted_polygons):
#         best_match, best_iou, best_match_index = 0, 0, 0
#         mask_predicted = np.zeros_like(image)
#         cv2.drawContours(mask_predicted, [predicted_polygon], -1, 255, cv2.FILLED)
    
#         for index_gt, (ground_truth_polygon, gt_center) in enumerate(ground_truths):
#             if index_gt in matched_indices:
#                 continue

#             distance = np.linalg.norm(np.array(pred_center) - np.array(gt_center))

#             if distance < distance_threshold:
#                 mask_groundtruth = np.zeros_like(image)
#                 cont = ground_truth_polygon
#                 cv2.fillPoly(mask_groundtruth, np.array([cont], dtype=np.int32), 255)

#                 iou = calculate_iou(mask_predicted, mask_groundtruth)

#                 if iou > best_iou:
#                     best_iou = iou
#                     best_match = mask_groundtruth
#                     best_match_index = index_gt
                    
#                     if best_iou > 0.9:
#                         break
#         if best_iou > 0:
#             matches.append((mask_predicted, best_match, best_iou))
#             matched_indices.append(best_match_index)

#     return matches, matched_indices

# def match_predicted_to_groundtruth_cv(predicted_polygons, ground_truths, droplet_shapes, distance_threshold, image):
#     best_iou = 0
#     best_match = None

#     matched_indices = []
#     matches = []

#     for predicted_polygon in predicted_polygons:
#         best_match, best_iou, best_match_index = 0, 0, 0
#         mask_predicted = np.zeros_like(image)

#         if predicted_polygon.overlappedIDs == []:
#             cont = droplet_shapes.get(predicted_polygon.id)
#             cv2.drawContours(mask_predicted, [cont], -1, 255, cv2.FILLED)
#         else:
#             cv2.circle(mask_predicted, (predicted_polygon.center_x, predicted_polygon.center_y), predicted_polygon.radius, 255, cv2.FILLED)
#         predicted_center = (predicted_polygon.center_x, predicted_polygon.center_y)

#         for index, (ground_truth_polygon, gt_center) in enumerate(ground_truths):
#             if index in matched_indices:
#                 continue

#             distance = np.linalg.norm(np.array(predicted_center) - np.array(gt_center))

#             if distance < distance_threshold:
#                 mask_groundtruth =  np.zeros_like(image)
#                 cont = ground_truth_polygon
#                 cv2.fillPoly(mask_groundtruth, np.array([cont], dtype=np.int32), 255)

#                 iou = calculate_iou(mask_predicted, mask_groundtruth)

#                 if iou > best_iou:
#                     best_iou = iou
#                     best_match = mask_groundtruth
#                     best_match_index = index
                    
#                     if best_iou > 0.9:
#                         break

#         matches.append((mask_predicted, best_match, best_iou))
#         matched_indices.append(best_match_index)

#     return matches, matched_indices

# def calculate_centroid_yolo(polygon):
#     polygon = polygon.reshape(-1, 2)
#     x_coords = [point[0] for point in polygon]
#     y_coords = [point[1] for point in polygon]
#     centroid_x = sum(x_coords) / len(polygon)
#     centroid_y = sum(y_coords) / len(polygon)
#     return centroid_x, centroid_y

# def calculate_centroid(polygon):
#     x_coords = [point[0] for point in polygon]
#     y_coords = [point[1] for point in polygon]
#     centroid_x = sum(x_coords) / len(polygon)
#     centroid_y = sum(y_coords) / len(polygon)
#     return centroid_x, centroid_y

# # def get_yolo_polygons_from_label(file_path, height, width):
# #     polygons = []
# #     mask_list = []
    
# #     with open(file_path, 'r') as file:
# #         for line in file:
# #             parts = line.strip().split()            
# #             coordinates = list(map(float, parts[1:]))
# #             polygon = [(coordinates[i] * height, coordinates[i+1] * width) for i in range(0, len(coordinates), 2)]
# #             polygons.append(polygon)
        
# #     # sort the polygons
# #     polygons_with_centroids = [(polygon, calculate_centroid(polygon)) for polygon in polygons]
# #     sorted_polygons = sorted(polygons_with_centroids, key=lambda item: (item[1][0], item[1][1]))
        
# #     return sorted_polygons

# def evaluate_matches(matches, matches_indices, ground_truth_masks, iou_threshold = 0.5):
#     precision_final, recall_final, f1_score, tp, fn, fp, avg_precision_final = 0, 0, 0, 0, 0, 0, 0
#     avg_precision_list = []

#     # calculate precision for 0.7
#     precision_final, recall_final, f1_score, tp, fn, fp = calculate_precision_metrics(matches, matches_indices, iou_threshold, ground_truth_masks)

#     # calculate map for 0.5
#     precision, recall, f1_score, tp, fn, fp = calculate_precision_metrics(matches, matches_indices, 0.5, ground_truth_masks)
#     avg_precision_final = calculate_avg_precision(precision, recall)
#     avg_precision_list.append(avg_precision_final)

#     # calculate map from 0.5 to 0.95
#     for iou_thr in np.linspace(0.5, 0.95, 10):
#         precision, recall, avg_precision, _, _, _ = calculate_precision_metrics(matches, matches_indices, iou_thr, ground_truth_masks)
#         avg_precision = calculate_avg_precision(precision, recall)
#         avg_precision_list.append(avg_precision)        

#     return precision_final, recall_final, f1_score, tp, fn, fp, avg_precision_final, np.mean(avg_precision)

# def calculate_precision_metrics(matches, matches_indices, iou_threshold, ground_truth_masks):
#     tp, fn, fp = 0, 0, 0

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


#     return precision, recall, f1_score, tp, fn, fp

# def write_final_csv_metrics(metric, path_segmentation, fieldnames_segmentation):
#     with open(path_segmentation, mode='a', newline='') as file:
#         new_row = {
#                 "file": metric[0], "precision": metric[1], "recall": metric[2], "f1_score": metric[3], "map50": metric[4], "map50-95": metric[5], "tp": metric[6], "fp": metric[7], "fn": metric[8], "segmentation_time": metric[9]
#             }
#         writer = csv.DictWriter(file, fieldnames=fieldnames_segmentation)
#         writer.writerow(new_row)

# def calculate_ground_truth_statistics(polygons, width, image_width_mm, image_area):
#     area_list, diameter_list, contour_area, final_no_droplets = get_droplets(polygons, width, image_width_mm)

#     vmd_value, coverage_percentage, rsf_value, _ = stats.calculate_statistics(diameter_list, image_area, contour_area)
        
#     ground_truth_stats = stats(vmd_value, rsf_value, coverage_percentage, final_no_droplets, 0, 0)

#     return ground_truth_stats

# def apply_ccv_segmentation(image_colors, image_gray, filename, dataset_results):

#     predicted_seg:seg.Segmentation_CCV = seg.Segmentation_CCV(image_colors, image_gray, filename, 
#                                                 save_image_steps = False, 
#                                                 create_masks = False, 
#                                                 segmentation_method = 0, 
#                                                 dataset_results_folder = dataset_results)
    
#     # calculate stats
#     droplet_area = [d.area for d in predicted_seg.droplets_data]

#     diameter_list = sorted(stats.area_to_diameter_micro(droplet_area, predicted_seg.width, config.WIDTH_MM))
    
#     image_area = predicted_seg.width * predicted_seg.height
#     vmd_value, coverage_percentage, rsf_value, _ = stats.calculate_statistics(diameter_list, image_area, predicted_seg.contour_area)
    
#     no_droplets_overlapped = 0
#     for drop in predicted_seg.droplets_data:
#         if len(drop.overlappedIDs) > 0:
#             no_droplets_overlapped += 1

#     overlaped_percentage = no_droplets_overlapped /  predicted_seg.final_no_droplets * 100

#     #cv2.imwrite(os.path.join("results\\metrics\\droplet", "final_seg.png"), predicted_seg.detected_image)
#     predicted_stats = stats(vmd_value, rsf_value, coverage_percentage, predicted_seg.final_no_droplets, no_droplets_overlapped, overlaped_percentage, predicted_seg.droplets_data)
#     sorted_droplets = sorted(predicted_seg.droplets_data, key=lambda droplet: (droplet.center_x, droplet.center_y))

#     return sorted_droplets, predicted_seg.droplet_shapes, predicted_seg.width, predicted_seg.height, predicted_stats
    
# def apply_yolo_segmentation(image_path, image, width_mm, width, height, square_size, crop_index, yolo_model):

#     original_image = image.copy()
#     detected_image = copy.copy(image)
    
#     # predict image results
#     results = yolo_model(image_path, conf=0.1)
#     segmentation_result = results[0].masks.xy

#     predicted_droplets_adjusted = []
#     detected_pts = []

#     i, j = crop_index
#     left_offset = i * square_size
#     top_offset = j * square_size

#     for polygon in segmentation_result:
#         pts = np.array(polygon, np.int32)
#         pts = pts.reshape((-1, 1, 2))
#         detected_pts.append(pts)

#     #detection_coords = [np.array(coords).reshape(-1, 1, 2) for coords in detected_pts]
    
#     for coords in detected_pts:
#         adjusted_coords = []
#         for point in coords:
#             x, y = point[0]
#             new_x = x + left_offset
#             new_y = y + top_offset
#             adjusted_coords.append([new_x, new_y])
    
#         predicted_droplets_adjusted.append(np.array(adjusted_coords, dtype=np.int32))

#     return predicted_droplets_adjusted

# def calculate_yolo_stats(points, width, width_mm, height, filename, dataset_image_path):
#     area_list = []
#     area_sum = 0
#     num_pols = 0
#     im = cv2.imread(os.path.join(dataset_image_path, filename + ".png"))

#     list_polygons = []
    
#     for pts in points:
#         for p in pts:
#             if len(p) > 4:
#                 cv2.drawContours(im, [p], -1, (0, 0, 255), thickness=1) 
#                 flattened_array = p.reshape(-1, 2)
#                 coordinates = [tuple(point) for point in flattened_array]
                
#                 pol = Polygon(coordinates)
#                 area = pol.area
#                 area_sum += area
#                 area_list.append(area)
#                 num_pols += 1

#                 list_polygons.append(p)


#     polygons_with_centroids = [(polygon, calculate_centroid_yolo(polygon)) for polygon in list_polygons]
#     sorted_polygons = sorted(polygons_with_centroids, key=lambda item: (item[1][0], item[1][1]))
        
#     diameter_list = sorted(stats.area_to_diameter_micro(area_list, width, width_mm))
#     # calculate statistics
#     vmd_value, coverage_percentage, rsf_value, _ = stats.calculate_statistics(diameter_list, height*width, area_sum)

#     cv2.imwrite(os.path.join("results\\metrics\\droplet", "final_seg.png"), im)
#     predicted_stats = stats(vmd_value, rsf_value, coverage_percentage, num_pols, 0, 0)

#     return sorted_polygons, width, height, predicted_stats

# def divide_image_and_apply_yolo_segmentation(image_path, image_cv2, filename, yolo_model, square_dataset_path, square_size = 320):
#     img = Image.open(image_path)
#     img_width, img_height = img.size

#     # calculate number of squares in each dimension
#     num_squares_x = (img_width + square_size - 1) // square_size
#     num_squares_y = (img_height + square_size - 1) // square_size

#     predicted_polygons = []
#     # divide in squares
#     for i in range(num_squares_x):
#         for j in range(num_squares_y):
#             left = i * square_size
#             upper = j * square_size
#             right = min((i + 1) * square_size, img_width)
#             lower = min((j + 1) * square_size, img_height)

#             crop = img.crop((left, upper, right, lower))

#             new_image_path = f"{square_dataset_path}/image/{filename}_{i}_{j}.png"
#             crop_filename =new_image_path
#             crop.save(crop_filename)
#             print(f"Saved {crop_filename}")

#             width_mm = (right - left) * 76 / img_width

#             # detect with yolo and adjust the predicted labels to full image
#             predicted_polygons.append(apply_yolo_segmentation(new_image_path, image_cv2, width_mm, right-left, lower-upper, square_size, (i, j)), yolo_model)
    
#     return predicted_polygons

# #def main_real_ccv(path_statistics, path_segmentation, fieldnames_statistics, fieldnames_segmentation, path_dataset, path_dataset_results):
# def main_real_ccv(fieldnames_segmentation, fieldnames_statistics, path_csv_segmentation, path_csv_statistics, path_dataset, path_results, iou_threshold, distance_threshold):
#     directory_image = os.path.join(path_dataset, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
#     directory_label = os.path.join(path_dataset, config.DATA_GENERAL_LABEL_FOLDER_NAME)
#     directory_stats = os.path.join(path_dataset, config.DATA_GENERAL_STATS_FOLDER_NAME)

#     # manage folders to store the results of the segmentation
#     list_folders = []
#     list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_STATS_FOLDER_NAME))
#     list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_ACC_FOLDER_NAME))
#     list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME))
#     list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_INFO_FOLDER_NAME))
#     list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME))
#     list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_UNDISTORTED_FOLDER_NAME))
#     list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_MASK_SIN_FOLDER_NAME))
#     list_folders.append(os.path.join(path_results, config.RESULTS_GENERAL_MASK_OV_FOLDER_NAME))
#     FoldersUtil.manage_folders(list_folders)


#     with open(path_csv_segmentation, mode='w', newline='') as file:
#         csv.DictWriter(file, fieldnames=fieldnames_segmentation).writeheader()

#     with open(path_csv_statistics, mode='w', newline='') as file:
#         csv.DictWriter(file, fieldnames=fieldnames_statistics).writeheader()

#     for file in os.listdir(directory_image):
#         filename = file.split(".")[0]

#         image_gray = cv2.imread(os.path.join(directory_image, file), cv2.IMREAD_GRAYSCALE)
#         image_colors = cv2.imread(os.path.join(directory_image, file))  
#         image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)
#         width, height = image_colors.shape[:2]
    
#         image_area = width * height

#         if width > height: image_width_mm = 76
#         else: image_width_mm = 26

#         start_time = time.time()
        
#         # apply segmentation ccv
#         predicted_droplets, droplet_shapes, predicted_stats = util.compute_ccv_segmentation(file, filename, directory_image, path_results)
#         seg_time = time.time()

#         image_correct_predictions = copy.copy(image_colors)

#         # get groundtruth
#         groundtruth_polygons_with_centroid = util.create_yolo_mask(os.path.join(directory_label, filename + ".txt"), width, height)
#         ground_truth_polygons = [polygon for polygon, _ in groundtruth_polygons_with_centroid]

#         ground_truth_stats = calculate_ground_truth_statistics(ground_truth_polygons, width, image_width_mm, image_area)

#         # match each droplet to the segmentation
#         matches, matched_indices = match_predicted_to_groundtruth_cv(predicted_droplets, groundtruth_polygons_with_centroid, droplet_shapes, 10, im)

#         precision, recall, f1_score, tp, fn, fp,  map5, map595 = evaluate_matches(matches, matched_indices, ground_truth_polygons, 0.5)
        
#         segmentation_time = seg_time - start_time
#         util.write_final_csv_metrics((filename, precision, recall, f1_score, map5, map595, tp, fp, fn, segmentation_time), path_segmentation, fieldnames_segmentation)
            
#         util.write_stats_csv(filename, predicted_stats, ground_truth_stats, fieldnames_statistics, path_statistics)


#         end_time = time.time()
#         elapsed_time = end_time - start_time
        
#         print("Time taken:", elapsed_time, "seconds")
#         gc.collect()

# def main_real_yolo(path_statistics, path_segmentation, fieldnames_statistics, fieldnames_segmentation, dataset_image_path, dataset_yolo_label_path, yolo_model, path_dataset_squares):
   
#     with open(path_segmentation, mode='w', newline='') as file:
#         csv.DictWriter(file, fieldnames=fieldnames_segmentation).writeheader()

#     with open(path_statistics, mode='w', newline='') as file:
#         csv.DictWriter(file, fieldnames=fieldnames_statistics).writeheader()

#     for file in os.listdir(dataset_image_path):
  
#         filename = file.split(".")[0]
#         im = cv2.imread(os.path.join(dataset_image_path, file))
#         im_gray = cv2.imread(os.path.join(dataset_image_path, file), cv2.IMREAD_GRAYSCALE)
#         shape = im.shape
#         width = shape[0]
#         height = shape[1]
#         image_area = shape[0]*shape[1]

#         if width > height: image_width_mm = 76
#         else: image_width_mm = 26

#         start_time = time.time()
        
#         predicted_polygons = divide_image_and_apply_yolo_segmentation(os.path.join(dataset_image_path, file), im, filename, yolo_model, path_dataset_squares)
 
#         predicted_droplets_with_centroid, width, height, predicted_stats = calculate_yolo_stats(predicted_polygons, width, image_width_mm, height, filename)
#         seg_time = time.time()

#         # get groundtruth
#         groundtruth_polygons_with_centroid = get_yolo_polygons_from_label(os.path.join(dataset_yolo_label_path, filename + ".txt"), height, width)
#         ground_truth_polygons = [polygon for polygon, _ in groundtruth_polygons_with_centroid]
#         ground_truth_stats = calculate_ground_truth_statistics(ground_truth_polygons, width, image_width_mm, image_area)

#         # match each droplet to the segmentation
#         matches, matched_indices = match_predicted_to_groundtruth_yolo(predicted_droplets_with_centroid, groundtruth_polygons_with_centroid, 30, im)

#         precision, recall, f1_score, tp, fn, fp,  map5, map595 = evaluate_matches(matches, matched_indices, ground_truth_polygons, 0.5)
        
#         segmentation_time = seg_time - start_time
#         write_final_csv_metrics((filename, precision, recall, f1_score, map5, map595, tp, fp, fn, segmentation_time), path_segmentation, fieldnames_segmentation)
            
#         write_stats_csv(filename, predicted_stats, ground_truth_stats, fieldnames_statistics, path_statistics)

#         end_time = time.time()
#         elapsed_time = end_time - start_time
        
#         print("Time taken:", elapsed_time, "seconds")


