import sys
import numpy as np
import cv2

sys.path.insert(0, 'src')
sys.path.insert(0, 'src/common')
from Util import *
from Variables import *
from Droplet import *
from Statistics import *

class Accuracy:
    def __init__(self, calculated_droplets:dict[int, Droplet], groundtruth_droplets:dict[int, Droplet], filename, calculated_stats:Statistics, groundtruth_stats:Statistics):
        self.calculated_droplets = calculated_droplets
        self.groundtruth_droplets = groundtruth_droplets
        self.calculated_stats = calculated_stats
        self.groundtruth_stats = groundtruth_stats

        self.filename = filename
        
        self.calculate_accuracy_detected()
        self.find_pairs()
        self.calculate_accuracy_overlapped()

        self.write_stats_file()
    
    def calculate_accuracy_detected(self):
        # read the predicted and ground truth masks
        pred_overlapped_mask = cv2.imread(os.path.join(path_to_masks_overlapped_pred_folder, self.filename + ".png"), cv2.IMREAD_GRAYSCALE)
        pred_single_mask = cv2.imread(os.path.join(path_to_masks_single_pred_folder, self.filename + ".png"), cv2.IMREAD_GRAYSCALE)
        gt_overlapped_mask = cv2.imread(os.path.join(path_to_masks_overlapped_gt_folder, self.filename + ".png"), cv2.IMREAD_GRAYSCALE)
        gt_single_mask = cv2.imread(os.path.join(path_to_masks_single_gt_folder, self.filename + ".png"), cv2.IMREAD_GRAYSCALE)
        
        self.iou = self.calculate_iou(pred_overlapped_mask, gt_overlapped_mask, pred_single_mask, gt_single_mask)
        self.dice_coefficient = self.calculate_dice(pred_single_mask, gt_single_mask, pred_single_mask, gt_single_mask) 


    def find_pairs(self):
        self.save_pairs_id = []
        for pred_stat in self.calculated_droplets.values():
            for gt_stat in self.groundtruth_droplets.values():
       
                distance = np.sqrt((pred_stat.center_x - gt_stat.center_x)**2 + (pred_stat.center_y - gt_stat.center_y)**2 )
                #print(gt_stat.id, ' ', pred_stat.id, ' ',distance, ' ', abs(pred_stat.diameter - gt_stat.diameter))
                if distance < distance_threshold and abs(pred_stat.diameter - gt_stat.diameter) < diameter_threshold:
                    #print(gt_stat.id, ' ', pred_stat.id, ' ',distance, ' ', abs(pred_stat.diameter - gt_stat.diameter))
                    self.save_pairs_id.append((gt_stat.id, pred_stat.id))
                    
                    break
           

    def calculate_accuracy_overlapped(self):
        self.true_positives_overlapped = 0
        self.false_positives_overlapped = 0
        self.false_negatives_overlapped = 0
        self.true_negative_overlapped = 0
       
        for pair in self.save_pairs_id:
            gt_drop = self.groundtruth_droplets.get(pair[0])
            pred_drop = self.calculated_droplets.get(pair[1])
        
            if (gt_drop.overlappedIDs == []) and (pred_drop.overlappedIDs == []):
                self.true_positives_overlapped += 1
            elif not (gt_drop.overlappedIDs == []) and (pred_drop.overlappedIDs == []):
                self.false_positives_overlapped +=1
            elif not (gt_drop.overlappedIDs == []) and  not (pred_drop.overlappedIDs == []):
                self.true_negative_overlapped += 1
        
        self.false_negatives_overlapped = len(self.calculated_droplets) - self.true_positives_overlapped
        
        # calculate precision, recall, and F1-score
        self.precision_overlapped, self.recall_overlapped, self.f1_score_overlapped = self.calculate_parameters(self.true_positives_overlapped, self.true_negative_overlapped, self.false_positives_overlapped, self.false_negatives_overlapped)

            

    def write_stats_file(self):
        statistics_file_path = path_to_statistics_pred_folder + '\\' + self.filename + '.txt'
        with open(statistics_file_path, 'w') as f:

            f.write(f"Precision overlapped: {self.precision_overlapped:.5f}\n")
            f.write(f"Recall overlapped: {self.recall_overlapped:.5f}\n")
            f.write(f"F1-score overlapped: {self.f1_score_overlapped:.5f}\n\n")

            f.write(f"IOU overlapped: {self.iou:.2f}\n\n")
            f.write(f"Dice overlapped: {self.dice_coefficient:.2f}\n\n")

            f.write(f"VMD calculated: {self.calculated_stats.vmd_value:.2f} VMD groundtruth: {self.groundtruth_stats.vmd_value:.2f}\n")
            f.write(f"RSF calculated: {self.calculated_stats.rsf_value:.5f} RSF groundtruth: {self.groundtruth_stats.rsf_value:.5f}\n")
            f.write(f"Coverage percentage calculated: {self.calculated_stats.coverage_percentage:.2f} Coverage percentage groundtruth: {self.groundtruth_stats.coverage_percentage:.2f}\n")
            f.write(f"Number of droplets calculated: {self.calculated_stats.no_droplets:d} Number of droplets groundtruth: {self.groundtruth_stats.no_droplets:d}\n")
            
    def write_scores_file(precision_o, recall_o, f1_score_o, iou, dice):
    
        statistics_file_path = path_to_statistics_pred_folder + '\\' + "OVERALL_STATS" + '.txt'
        with open(statistics_file_path, 'w') as f:
  
            f.write(f"Precision overlapped: {precision_o:.5f}\n")
            f.write(f"Recall overlapped: {recall_o:.5f}\n")
            f.write(f"F1-score overlapped: {f1_score_o:.5f}\n\n")

            f.write(f"IOU overlapped: {iou:.2f}\n\n")
            f.write(f"Dice overlapped: {dice:.2f}\n\n")

    def calculate_iou(self, gt_ov_mask, pred_ov_mask, gt_s_mask, pred_s_mask):
        intersection_ov = np.logical_and(gt_ov_mask, pred_ov_mask) 
        intersection_s = np.logical_and(gt_s_mask, pred_s_mask)
        union_ov = np.logical_or(gt_ov_mask, pred_ov_mask) 
        union_s = np.logical_or(gt_s_mask, pred_s_mask)

        iou = (np.sum(intersection_ov) + np.sum(intersection_s)) / (np.sum(union_ov) + np.sum(union_s))
        return iou

    def calculate_dice(self, gt_ov_mask, pred_ov_mask, gt_s_mask, pred_s_mask):
        intersection_ov = np.logical_and(gt_ov_mask, pred_ov_mask) 
        intersection_s = np.logical_and(gt_s_mask, pred_s_mask)

        dice = 2 * (np.sum(intersection_ov) + np.sum(intersection_s)) / ((np.sum(gt_s_mask) + np.sum(gt_ov_mask)) + (np.sum(pred_s_mask) + np.sum(pred_ov_mask)))
        return dice

    
    def calculate_parameters(self, true_positives:int, true_negatives:int, false_positives:int, false_negatives:int):
        #accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) 
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
      
        f1_score = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1_score


        # self.true_positives_detect = 0
        # self.false_positives_detect = 0
        # self.false_negatives_detect = 0
        # #print(len(self.calculated_droplets))
        # self.save_pairs_id = []
        # for pred_stat in self.calculated_droplets.values():
        #     match_found = False
        #     for gt_stat in self.groundtruth_droplets.values():
        #         distance = np.sqrt((pred_stat.center_x - gt_stat.center_x)**2 + (pred_stat.center_y - gt_stat.center_y)**2 )
    
        #         if distance < distance_threshold and abs(pred_stat.diameter - gt_stat.diameter) < diameter_threshold:
        #             #print(gt_stat.id, ' ', pred_stat.id, ' ',distance, ' ', abs(pred_stat.diameter - gt_stat.diameter))
        #             self.save_pairs_id.append((gt_stat.id, pred_stat.id))
        #             match_found = True
        #             self.true_positives_detect += 1
        #             break
        #     if not match_found:
        #         self.false_positives_detect += 1
        
        # self.false_negatives_detect = len(self.calculated_droplets) - self.true_positives_detect

        # # calculate precision, recall, and F1-score
        # self.precision_detect, self.recall_detect, self.f1_score_detect = self.calculate_parameters(self.true_positives_detect, 0, self.false_positives_detect, self.false_negatives_detect)



        

