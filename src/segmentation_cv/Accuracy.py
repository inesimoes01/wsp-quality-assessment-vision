import numpy as np
import cv2
import os 
import pandas as pd

import Util
import config
from Statistics import Statistics
from Droplet import Droplet

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

        self.write_accuracy_file()
        self.write_statistics_file()
    
    def calculate_accuracy_detected(self):
        # read the predicted and ground truth masks
        pred_overlapped_mask = cv2.imread(os.path.join(config.RESULTS_CV_MASK_OV_DIR, self.filename + ".png"), cv2.IMREAD_GRAYSCALE)
        pred_single_mask = cv2.imread(os.path.join(config.RESULTS_CV_MASK_SIN_DIR, self.filename + ".png"), cv2.IMREAD_GRAYSCALE)
        gt_overlapped_mask = cv2.imread(os.path.join(config.DATA_ARTIFICIAL_WSP_MASK_OV_DIR, self.filename + ".png"), cv2.IMREAD_GRAYSCALE)
        gt_single_mask = cv2.imread(os.path.join(config.DATA_ARTIFICIAL_WSP_MASK_SIN_DIR, self.filename + ".png"), cv2.IMREAD_GRAYSCALE)
        
        # calculate accuracy
        self.iou_overall, self.iou_single, self.iou_overlapped = self.calculate_iou(pred_overlapped_mask, gt_overlapped_mask, pred_single_mask, gt_single_mask)
        self.dice_coefficient = self.calculate_dice(pred_single_mask, gt_single_mask, pred_single_mask, gt_single_mask) 


    def find_pairs(self):
        data = {
            'DropletID': ['CenterX', 'CenterY', 'Area', 'OverlappedDropletsID', '', 'DropletID_GT', 'CenterX_GT', 'CenterY_GT', 'Area_GT', 'OverlappedDropletsID_GT'],
        }

        df = pd.DataFrame(columns=data)

        self.save_pairs_id = []
        for pred_stat in self.calculated_droplets.values():
            for gt_stat in self.groundtruth_droplets.values():
                distance = np.sqrt((pred_stat.center_x - gt_stat.center_x)**2 + (pred_stat.center_y - gt_stat.center_y)**2 )
                
                if distance < config.ACCURACY_DISTANCE_THRESHOLD and abs(pred_stat.area - gt_stat.area) < config.ACCURACY_DIAMETER_THRESHOLD:
                    self.save_pairs_id.append((gt_stat.id, pred_stat.id))
                    
                    new_row = {'DropletID': pred_stat.id, 'CenterX': pred_stat.center_x, 'CenterY': pred_stat.center_y, 'Area': pred_stat.area, 'OverlappedDropletsID': pred_stat.overlappedIDs, 
                        '': '', 
                        'DropletID_GT': gt_stat.id, 'CenterX_GT': gt_stat.center_x, 'CenterY_GT': gt_stat.center_y, 'Area_GT': gt_stat.area, 'OverlappedDropletsID_GT': gt_stat.overlappedIDs 
                        }
                    df = df._append(new_row, ignore_index=True)
                    break

        df.to_csv(os.path.join(config.RESULTS_CV_INFO_DIR, self.filename + '.csv'), index=False)
           
        
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

    def write_statistics_file(self):

        data = {
            '': ['VMD', 'RSF', 'Coverage %', 'Nº Droplets', 'Overlapped Droplets %'],
            'GroundTruth': [self.groundtruth_stats.vmd_value, self.groundtruth_stats.rsf_value, self.groundtruth_stats.coverage_percentage, self.groundtruth_stats.no_droplets, self.groundtruth_stats.overlaped_percentage],
            'Calculado': [self.calculated_stats.vmd_value, self.calculated_stats.rsf_value, self.calculated_stats.coverage_percentage, self.calculated_stats.no_droplets, self.calculated_stats.overlaped_percentage],   
        }

        df = pd.DataFrame(data)
        df['Erro'] = abs(df['GroundTruth'] - df['Calculado']) / df['GroundTruth'] * 100
        df.to_csv(os.path.join(config.RESULTS_CV_STATISTICS_DIR, self.filename + '.csv'), index=False)
    
    def write_accuracy_file(self):
        statistics_file_path = os.path.join(config.RESULTS_CV_ACCURACY_DIR, self.filename + '.txt')
        with open(statistics_file_path, 'w') as f:

            f.write(f"Precision: {self.precision_overlapped:.5f}\n")
            f.write(f"Recall: {self.recall_overlapped:.5f}\n")
            f.write(f"F1-score: {self.f1_score_overlapped:.5f}\n\n")

            f.write(f"IOU overall: {self.iou_overall:.2f}\n\n")
            f.write(f"IOU single: {self.iou_single:.2f}\n\n")
            f.write(f"IOU overlapped: {self.iou_overlapped:.2f}\n\n")
            f.write(f"Dice: {self.dice_coefficient:.2f}\n\n")

    def write_final_accuracy_file(precision_o, recall_o, f1_score_o, iou, dice):
        statistics_file_path = os.path.join(config.RESULTS_CV_ACCURACY_DIR, 'OVERALL_ACCURACY' + '.txt')

        with open(statistics_file_path, 'w') as f:
            f.write(f"Precision: {precision_o:.5f}\n")
            f.write(f"Recall: {recall_o:.5f}\n")
            f.write(f"F1-score: {f1_score_o:.5f}\n\n")

            f.write(f"IOU: {iou:.2f}\n\n")
            f.write(f"Dice: {dice:.2f}\n\n")

    def calculate_iou(self, gt_ov_mask, pred_ov_mask, gt_s_mask, pred_s_mask):
        intersection_ov = np.logical_and(gt_ov_mask, pred_ov_mask) 
        intersection_s = np.logical_and(gt_s_mask, pred_s_mask)
        
        union_ov = np.logical_or(gt_ov_mask, pred_ov_mask) 
        union_s = np.logical_or(gt_s_mask, pred_s_mask)

        iou_overlapped = np.sum(intersection_ov) / np.sum(union_ov) * 100

        iou_single = np.sum(intersection_s) / np.sum(union_s) * 100

        iou_overall = (np.sum(intersection_ov) + np.sum(intersection_s)) / (np.sum(union_ov) + np.sum(union_s))*100
        return iou_overall, iou_single, iou_overlapped

    def calculate_dice(self, gt_ov_mask, pred_ov_mask, gt_s_mask, pred_s_mask):
        intersection_ov = np.logical_and(gt_ov_mask, pred_ov_mask) 
        intersection_s = np.logical_and(gt_s_mask, pred_s_mask)

        dice = 2 * (np.sum(intersection_ov) + np.sum(intersection_s)) / ((np.sum(gt_s_mask) + np.sum(gt_ov_mask)) + (np.sum(pred_s_mask) + np.sum(pred_ov_mask)))
        return dice

    def calculate_parameters(self, true_positives:int, true_negatives:int, false_positives:int, false_negatives:int):
        if ((true_positives + false_positives) == 0) or true_positives == 0:  
            return 0, 0, 0
        else: 
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1_score = 2 * (precision * recall) / (precision + recall)
            return precision, recall, f1_score
        
    def write_final_statistics_file(groundtruth:Statistics, vmd_accum_gt, rsf_accum_gt, perc_accum_gt, no_accum_gt, percov_accum_gt, vmd_accum_c, rsf_accum_c, perc_accum_c, no_accum_c, percov_accum_c):
        data = {
            '': ['VMD', 'RSF', 'Coverage %', 'Nº Droplets', 'Overlapped Droplets %'],
            'GroundTruth': [vmd_accum_gt, rsf_accum_gt, perc_accum_gt, no_accum_gt, percov_accum_gt],
            'Calculado': [vmd_accum_c, rsf_accum_c, perc_accum_c, no_accum_c, percov_accum_c],   
        }

        df = pd.DataFrame(data)
        df['Erro'] = abs(df['GroundTruth'] - df['Calculado']) / df['GroundTruth'] * 100
        df.to_csv(os.path.join(config.RESULTS_CV_STATISTICS_DIR, 'OVERALL_STATISTICS.csv'), index=False)