import sys
import numpy as np

sys.path.insert(0, 'src/others')
from util import *
from paths import *

class Accuracy:
    def __init__(self, calculated_stats, groundtruth_stats):
        # save arrays of circles
        self.calculated_stats = calculated_stats
        self.groundtruth_stats = groundtruth_stats
        self.calculate_accuracy_overlapped()
    
    def calculate_accuracy_overlapped(self):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for c_stat in self.calculated_stats:
            match_found = False
            for gt_stat in self.groundtruth_stats:
                distance = np.sqrt((c_stat.center_x - gt_stat.center_x)**2 + (c_stat.center_y - gt_stat.center_y)**2 )

                if distance < distance_threshold and abs(c_stat.radius - gt_stat.radius) < radius_threshold:
                    match_found = True
                    true_positives += 1
                    break
            if not match_found:
                false_positives += 1
        
        false_negatives = len(self.groundtruth_stats) - true_positives

        # calculate precision, recall, and F1-score
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Print evaluation results
        print("Evaluation Results for :")
        print("True Positives:", true_positives)
        print("False Positives:", false_positives)
        print("False Negatives:", false_negatives)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1_score)
            




        

