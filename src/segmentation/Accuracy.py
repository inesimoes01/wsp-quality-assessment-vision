import sys
import numpy as np

sys.path.insert(0, 'src/others')
from util import *
from variables import *

sys.path.insert(0, 'src')
from Droplet import *

class Accuracy:
    def __init__(self, calculated_stats:list[Droplet], groundtruth_stats:list[Droplet], filename):
        self.calculated_stats = calculated_stats
        self.groundtruth_stats = groundtruth_stats
        self.filename = filename
        self.calculate_accuracy_overlapped()
        self.write_stats_file()
    
    def calculate_accuracy_overlapped(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

        self.save_pairs = []
        for c_stat in self.calculated_stats:
            match_found = False
            for gt_stat in self.groundtruth_stats:
                distance = np.sqrt((c_stat.center_x - gt_stat.center_x)**2 + (c_stat.center_y - gt_stat.center_y)**2 )
                
                if distance < distance_threshold and abs(c_stat.radius - gt_stat.radius) < radius_threshold:
                    self.save_pairs.append((gt_stat.id, c_stat.id))
                    match_found = True
                    self.true_positives += 1
                    break
            if not match_found:
                self.false_positives += 1
        
        self.false_negatives = len(self.groundtruth_stats) - self.true_positives

        # calculate precision, recall, and F1-score
        self.precision = self.true_positives / (self.true_positives + self.false_positives)
        self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)


    def write_stats_file(self):
        statistics_file_path = path_to_statistics_c_folder + '\\' + self.filename + '.txt'
        with open(statistics_file_path, 'w') as f:
            f.write(f"True Positives: {self.true_positives:d}\n")
            f.write(f"False Positives: {self.false_positives:d}\n")
            f.write(f"False Negatives: {self.false_negatives:d}\n")
            f.write(f"Precision: {self.precision:.5f}\n")
            f.write(f"Recall: {self.recall:.5f}\n")
            f.write(f"F1-score: {self.f1_score:.5f}\n")
            f.write(f"\n \nMATCHES: [gt_id] [c_id] \n")
            for pair in self.save_pairs:
                f.write(f"{pair}\n")
        





        

