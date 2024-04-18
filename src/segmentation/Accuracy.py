import sys
import numpy as np

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
        self.calculate_accuracy_overlapped()
        self.write_stats_file()
    
    def calculate_accuracy_detected(self):
        true_positives_detect = 0
        false_positives_detect = 0
        false_negatives_detect = 0

        self.save_pairs_id = []
        for c_stat in self.calculated_droplets.values():
            match_found = False
            for gt_stat in self.groundtruth_droplets.values():
                distance = np.sqrt((c_stat.center_x - gt_stat.center_x)**2 + (c_stat.center_y - gt_stat.center_y)**2 )
                
                if distance < distance_threshold and abs(c_stat.radius - gt_stat.radius) < radius_threshold:
                    self.save_pairs_id.append((gt_stat.id, c_stat.id))
                    match_found = True
                    true_positives_detect += 1
                    break
            if not match_found:
                false_positives_detect += 1
        
        false_negatives_detect = len(self.calculated_droplets) - true_positives_detect

        # calculate precision, recall, and F1-score
        self.precision_detect, self.recall_detect, self.f1_score_detect =  self.calculate_parameters(true_positives_detect, 0, false_positives_detect, false_negatives_detect)

    def calculate_accuracy_overlapped(self):
        true_positives_overlapped = 0
        false_positives_overlapped = 0
        false_negatives_overlapped = 0
        true_negative_overlapped = 0

        for pair in self.save_pairs_id:
            gt_drop = self.groundtruth_droplets.get(pair[0])
            c_drop = self.calculated_droplets.get(pair[1])

            if (gt_drop.overlappedIDs == []) and (c_drop.overlappedIDs == []):
                true_positives_overlapped += 1
            elif not (gt_drop.overlappedIDs == []) and (c_drop.overlappedIDs == []):
                false_positives_overlapped +=1
            elif not (gt_drop.overlappedIDs == []) and  not (c_drop.overlappedIDs == []):
                true_negative_overlapped += 1
        
        false_negatives_overlapped = len(self.calculated_droplets) - true_positives_overlapped

        # calculate precision, recall, and F1-score
        self.precision_overlapped, self.recall_overlapped, self.f1_score_overlapped = self.calculate_parameters(true_positives_overlapped, true_negative_overlapped, false_positives_overlapped, false_negatives_overlapped)

    def calculate_parameters(self, true_positives:int, true_negatives:int, false_positives:int, false_negatives:int):
        #accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) 
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1_score
            

    def write_stats_file(self):
        statistics_file_path = path_to_statistics_c_folder + '\\' + self.filename + '.txt'
        with open(statistics_file_path, 'w') as f:
            f.write(f"Precision detect: {self.precision_detect:.5f}\n")
            f.write(f"Recall detect: {self.recall_detect:.5f}\n")
            f.write(f"F1-score detect: {self.f1_score_detect:.5f}\n\n")

            f.write(f"Precision overlapped: {self.precision_overlapped:.5f}\n")
            f.write(f"Recall overlapped: {self.recall_overlapped:.5f}\n")
            f.write(f"F1-score overlapped: {self.f1_score_overlapped:.5f}\n\n")

            f.write(f"VMD calculated: {self.calculated_stats.vmd_value:.2f} VMD groundtruth: {self.groundtruth_stats.vmd_value:.2f}\n")
            f.write(f"RSF calculated: {self.calculated_stats.rsf_value:.5f} RSF groundtruth: {self.groundtruth_stats.rsf_value:.5f}\n")
            f.write(f"Coverage percentage calculated: {self.calculated_stats.coverage_percentage:.2f} Coverage percentage groundtruth: {self.groundtruth_stats.coverage_percentage:.2f}\n")
            f.write(f"Number of droplets calculated: {self.calculated_stats.no_droplets:d} Number of droplets groundtruth: {self.groundtruth_stats.no_droplets:d}\n")
            
            # f.write(f"\n \nMATCHES: [gt_id] [c_id] \n")
            # for pair in self.save_pairs:
            #     f.write(f"{pair}\n")
        





        

