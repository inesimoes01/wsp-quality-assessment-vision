import statistics
import csv
import os
import sys
sys.path.insert(0, 'src/common')
from Variables import *
import matplotlib.pyplot as plt


class Dataset_Statistics():
    def __init__(self):
        self.no_overlapped_droplets = []
        for wsp_stats in os.listdir(path_to_statistics_gt_folder):
            with open(os.path.join(path_to_statistics_gt_folder, wsp_stats), 'r') as f:
                lines = f.readlines()
                self.no_overlapped_droplets.append(int(lines[0].split(":")[1].strip()))
            
        plt.figure(figsize=(8, 6))
        plt.hist(self.no_overlapped_droplets, bins=10, color='skyblue', edgecolor='black')
        plt.title('Distribution of Number of Droplets')
        plt.xlabel('Number of Droplets')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


    def read_csv(self, csv_file):
        self.drop_diameter = []
        no_elipses = 0
        with open(csv_file, mode='r', newline='') as file:
            diameter_column = csv.reader(file)
            for row in diameter_column:
                self.drop_diameter.append(row["Diameter"]) 
                #if (row["isElipse"] == "False"):
    
