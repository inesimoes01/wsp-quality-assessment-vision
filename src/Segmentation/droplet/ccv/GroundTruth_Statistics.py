import os
import csv
import ast
import pandas as pd


import config
from Droplet import Droplet
from Statistics import Statistics

#TODO read the ellipse bool

class GroundTruth_Statistics:
    def __init__(self, filename):
        self.filename = filename
        self.read_stats_file()
        self.stats = Statistics(self.vmd_value, self.rsf_value, self.coverage_percentage, self.no_total_droplets, self.droplets)

    def read_stats_file(self):
        stats_file_path = (os.path.join(config.DATA_ARTIFICIAL_WSP_STATISTICS_DIR, self.filename + ".csv"))
        data = pd.read_csv(stats_file_path)

        # Assign each value to a variable
        self.vmd_value = data.at[0, 'GroundTruth']
        self.rsf_value = data.at[1, 'GroundTruth']
        self.coverage_percentage = data.at[2, 'GroundTruth']
        self.no_total_droplets = data.at[3, 'GroundTruth']
        self.overlapped_percentage = data.at[4, 'GroundTruth']
        self.no_overlapped_droplets = data.at[5, 'GroundTruth']
                  

        self.droplets:list[Droplet] = []
        dropletinfo_file_path = (os.path.join(config.DATA_ARTIFICIAL_WSP_INFO_DIR, self.filename + ".csv"))
        
        with open(dropletinfo_file_path, 'r') as f:
            csv_reader = csv.reader(f)
            isFirst = True
            for row in csv_reader:
                if (isFirst): isFirst = False
                else: 
                    overlapped = row[4]
                    if (row[4] != []):
                        overlapped = ast.literal_eval(overlapped)
                    
                    self.droplets.append(Droplet(int(row[1]), int(row[2]), float(row[3]), int(row[0]), overlapped))
        