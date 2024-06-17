import os
import csv
import ast

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
        stats_file_path = (os.path.join(config.DATA_ARTIFICIAL_RAW_STATISTICS_DIR, self.filename + ".txt"))
        with open(stats_file_path, 'r') as f:
            lines = f.readlines()
            self.no_total_droplets = int(lines[0].split(":")[1].strip())
            self.coverage_percentage = float(lines[1].split(":")[1].strip())
            self.vmd_value = float(lines[2].split(":")[1].strip())
            self.rsf_value = float(lines[3].split(":")[1].strip())
            self.no_overlapped_droplets = int(lines[4].split(":")[1].strip())

        self.droplets:list[Droplet] = []
        dropletinfo_file_path = (os.path.join(config.DATA_ARTIFICIAL_RAW_INFO_DIR, self.filename + ".csv"))
        with open(dropletinfo_file_path, 'r') as f:
            csv_reader = csv.reader(f)
            isFirst = True
            for row in csv_reader:
                if (isFirst): isFirst = False
                else: 
                    overlapped = row[5]
                    if (row[5] != []):
                        overlapped = ast.literal_eval(overlapped)
                    
                    self.droplets.append(Droplet(row[1], int(row[2]), int(row[3]), int(row[4]), int(row[0]), overlapped))
        