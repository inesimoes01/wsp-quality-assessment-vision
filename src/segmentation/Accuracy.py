import sys

sys.path.insert(0, 'src/others')
from util import *
from paths import *

class Accuracy:
    def __init__(self, wsp_real_stats, filename):
        self.wsp_real_stats = wsp_real_stats
        self.filename = filename
        self.get_ground_truth()
        self.calculate_accuracy()
    

# def get_ground_truth(stats_file_path):
#     with open(stats_file_path, 'r') as f:
#         for line in f:
#             if "Number of droplets: " in line:
#                 number_of_droplets = int(line.split(":")[1].strip())
#             if "Number of overlapped droplets: " in line:
#                 no_overlapped_droplets = int(line.split(":")[1].strip())
#     return number_of_droplets, no_overlapped_droplets
        
    def get_ground_truth(self):
        stats_file_path = (os.path.join(path_to_statistics_folder, self.filename + ".txt"))
        with open(stats_file_path, 'r') as f:
            for line in f:
                if "Number of droplets: " in line:
                    self.real_no_total_droplets = int(line.split(":")[1].strip())
                if "Number of overlapped droplets: " in line:
                    self.real_no_overlapped_droplets = int(line.split(":")[1].strip())
    
    def calculate_accuracy_overlapped(self):
        return
            
            

        

