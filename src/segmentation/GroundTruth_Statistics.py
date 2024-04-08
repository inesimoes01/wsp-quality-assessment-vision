import re
import cv2
import sys
import copy

sys.path.insert(0, 'src')
sys.path.insert(0, 'src/common')
from Droplet import *
from Variables import *
from Util import *
from Statistics import *

class GroundTruth_Statistics:
    def __init__(self, filename, image):
        self.filename = filename
        self.image = image

        # read file
        self.read_stats_file()
        self.save_roi()

        self.stats = Statistics(self.vmd_value, self.rsf_value, self.coverage_percentage, self.no_total_droplets)

    def read_stats_file(self):
        stats_file_path = (os.path.join(path_to_statistics_gt_folder, self.filename + ".txt"))
        with open(stats_file_path, 'r') as f:
            lines = f.readlines()
            self.no_total_droplets = int(lines[0].split(":")[1].strip())
            self.coverage_percentage = float(lines[1].split(":")[1].strip())
            self.vmd_value = float(lines[2].split(":")[1].strip())
            self.rsf_value = float(lines[3].split(":")[1].strip())
            self.no_overlapped_droplets = int(lines[4].split(":")[1].strip())

            self.droplets:list[Droplet] = []
            droplet_info_regex_overlapped = r"Droplet no (\d+) \((\d+), (\d+), (\d+)\): \[([\d, ]+)\]"
            droplet_info_regex = r"Droplet no (\d+) \((\d+), (\d+), (\d+)\): \[\]"
            droplet_id_regex = r"\d+"

            for line in lines[7:]:
                matches_overlapped = re.finditer(droplet_info_regex_overlapped, line)
                matches = re.finditer(droplet_info_regex, line)

                # lines with overlapped droplets information
                for match in matches_overlapped:
                    droplet_id, center_x, center_y, radius, overlapped_ids_str = match.groups()
                    overlapped_ids = [int(id_) for id_ in re.findall(droplet_id_regex, overlapped_ids_str)]
                    self.droplets.append(Droplet(int(center_x), int(center_y), int(radius), int(droplet_id), overlapped_ids))
                for match in matches:
                    droplet_id, center_x, center_y, radius = match.groups()
                    overlapped_ids = []
                    self.droplets.append(Droplet(int(center_x), int(center_y), int(radius), int(droplet_id), overlapped_ids))

    def save_roi(self):   
        self.enumerate_image = copy.copy(self.image)                      
        for drop in self.droplets:
            x1 = max(drop.center_x - drop.radius - border_expand, 0)
            y1 = max(drop.center_y - drop.radius - border_expand, 0)
            x2 = min(drop.center_x + drop.radius + border_expand, self.image.shape[1])
            y2 = min(drop.center_y + drop.radius + border_expand, self.image.shape[0])  

            object_roi = self.image[y1:y2, x1:x2]
            folder = path_to_inputs_folder + '\\' + self.filename
            create_folders(folder)
            cv2.imwrite(folder + '\\' + str(drop.id) + '.png', object_roi)

            cv2.putText(self.enumerate_image, f'{drop.id}', (int(drop.center_x-drop.radius), int(drop.center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imwrite(path_to_numbered_folder + '\\GT_' + self.filename + '.png', self.enumerate_image)
        
            
         
            
            
            #for line in lines[7:]:  # skip header lines
                # parts = line.strip().split(': ')
                # droplet_number = int(parts[0].split()[-1])
                # overlapping_droplets = [int(d) for d in parts[1][1:-1].split(', ')]
                
                # # Store the overlapped droplets in the dictionary
                # self.overlapped_droplets[droplet_number] = overlapping_droplets 

            # for line in f:
            #     if "Number of droplets: " in line:
            #         self.no_total_droplets = int(line.split(":")[1].strip())
            #     if "Coverage percentage: " in line:
            #         self.coverage_percentage = int(line.split(":")[1].strip())
            #     if "VMD value: " in line:
            #         self.vmd_value = int(line.split(":")[1].strip())
            #     if "RSF value: " in line:
            #         self.coverage_percentage = int(line.split(":")[1].strip())
            #     if "Number of overlapped droplets: " in line:
            #         self.no_overlapped_droplets = int(line.split(":")[1].strip())
            #     if "OVERLAPPED DROPLETS" in line:
 
                    
            #         parts = line.split(':')
            #         droplet_no = int(parts[0].split()[-1])  # Extract droplet number
            #         overlapped_droplets = [int(x) for x in parts[1].strip()[1:-1].split(',')]  # Extract overlapped droplets
            #         # Store the information in the dictionary
            #         drop[droplet_no] = overlapped_droplets
            #         self.coverage_percentage = int(line.split(":")[1].strip())
