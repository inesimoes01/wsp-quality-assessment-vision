import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt 
import copy
import csv
import pandas as pd

from CreateColors import *

sys.path.insert(0, 'src/common')
from Util import *
import config as config
from Statistics import *

from CreateWSP import CreateWSP

class WSP_Statistics:
    def __init__(self, wsp_image:CreateWSP, colors):
        self.colors = colors
        self.wsp_image = wsp_image
        self.image_num = copy.copy(wsp_image.blur_image)
     
        # find the circles that overlap
        self.find_overlapping_circles()
     
        # calculate statistics
        self.volume_list = sorted(Statistics.radius_to_volume(self.wsp_image.droplet_radius, wsp_image.width))
        cumulative_fraction = Statistics.calculate_cumulative_fraction(self.volume_list)
        vmd_value = Statistics.calculate_vmd(cumulative_fraction, self.volume_list)
        coverage_percentage = Statistics.calculate_coverage_percentage_gt(self.wsp_image.rectangle, self.wsp_image.height, self.wsp_image.width, self.wsp_image.colors.background_color_1, self.wsp_image.colors.background_color_2)
        rsf_value = Statistics.calculate_rsf(cumulative_fraction, self.volume_list, vmd_value)

        # create info and statistics files
        self.stats:Statistics = Statistics(vmd_value, rsf_value, coverage_percentage, wsp_image.num_spots, wsp_image.droplets_data)
        self.save_dropletinfo_csv()
        self.save_statistics_to_folder()

        # create masks
        path_mask_overlapped = os.path.join(config.DATA_ARTIFICIAL_RAW_MASK_OV_DIR, str(self.wsp_image.filename) + '.png')
        path_mask_single = os.path.join(config.DATA_ARTIFICIAL_RAW_MASK_SIN_DIR, str(self.wsp_image.filename) + '.png')
        self.create_masks(path_mask_overlapped, path_mask_single)

        # create the labels
        path_labels = os.path.join(config.DATA_ARTIFICIAL_RAW_LABEL_DIR, str(self.wsp_image.filename) + '.txt')
        self.mask_to_label(path_mask_single, path_labels, 0)
        self.mask_to_label(path_mask_overlapped, path_labels, 1)

 
        

  
    def find_overlapping_circles(self):
        self.no_overlapped_droplets = 0
        self.enumerate_image = copy.copy(self.wsp_image.blur_image)

        # iterate over each droplet and compare with all other droplets
        for droplet in self.wsp_image.droplets_data:
            center_y1 = droplet.center_y
            center_x1 = droplet.center_x
            r1 = int(droplet.radius)
            id1 = droplet.id

            for droplet2 in self.wsp_image.droplets_data:
                id2 = droplet2.id
                center_y2 = droplet2.center_y
                center_x2 = droplet2.center_x
                r2 = int(droplet2.radius)

                center_distance = np.sqrt((center_x2 - center_x1)**2 + (center_y2 - center_y1)**2)

                # if they overlap, mark it as overlapped
                if (center_distance < (r1 + r2) and id2 != id1):
                    droplet.overlappedIDs += [id2]
            
            cv2.putText(self.image_num, str(id1), (center_x1 + 2, center_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        path = os.path.join(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR, str(self.wsp_image.filename) + '_num.png')

        cv2.imwrite(path, self.image_num) 
       
    def verify_VDM(droplet_radii, vmd_value):
        check_vmd_s = 0
        check_vmd_h = 0
        equal = 0
        for i in range(len(droplet_radii)):
            if (vmd_value > droplet_radii[i]):
                check_vmd_h += 1
            if (vmd_value < droplet_radii[i]):
                check_vmd_s += 1
            if (droplet_radii[i] == 10):
                equal += 1

        print("len ", len(droplet_radii))
        print("number of droplets ", check_vmd_s, " ", check_vmd_h, " ", equal)
    
    def save_statistics_to_folder(self):
         
        data = {
            '': ['VMD', 'RSF', 'Coverage %', 'Nº Droplets', 'Overlapped Droplets %', 'Number of overlapped droplets'],
            'GroundTruth': [self.stats.vmd_value, self.stats.rsf_value, self.stats.coverage_percentage, self.stats.no_droplets, self.stats.overlaped_percentage, self.stats.no_droplets_overlapped], 
        }

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(config.DATA_ARTIFICIAL_RAW_STATISTICS_DIR, str(self.wsp_image.filename) + '.csv'), index=False, float_format='%.2f')

    def save_dropletinfo_csv(self):
        csv_file = os.path.join(config.DATA_ARTIFICIAL_RAW_INFO_DIR, str(self.wsp_image.filename) + '.csv')
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["DropletID", "isElipse", "CenterX", "CenterY", "Radius", "OverlappedDropletsID"])
            for drop in self.wsp_image.droplets_data:
                row = [drop.id, drop.isElipse, drop.center_x, drop.center_y, drop.radius, str(drop.overlappedIDs)]
                writer.writerow(row)

    def mask_to_label(self, path_mask, path_labels, classid):
        # load the binary mask and get its contours
        mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        H, W = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # convert the contours to polygons
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 5:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)
        
        # print the polygons
        with open('{}.txt'.format(path_labels[:-4]), 'a') as f:
            for polygon in polygons:
                f.write(f"{classid} {' '.join(map(str, polygon))}\n")
        
    def create_masks(self, path_mask_overlapped, path_mask_single):
        self.mask = np.zeros_like(self.wsp_image.rectangle)
        mask_overlapped = copy.copy(self.mask)
        mask_single = copy.copy(self.mask)
        
        for drop in self.wsp_image.droplets_data:
            radius = int(drop.radius)
            # single droplets
            if (drop.overlappedIDs == []):
                if drop.isElipse:
                    cv2.ellipse(mask_single, (drop.center_x, drop.center_y), (radius, radius + config.ELIPSE_MAJOR_AXE_VALUE), 5, 0, 360, 255, -1)
                else:
                    cv2.circle(mask_single, (drop.center_x, drop.center_y), radius, 255, -1)
            # overlapped droplets
            else:
                if drop.isElipse:
                    cv2.ellipse(mask_overlapped, (drop.center_x, drop.center_y), (radius, radius + config.ELIPSE_MAJOR_AXE_VALUE), 5, 0, 360, 255, -1)
                else:
                    cv2.circle(mask_overlapped, (drop.center_x, drop.center_y), radius, 255, -1)
        
        cv2.imwrite(path_mask_overlapped, mask_overlapped)
        cv2.imwrite(path_mask_single, mask_single)
