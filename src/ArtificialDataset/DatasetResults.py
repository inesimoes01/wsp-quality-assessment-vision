import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt 
import copy
import csv
import pandas as pd
import CreateMask

from CreateColors import *

sys.path.insert(0, 'src/common')
from Util import *
import config as config
from Statistics import *

from CreateWSP import CreateWSP

class DatasetResults:
    def __init__(self, wsp_image:CreateWSP, colors):
        self.colors = colors
        self.wsp_image = wsp_image
        self.image_num = copy.copy(wsp_image.blur_image)

        area_list = [drop.droplet_data.area for drop in wsp_image.list_of_individual_shapes_in_image]

        # calculate statistics
        self.volume_list = sorted(Statistics.area_to_volume(area_list, wsp_image.width, config.WIDTH_MM))
        vmd_value, coverage_percentage, rsf_value, cumulative_fraction = Statistics.calculate_statistics(self.volume_list, (self.wsp_image.width * self.wsp_image.height), self.wsp_image.droplet_area)

        # create info and statistics files
        droplet_data = [r.droplet_data for r in wsp_image.list_of_individual_shapes_in_image]
        self.stats:Statistics = Statistics(vmd_value, rsf_value, coverage_percentage, wsp_image.num_spots, droplet_data)
        self.save_dropletinfo_csv(droplet_data)
        self.save_statistics_to_folder()

        self.save_cumulative_fraction(self.volume_list, cumulative_fraction, str(self.wsp_image.filename))

        # create masks
        path_mask_overlapped = os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_MASK_OV_FOLDER_NAME, str(self.wsp_image.filename) + '.png')
        path_mask_single = os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_MASK_SIN_FOLDER_NAME, str(self.wsp_image.filename) + '.png')
        self.create_masks(path_mask_overlapped, path_mask_single)

        # create the labels
        path_labels = os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_LABEL_FOLDER_NAME, str(self.wsp_image.filename) + '.txt')
        CreateMask.write_label_file(path_labels, self.wsp_image.annotation_labels)
        # self.mask_to_label(path_mask_single, path_labels, 0)
        # self.mask_to_label(path_mask_overlapped, path_labels, 1)

    def save_cumulative_fraction(self, volumes_sorted, cumulative_fraction, filename):
        csv_file = os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_INFO_FOLDER_NAME, filename + 'cumulative_fraction.csv')
        
        with open(csv_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Volume', 'Cumulative Fraction'])
            for volume, cumulative_fraction in zip(volumes_sorted, cumulative_fraction):
                csvwriter.writerow([volume, cumulative_fraction])
       
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
        df.to_csv(os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_STATS_FOLDER_NAME, str(self.wsp_image.filename) + '.csv'), index=False, float_format='%.2f')

    def save_dropletinfo_csv(self, droplet_data:list[Droplet]):
        csv_file = os.path.join(config.DATA_ARTIFICIAL_WSP_DIR, config.DATA_GENERAL_INFO_FOLDER_NAME, str(self.wsp_image.filename) + '.csv')
        
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["DropletID", "CenterX", "CenterY", "Area", "OverlappedDropletsID", "Size"])
            for drop in droplet_data:
                row = [drop.id, drop.center_x, drop.center_y, drop.area, str(drop.overlappedIDs), drop.radius]
                writer.writerow(row)

    # def mask_to_label(self, path_mask, path_labels, classid):
    #     # load the binary mask and get its contours
    #     mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
    #     _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    #     H, W = mask.shape
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     # convert the contours to polygons
    #     polygons = []
    #     for cnt in contours:
    #         if cv2.contourArea(cnt) > 5:
    #             polygon = []
    #             for point in cnt:
    #                 x, y = point[0]
    #                 polygon.append(x / W)
    #                 polygon.append(y / H)
    #             polygons.append(polygon)
        
    #     # print the polygons
    #     with open('{}.txt'.format(path_labels[:-4]), 'a') as f:
    #         for polygon in polygons:
    #             f.write(f"{classid} {' '.join(map(str, polygon))}\n")
        
    def create_masks(self, path_mask_overlapped, path_mask_single):
        self.mask = np.zeros_like(self.wsp_image.rectangle)
        mask_overlapped = copy.copy(self.mask)
        mask_single = copy.copy(self.mask)

        for drop in self.wsp_image.list_of_singular_shapes_in_image:
            points = list(drop.exterior.coords)
            points = np.array(points, np.int32).reshape(-1, 1, 2)


            cv2.fillPoly(mask_single, [points], 255)
        
        for drop in self.wsp_image.list_of_intersected_shapes_in_image:
            points = list(drop.exterior.coords)
            points = np.array(points, np.int32).reshape(-1, 1, 2)

            cv2.fillPoly(mask_overlapped, [points], 255)

        
        cv2.imwrite(path_mask_overlapped, mask_overlapped)
        cv2.imwrite(path_mask_single, mask_single)
