import numpy as np
import os
import sys
import cv2
import ast
import csv

#from Droplet import Droplet
#import Util


sys.path.insert(0, 'src/common')
import config

from Droplet import Droplet
from Util import *

class Augmentation():
    def __init__(self):

        total_wsp = len(os.listdir(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR))
        filename_count = total_wsp - 1

        for image_file in sorted(os.listdir(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR), key=lambda x: int(x.split('.')[0])):
            # get name of the file
            parts = image_file.split(".")
            filename = int(parts[0])
            
            print(filename)

            # get all images and files / classes
            image = cv2.imread(os.path.join(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR, image_file))
            height_orig, width_orig = image.shape[:2]
            mask_overlapped = cv2.imread(os.path.join(config.DATA_ARTIFICIAL_WSP_MASK_OV_DIR, image_file))
            mask_single = cv2.imread(os.path.join(config.DATA_ARTIFICIAL_WSP_MASK_SIN_DIR, image_file))
            
            orginal_droplet_info = self.read_droplet_csv_file(filename)
            
            # TRANSFORM 1 - flip horizontally (500)
            filename_count += 1
            image_flipped_horizontally = cv2.flip(image, 1)
            cv2.imwrite(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_IMAGE_DIR, str(filename_count) + ".png"), image_flipped_horizontally)
            
            self.flip_horizontal_yolo_labels(filename_count, filename)

            mask_overlapped_flipped_horizontally = cv2.flip(mask_overlapped, 1)
            cv2.imwrite(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_MASK_OV_DIR, str(filename_count) + ".png"), mask_overlapped_flipped_horizontally)
            mask_single_flipped_horizontally = cv2.flip(mask_single, 1)
            cv2.imwrite(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_MASK_SIN_DIR, str(filename_count) + ".png"), mask_single_flipped_horizontally)
            new_droplet_info = self.get_new_droplet_info(1, height_orig, width_orig, orginal_droplet_info)
            save_dropletinfo_csv(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_INFO_DIR, str(filename_count) + ".csv"), new_droplet_info)
            
            self.duplicate_statistic_file(filename, filename_count)
            
            # TRANSFORM 2 - flip vertically (500)
            filename_count += 1
            image_flipped_vertically = cv2.flip(image, 0)
            cv2.imwrite(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_IMAGE_DIR, str(filename_count) + ".png"), image_flipped_vertically)
            
            self.flip_vertical_yolo_labels(filename_count, filename)

            mask_overlapped_flipped_vertically = cv2.flip(mask_overlapped, 0)
            cv2.imwrite(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_MASK_OV_DIR, str(filename_count) + ".png"), mask_overlapped_flipped_vertically)
            mask_single_flipped_vertically = cv2.flip(mask_single, 0)
            cv2.imwrite(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_MASK_SIN_DIR, str(filename_count) + ".png"), mask_single_flipped_vertically)
            new_droplet_info = self.get_new_droplet_info(2, height_orig, width_orig, orginal_droplet_info)
            save_dropletinfo_csv(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_INFO_DIR, str(filename_count) + ".csv"), new_droplet_info)

            self.duplicate_statistic_file(filename, filename_count)

            # TRANSFORM 3 - flip horizontally + flip vertically (500)
            filename_count += 1
            image_flipped_horizontally_vertically = cv2.flip(image_flipped_vertically, 1)
            cv2.imwrite(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_IMAGE_DIR, str(filename_count) + ".png"), image_flipped_horizontally_vertically)
            
            mask_overlapped_flipped_horizontally_vertically = cv2.flip(mask_overlapped_flipped_vertically, 1)
            cv2.imwrite(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_MASK_OV_DIR, str(filename_count) + ".png"), mask_overlapped_flipped_horizontally_vertically)
            mask_single_flipped_horizontally_vertically = cv2.flip(mask_single_flipped_vertically, 1)
            cv2.imwrite(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_MASK_SIN_DIR, str(filename_count) + ".png"), mask_single_flipped_horizontally_vertically)
            new_droplet_info = self.get_new_droplet_info(3, height_orig, width_orig, orginal_droplet_info)
            save_dropletinfo_csv(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_INFO_DIR, str(filename_count) + ".csv"), new_droplet_info)
            
            self.flip_horizontally_vertical_yolo_labels(filename_count, filename)
      
            self.duplicate_statistic_file(filename, filename_count)

            if(int(filename) > 500):
                break

    

    def flip_horizontally_vertical_yolo_labels(self, filename_count, curr_file):

        flipped_labels = []
        with open(os.path.join(config.DATA_ARTIFICIAL_RAW_LABEL_DIR, str(curr_file) + ".txt"), 'r') as file:
            wsp_lines = file.readlines()

        for line in wsp_lines:
            data = line.strip().split()
            if data == '':
                continue
          
            class_id = data[0]
            
            coords = [float(x) for x in data[1:]]
            
            flipped_coords = []
            
            for i in range(0, len(coords), 2):
                x, y = coords[i], coords[i+1]
                flipped_x = 1.0 - x  # Horizontal flip
                flipped_y = 1.0 - y  # Vertical flip
                flipped_coords.append(flipped_x)
                flipped_coords.append(flipped_y)
                #flipped_coords.extend(flipped_x, flipped_y)
            
            flipped_labels.append(flipped_coords)

        with open(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_LABEL_DIR, str(filename_count) + ".txt"), 'w') as f:
            for coords in flipped_labels:
                coord_str = " ".join([f"{p}" for p in coords])
                f.write(f"{0} {coord_str}\n")
        
        
    
    def flip_vertical_yolo_labels(self, filename_count, curr_file):

        flipped_labels = []
        with open(os.path.join(config.DATA_ARTIFICIAL_RAW_LABEL_DIR, str(curr_file) + ".txt"), 'r') as file:
            wsp_lines = file.readlines()
        
     
        for line in wsp_lines:
            data = line.strip().split()
            if data == '':
                continue
          
            class_id = data[0]
            
            coords = [float(x) for x in data[1:]]
            
            flipped_coords = []
            
            for i in range(0, len(coords), 2):
                x, y = coords[i], coords[i+1]
                flipped_y = 1.0 - y  # Vertical flip
                #flipped_coords.extend(x, flipped_y)
                flipped_coords.append(x)
                flipped_coords.append(flipped_y)
            
            flipped_labels.append(flipped_coords)

        with open(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_LABEL_DIR, str(filename_count) + ".txt"), 'w') as f:
            
            for coords in flipped_labels:
                coord_str = " ".join([f"{p}" for p in coords])
                f.write(f"{0} {coord_str}\n")
        
        

    def flip_horizontal_yolo_labels(self, filename_count, curr_file):
        flipped_labels = []

        with open(os.path.join(config.DATA_ARTIFICIAL_RAW_LABEL_DIR, str(curr_file) + ".txt"), 'r') as file:
            wsp_lines = file.readlines()
     
        for line in wsp_lines:
            data = line.strip().split()
            if data == '':
                continue
          
            class_id = data[0]
            
            coords = [float(x) for x in data[1:]]
            
            flipped_coords = []
            
            for i in range(0, len(coords), 2):
                x, y = coords[i], coords[i+1]
                flipped_x = 1.0 - x  # Horizontal flip
                
                #flipped_coords.extend(flipped_x, y)
                flipped_coords.append(flipped_x)
                flipped_coords.append(y)
            
            
            flipped_labels.append(flipped_coords)

        with open(os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_LABEL_DIR, str(filename_count) + ".txt"), 'w') as f:
            for coords in flipped_labels:
                coord_str = " ".join([f"{p}" for p in coords])
                f.write(f"{0} {coord_str}\n")
         

    def read_droplet_csv_file(self, filename:int):
        droplet_info_original:list[Droplet] = [] 
        csv_path = os.path.join(config.DATA_ARTIFICIAL_WSP_INFO_DIR, str(filename) + ".csv")
        with open(csv_path, mode='r', newline='') as file:
            csv_reader = csv.DictReader(file)

            for row in csv_reader:
                values_overlappedids = ast.literal_eval(row["OverlappedDropletsID"])
                droplet_info_original.append(Droplet(int(row["CenterX"]), int(row["CenterY"]), int(row["Area"]), int(row["DropletID"]), values_overlappedids))
        
        return droplet_info_original
    
    def get_new_droplet_info(self, transformation:int, orig_height:int, orig_width:int, droplet_info:list[Droplet]):
        match transformation:
            case 1:
                # horizontal flip
                center_column_index = orig_width // 2
                new_droplet_info = []
                for drop in droplet_info:
                    new_x = 2 * center_column_index - drop.center_x
                    new_droplet_info.append(Droplet(new_x, drop.center_y, drop.area, drop.id, drop.overlappedIDs))
            case 2: 
                # vertical flip
                center_row_index = orig_height // 2
                new_droplet_info = []
                for drop in droplet_info:
                    new_y = 2 * center_row_index - drop.center_y
                    new_droplet_info.append(Droplet(drop.center_x, new_y, drop.area, drop.id, drop.overlappedIDs))
            case 3:
                # horizontal + vertical flip
                center_row_index = orig_height // 2
                center_column_index = orig_width // 2
                new_droplet_info = []
                for drop in droplet_info:
                    new_y = 2 * center_row_index - drop.center_y
                    new_x = 2 * center_column_index - drop.center_x
                    new_droplet_info.append(Droplet(new_x, new_y, drop.area, drop.id, drop.overlappedIDs))
        
        return new_droplet_info
    
    def duplicate_statistic_file(self, original_name, new_name):
        original_file = os.path.join(config.DATA_ARTIFICIAL_WSP_STATISTICS_DIR, str(original_name) + ".csv")
        new_file = os.path.join(config.DATA_ARTIFICIAL_AUGMENTED_STATISTICS_DIR, str(new_name) + ".csv")

        try:
            with open(original_file, 'r') as original:
                content = original.read()

            with open(new_file, 'w') as new:
                new.write(content)

        except FileNotFoundError:
            print("File not found.")

    

Augmentation()
