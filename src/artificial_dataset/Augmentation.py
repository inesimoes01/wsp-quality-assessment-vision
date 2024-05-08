import numpy as np
import os
import sys
import cv2
import ast
import csv

sys.path.insert(0, 'src/common')
from Variables import *
from Droplet import *
from Util import *

class Augmentation():
    def __init__(self):

        total_wsp = len(os.listdir(path_to_images_folder))
        filename_count = total_wsp - 1

        for image_file in sorted(os.listdir(path_to_images_folder), key=lambda x: int(x.split('.')[0])):
            # get name of the file
            parts = image_file.split(".")
            filename = int(parts[0])
            
            print(filename)

            # get all images and files / classes
            image = cv2.imread(os.path.join(path_to_images_folder, image_file))
            height_orig, width_orig = image.shape[:2]
            mask_overlapped = cv2.imread(os.path.join(path_to_masks_overlapped_gt_folder, image_file))
            mask_single = cv2.imread(os.path.join(path_to_masks_single_gt_folder, image_file))
            
            orginal_droplet_info = self.read_droplet_csv_file(filename)
            
            # TRANSFORM 1 - flip horizontally (500)
            filename_count += 1
            image_flipped_horizontally = cv2.flip(image, 1)
            cv2.imwrite(os.path.join(path_to_images_folder, str(filename_count) + ".png"), image_flipped_horizontally)
            mask_overlapped_flipped_horizontally = cv2.flip(mask_overlapped, 1)
            cv2.imwrite(os.path.join(path_to_masks_overlapped_gt_folder, str(filename_count) + ".png"), mask_overlapped_flipped_horizontally)
            mask_single_flipped_horizontally = cv2.flip(mask_single, 1)
            cv2.imwrite(os.path.join(path_to_masks_single_gt_folder, str(filename_count) + ".png"), mask_single_flipped_horizontally)
            new_droplet_info = self.get_new_droplet_info(1, height_orig, width_orig, orginal_droplet_info)
            save_dropletinfo_csv(os.path.join(path_to_dropletinfo_gt_folder, str(filename_count) + ".csv"), new_droplet_info)
            mask_to_label(os.path.join(path_to_masks_single_gt_folder, str(filename_count) + ".png"), 
                            os.path.join(path_to_labels_yolo, str(filename_count) + ".txt"), 0)
            mask_to_label(os.path.join(path_to_masks_overlapped_gt_folder, str(filename_count) + ".png"), 
                            os.path.join(path_to_labels_yolo, str(filename_count) + ".txt"), 1)
            self.duplicate_statistic_file(filename, filename_count)
            
            # TRANSFORM 2 - flip vertically (500)
            filename_count += 1
            image_flipped_vertically = cv2.flip(image, 0)
            cv2.imwrite(os.path.join(path_to_images_folder, str(filename_count) + ".png"), image_flipped_vertically)
            mask_overlapped_flipped_vertically = cv2.flip(mask_overlapped, 0)
            cv2.imwrite(os.path.join(path_to_masks_overlapped_gt_folder, str(filename_count) + ".png"), mask_overlapped_flipped_vertically)
            mask_single_flipped_vertically = cv2.flip(mask_single, 0)
            cv2.imwrite(os.path.join(path_to_masks_single_gt_folder, str(filename_count) + ".png"), mask_single_flipped_vertically)
            new_droplet_info = self.get_new_droplet_info(2, height_orig, width_orig, orginal_droplet_info)
            save_dropletinfo_csv(os.path.join(path_to_dropletinfo_gt_folder, str(filename_count) + ".csv"), new_droplet_info)
            mask_to_label(os.path.join(path_to_masks_single_gt_folder, str(filename_count) + ".png"), 
                            os.path.join(path_to_labels_yolo, str(filename_count) + ".txt"), 0)
            mask_to_label(os.path.join(path_to_masks_overlapped_gt_folder, str(filename_count) + ".png"), 
                            os.path.join(path_to_labels_yolo, str(filename_count) + ".txt"), 1)
            self.duplicate_statistic_file(filename, filename_count)

            # TRANSFORM 3 - flip horizontally + flip vertically (500)
            filename_count += 1
            image_flipped_horizontally_vertically = cv2.flip(image_flipped_vertically, 1)
            cv2.imwrite(os.path.join(path_to_images_folder, str(filename_count) + ".png"), image_flipped_horizontally_vertically)
            mask_overlapped_flipped_horizontally_vertically = cv2.flip(mask_overlapped_flipped_vertically, 1)
            cv2.imwrite(os.path.join(path_to_masks_overlapped_gt_folder, str(filename_count) + ".png"), mask_overlapped_flipped_horizontally_vertically)
            mask_single_flipped_horizontally_vertically = cv2.flip(mask_single_flipped_vertically, 1)
            cv2.imwrite(os.path.join(path_to_masks_single_gt_folder, str(filename_count) + ".png"), mask_single_flipped_horizontally_vertically)
            new_droplet_info = self.get_new_droplet_info(3, height_orig, width_orig, orginal_droplet_info)
            save_dropletinfo_csv(os.path.join(path_to_dropletinfo_gt_folder, str(filename_count) + ".csv"), new_droplet_info)
            mask_to_label(os.path.join(path_to_masks_single_gt_folder, str(filename_count) + ".png"), 
                            os.path.join(path_to_labels_yolo, str(filename_count) + ".txt"), 0)
            mask_to_label(os.path.join(path_to_masks_overlapped_gt_folder, str(filename_count) + ".png"), 
                            os.path.join(path_to_labels_yolo, str(filename_count) + ".txt"), 1)
            self.duplicate_statistic_file(filename, filename_count)


    def read_droplet_csv_file(self, filename:int):
        droplet_info_original:list[Droplet] = [] 
        csv_path = os.path.join(path_to_dropletinfo_gt_folder, str(filename) + ".csv")
        with open(csv_path, mode='r', newline='') as file:
            csv_reader = csv.DictReader(file)

            for row in csv_reader:
                values_overlappedids = ast.literal_eval(row["OverlappedDropletsID"])
                droplet_info_original.append(Droplet(bool(row["isElipse"]), int(row["CenterX"]), int(row["CenterY"]), int(row["Diameter"]), int(row["DropletID"]), values_overlappedids))
        
        return droplet_info_original
    
    def get_new_droplet_info(self, transformation:int, orig_height:int, orig_width:int, droplet_info:list[Droplet]):
        match transformation:
            case 1:
                # horizontal flip
                center_column_index = orig_width // 2
                new_droplet_info = []
                for drop in droplet_info:
                    new_x = 2 * center_column_index - drop.center_x
                    new_droplet_info.append(Droplet(drop.isElipse, new_x, drop.center_y, drop.diameter, drop.id, drop.overlappedIDs))
            case 2: 
                # vertical flip
                center_row_index = orig_height // 2
                new_droplet_info = []
                for drop in droplet_info:
                    new_y = 2 * center_row_index - drop.center_y
                    new_droplet_info.append(Droplet(drop.isElipse, drop.center_x, new_y, drop.diameter, drop.id, drop.overlappedIDs))
            case 3:
                # horizontal + vertical flip
                center_row_index = orig_height // 2
                center_column_index = orig_width // 2
                new_droplet_info = []
                for drop in droplet_info:
                    new_y = 2 * center_row_index - drop.center_y
                    new_x = 2 * center_column_index - drop.center_x
                    new_droplet_info.append(Droplet(drop.isElipse, new_x, new_y, drop.diameter, drop.id, drop.overlappedIDs))
        
        return new_droplet_info
    
    def duplicate_statistic_file(self, original_name, new_name):
        original_file = os.path.join(path_to_statistics_gt_folder, str(original_name) + ".txt")
        new_file = os.path.join(path_to_statistics_gt_folder, str(new_name) + ".txt")

        try:
            with open(original_file, 'r') as original:
                content = original.read()

            with open(new_file, 'w') as new:
                new.write(content)

        except FileNotFoundError:
            print("File not found.")

    

Augmentation()
