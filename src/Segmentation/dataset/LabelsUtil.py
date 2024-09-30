import json
import os
import cv2
import sys 
import pandas as pd
from shapely import Polygon

sys.path.insert(0, 'src')
import Common.config as config 
from Common.Statistics import Statistics as stats


class_mapping = {
    0: "droplet",
}

input_path_test = "data\\droplets\\synthetic_dataset_normal_droplets\\divided\\test"
output_path_test = "data\\droplets\\synthetic_dataset_normal_droplets\\mrcnn_ready\\test"

input_path_train = "data\\droplets\\synthetic_dataset_normal_droplets\\divided\\train"
output_path_train = "data\\droplets\\synthetic_dataset_normal_droplets\\mrcnn_ready\\train"

input_path_val = "data\\droplets\\synthetic_dataset_normal_droplets\\divided\\val"
output_path_val = "data\\droplets\\synthetic_dataset_normal_droplets\\mrcnn_ready\\val"


def convert_dataset_from_yolo_to_vgg(input_path, output_path, class_mapping):
    images_path = os.path.join(input_path, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
    labels_path = os.path.join(input_path, config.DATA_GENERAL_LABEL_FOLDER_NAME)

    if os.path.exists(os.path.join(output_path, "labels.json")):
        os.remove(os.path.join(output_path, "labels.json"))

    if not os.path.exists(os.path.join(output_path)):
        os.makedirs(os.path.join(output_path))

    label_files = os.listdir(labels_path)
    output_json_file = os.path.join(output_path, "labels.json")

    annotations = []
    
    for yolo_label_file in label_files:
        parts = yolo_label_file.split(".")
        filename = parts[0]

        im = cv2.imread(os.path.join(images_path, filename + ".png"))
        width, height = im.shape[:2]
        image_size = width * height
        
        yolo_label_file = os.path.join(labels_path, yolo_label_file)
        
        vgg_annotations = yolo_to_vgg(yolo_label_file, output_json_file, class_mapping, filename, image_size, width, height)

        annotations.append(vgg_annotations)

    with open(output_json_file, 'w') as json_file:
        json.dump(annotations, json_file, indent=2)
            

def yolo_to_vgg(yolo_txt_path, output_json_path, class_mapping, image_filename, image_size, width, height):
    """
    Convert YOLOv8 annotations to VGG JSON format.
    
    :param yolo_txt_path: Path to the YOLOv8 annotation txt file.
    :param output_json_path: Path to save the output VGG JSON file.
    :param class_mapping: Dictionary mapping YOLOv8 class IDs to class names.
    :param image_filename: Filename of the annotated image.
    :param image_size: Size of the annotated image.
    """
    vgg_annotations = {}
    
    # initialize VGG annotation
    vgg_annotations[image_filename + str(image_size)] = {
        "filename": image_filename,
        "size": image_size,
        "regions": [],
        "file_attributes": {}
    }
    
    with open(yolo_txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            coordinates = [float(x) for x in parts[1:]]

            if len(coordinates) % 2 != 0:
                raise ValueError(f"Invalid coordinates in line: {line}")
                
            # extract x and y coordinates
            all_points_x = [int(x * width) for x in coordinates[0::2]]
            all_points_y = [int(y * height) for y in coordinates[1::2]]


            # Create a region for this object
            region = {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": all_points_x,
                    "all_points_y": all_points_y
                },
                "region_attributes": {
                    "class": class_mapping.get(class_id, "droplet")
                }
            }
            
            vgg_annotations[image_filename + str(image_size)]["regions"].append(region)
    return vgg_annotations
    # save VGG annotations to a JSON file
    
       

def calculate_statistics(shapes, width_px, height_px, width_mm, statistics_file_path):
    total_droplet_area = 0
    total_no_droplets = 0
    list_droplet_area = []
    list_polygons = []
    
    # get general information from the polygons
    for pred in shapes:
        if len(pred) >= 8:
            coordinate_pairs = []
            for i in range(0, len(pred), 2):
                x = pred[i] * width_px
                y = pred[i + 1] * height_px
                coordinate_pairs.append((x, y))

            polygon = Polygon(coordinate_pairs)
            list_polygons.append(polygon)
            polygon_area = polygon.area

            total_droplet_area += polygon_area
            list_droplet_area.append(polygon_area)
            total_no_droplets += 1

    # get list of the droplet diameters
    diameter_list = sorted(stats.area_to_diameter_micro(list_droplet_area, width_px, width_mm))
    
    # find the overlapping polygons
    no_droplets_overlapped = 0
    overlapping_polygons = []
    for i, polygon in enumerate(list_polygons):
        if not polygon.is_valid:
            polygon = polygon.buffer(0)  # Attempt to fix invalid polygon
        for j, other_polygon in enumerate(list_polygons):
            if i != j:
                if not other_polygon.is_valid:
                    other_polygon = other_polygon.buffer(0)  # Attempt to fix invalid other polygon
                if polygon.intersects(other_polygon):
                    overlapping_polygons.append(i)
                    overlapping_polygons.append(j)
                    

    no_droplets_overlapped = len(overlapping_polygons)
    overlaped_percentage = no_droplets_overlapped / total_no_droplets * 100

    # calculate statistics
    vmd_value, coverage_percentage, rsf_value, _ = stats.calculate_statistics(diameter_list, height_px * width_px, total_droplet_area)    
    predicted_stats = stats(vmd_value, rsf_value, coverage_percentage, total_no_droplets, no_droplets_overlapped, overlaped_percentage)
    
    data = {
        '': config.FIELDNAMES_STATISTICS,
        'Groundtruth': [predicted_stats.vmd_value, predicted_stats.rsf_value, predicted_stats.coverage_percentage, predicted_stats.no_droplets, predicted_stats.overlaped_percentage, predicted_stats.no_droplets_overlapped], 
    }
    df = pd.DataFrame(data)
    df.to_csv(statistics_file_path, index=False, float_format='%.2f')

    return predicted_stats  

convert_dataset_from_yolo_to_vgg(input_path_test, output_path_test, class_mapping)

convert_dataset_from_yolo_to_vgg(input_path_train, output_path_train, class_mapping)

convert_dataset_from_yolo_to_vgg(input_path_val, output_path_val, class_mapping)

