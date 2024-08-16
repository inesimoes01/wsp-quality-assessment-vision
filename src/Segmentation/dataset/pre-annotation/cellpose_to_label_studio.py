import json
import uuid
import cv2
import numpy as np
from matplotlib import pyplot as plt 
import os


def find_unclosed_contours(contours):
    unclosed_contours = []
    for contour in contours:
        points_set = set()
        for point in contour:
            point_tuple = tuple(point[0])
            if point_tuple in points_set:
                unclosed_contours.append(contour)
                break
            points_set.add(point_tuple)
    return unclosed_contours

def remove_duplicate_lines(contour):
    unique_points = []
    points_set = set()
    for point in contour:
        point_tuple = tuple(point[0])
        if point_tuple not in points_set:
            unique_points.append(point)
            points_set.add(point_tuple)
    return np.array(unique_points)

def find_open_endpoints(contour):
    point_count = {}
    for point in contour:
        point_tuple = tuple(point[0])
        if point_tuple in point_count:
            point_count[point_tuple] += 1
        else:
            point_count[point_tuple] = 1
    open_endpoints = [point for point, count in point_count.items() if count == 1]
    return open_endpoints

def close_contour(contour):
    open_endpoints = find_open_endpoints(contour)
    if len(open_endpoints) == 2:
        contour = np.append(contour, [[open_endpoints[0]]], axis=0)
        contour = np.append(contour, [[open_endpoints[1]]], axis=0)
        return contour
    return contour



def cellpose_to_label_studio(image_width, image_height, path_to_cellpose_annotation, path_to_label_studio_annotation):
    annotations = []   
    points = [] 

    with open(path_to_cellpose_annotation, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            points = list(map(int, line.strip().split(',')))
            coordinates = [[points[i] / image_width * 100, points[i+1] / image_height * 100] for i in range(0, len(points), 2)]
     
            result = {
                    "original_width": image_width,
                    "original_height": image_height,
                    "image_rotation": 0,
                    "value": {
                        "points": coordinates, 
                        "closed": True, 
                        "polygonlabels": ["droplet"]},
                    "id": str(uuid.uuid4())[0:8],
                    "from_name": "label",
                    "to_name": "image",
                    "type": "polygonlabels",
                    "origin": "manual",
                 
            
            }
                
            annotations.append(result)

    return annotations

def save_annotations_to_json(results, image_name, path):

    task = {
        'data': {'image': image_name},
        'annotations': [],
        'predictions': [{
            "result": results,
            "project": 10
        }],
    }

    with open(path, 'w') as f:
        json.dump(task, f)


# Example usage
def main():
   
    path = "src\\Segmentation\\dataset\\pre-annotation\\files"   
    for file in os.listdir(path):
        image_path = os.path.join(path, file)
        image = cv2.imread(image_path)

        H, W = image.shape[:2]
        file_name_original = file.split("-")[1]
        file_name_original = file_name_original.split(".")[0]
        image_name = "/data/upload/10/" + str(file)

        
        parts = file.split(".")
        filename = parts[0]

        results_path = os.path.join("results\\cellpose\\pre_annotations\\label", filename + ".json")
        
        path_to_cellpose_annotation = os.path.join("results\\cellpose\\pre_annotations", file_name_original + "_cp_outlines.txt")
        
        if not os.path.isfile(path_to_cellpose_annotation):
            continue
        
        annotations = cellpose_to_label_studio(W, H, path_to_cellpose_annotation, results_path)
        #annotations = contour_to_label_studio_format(contours, W, H, path_to_file)
        save_annotations_to_json(annotations, image_name, results_path)
    
    with open(results_path, 'a') as f:
        f.write("]\n")

main()