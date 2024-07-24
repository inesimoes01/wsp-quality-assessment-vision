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


def contour_to_label_studio_format(contours, image_width, image_height, path_to_file):
    annotations = []    
    for contour in contours:

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        if area < 4:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.8:
            class_name = 'single'
        else:
            class_name = 'overlapped'
        
        points = []

        for point in contour:
            # Normalize points to percentages of the image dimensions
            x_percent = (point[0][0] / image_width) * 100
            y_percent = (point[0][1] / image_height) * 100
            points.append([x_percent, y_percent])

        
        if class_name == 'single':
            with open(path_to_file, 'a') as f:
                f.write("np.array([\n")
                for row in points:
                    f.write(f"    [{row[0]}, {row[1]}],\n")
                    
                f.write("], dtype=np.float32),\n")

        result = {
            "id": str(uuid.uuid4())[0:8],
            "type": "polygonlabels",
            "value": {
                "points": points, 
                "closed": True, 
                "polygonlabels": [class_name]},
            "origin": "manual",
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": image_width,
            "original_height": image_height,
        }
           
    
        annotations.append(result)



    
    return annotations

def save_annotations_to_json(results, image_name, path):

    task = {
        'data': {'image': image_name},
        'annotations': [],
        'predictions': [{
            "model_version": "v1",
            "result": results,
            "score": 0.5,
            "mislabeling": 0,
            "project": 8
        }],
    }

    with open(path, 'w') as f:
        json.dump(task, f)


# Example usage
if __name__ == "__main__":
    # Example contours (list of list of points)
    path = "C:\\Users\\mines\\AppData\\Local\\label-studio\\label-studio\\media\\upload\\6"
    
    path_to_file = 'shape_list.py'
    with open(path_to_file, 'w') as f:
        f.write(f"import numpy as np\n\n")
        f.write(f"shapes = [")
   
    for file in os.listdir(path):
        image_path = os.path.join(path, file)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_name = "/data/upload/6/" + str(file)
        
        # binary threshold
        _, threshold = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        H, W = threshold.shape


        parts = file.split(".")
        filename = parts[0]

        results_path = os.path.join("data\\real_dataset\\processed\\label", filename + ".json")

        annotations = contour_to_label_studio_format(contours, W, H, path_to_file)
        save_annotations_to_json(annotations, image_name, results_path)
    
    with open(path_to_file, 'a') as f:
        f.write("]\n")