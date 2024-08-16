from PIL import Image
import os
import cv2
from shapely import Polygon
import numpy as np
import sys

import copy

sys.path.insert(0, 'src')
import Common.config as config 
from Statistics import Statistics as stats


folder_path = "data\\testing\\synthetic_dataset_squares"
image_path = "image"
label_path = "studio_label"
yolo_path = "label"

train_model_path = "results\\yolo_droplet\\50epc_droplet4\\weights\\best.pt"
model = YOLO(train_model_path)

metrics_path_csv_file = "results\\metrics\\droplet\\synthetic_dataset\\eval_synthetic_dataset_cv2.csv"
metrics_path_csv_file_stats = "results\\metrics\\droplet\\synthetic_dataset\\stats_synthetic_dataset_cv2.csv"

metrics_yolo_path_csv_file = "results\\metrics\\droplet\\synthetic_dataset\\eval_synthetic_dataset_yolo.csv"
metrics_yolo_path_csv_file_stats = "results\\metrics\\droplet\\synthetic_dataset\\stats_synthetic_dataset_yolo.csv"

def calculate_centroid_yolo(polygon):
    polygon = polygon.reshape(-1, 2)
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    centroid_x = sum(x_coords) / len(polygon)
    centroid_y = sum(y_coords) / len(polygon)
    return centroid_x, centroid_y

def divide_image_with_labels(image_path, filename, label_path, output_folder, square_size=320):
    # Open the image
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img = Image.open(image_path)
    img_width, img_height = img.size

    with open(os.path.join(label_path, filename + ".txt"), 'r') as f:
        labels = f.readlines()

    # Calculate number of squares in each dimension
    num_squares_x = (img_width + square_size - 1) // square_size
    num_squares_y = (img_height + square_size - 1) // square_size

    for i in range(num_squares_x):
        for j in range(num_squares_y):
            left = i * square_size
            upper = j * square_size
            right = min((i + 1) * square_size, img_width)
            lower = min((j + 1) * square_size, img_height)

            crop = img.crop((left, upper, right, lower))

            new_image_path = f"{output_folder}/image/{filename}_{i}_{j}.png"
            crop_filename =new_image_path
            crop.save(crop_filename)
            print(f"Saved {crop_filename}")

            width_mm = (right - left) * 76 / img_width

            # detect with yolo and create predicted stats
            sorted_polygons, width, height, predicted_stats = apply_yolo_segmentation(new_image_path, width_mm, right-left, lower-upper)



            # # labels of groundtruth
            # crop_labels = []
            # for label in labels:
            #     parts = list(map(float, label.split()))
            #     cls_id = int(parts[0])
            #     points = parts[1:]

            #     # Adjust segmentation points
            #     adjusted_points = []
            #     inside = False
            #     for k in range(0, len(points), 2):
            #         x_abs = points[k] * img_width
            #         y_abs = points[k + 1] * img_height

            #         if left <= x_abs <= right and upper <= y_abs <= lower:
            #             new_x = (x_abs - left) / (right - left)
            #             new_y = (y_abs - upper) / (lower - upper)
            #             adjusted_points.extend((new_x, new_y))
            #             inside = True

            #     if inside:
            #         crop_labels.append(f"{cls_id} " + " ".join(map(str, adjusted_points)))

            #     for points in adjusted_points:
            #         pol = Polygon(points)
            #         area = pol.area
            #         area_sum += area
            #         area_list.append(area)
            #         num_pols += 1
            #     # groundtruth stats
            #     vmd_value, coverage_percentage, rsf_value, _ = stats.calculate_statistics(diameter_list, image_area, contour_area)
            #     ground_truth_stats = stats(vmd_value, rsf_value, coverage_percentage, final_no_droplets, 0, 0)
            

            # Save the labels for the cropped image
            crop_label_filename = f"{output_folder}/label/{filename}_{i}_{j}.txt"
            with open(crop_label_filename, 'w') as f:
                f.write('\n'.join(crop_labels))
            print(f"Saved {crop_label_filename}")

        
def apply_yolo_segmentation(image_path, image, width_mm, width, height):
    im = cv2.imread(image_path)
    original_image = image.copy()

    detected_image = copy.copy(image)
    
    # predict image results
    results = model(image_path, conf=0.2)
    segmentation_result = results[0].masks.xy

    predicted_droplets = []

    # save each polygon
    area_list = []
    area_sum = 0
    num_pols = 0

    for polygon in segmentation_result:
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))

        if len(pts) > 0:
        
            cv2.drawContours(detected_image, [pts], -1, (0, 0, 255), thickness=1) 

            flattened_array = pts.reshape(-1, 2)
            coordinates = [tuple(point) for point in flattened_array]
            pol = Polygon(coordinates)
            area = pol.area
            area_sum += area
            area_list.append(area)
            num_pols += 1

            predicted_droplets.append(pts)

    polygons_with_centroids = [(polygon, calculate_centroid_yolo(polygon)) for polygon in predicted_droplets]
    sorted_polygons = sorted(polygons_with_centroids, key=lambda item: (item[1][0], item[1][1]))
        
    diameter_list = sorted(stats.area_to_diameter_micro(area_list, width, width_mm))
    # calculate statistics
    vmd_value, coverage_percentage, rsf_value, _ = stats.calculate_statistics(diameter_list, height*width, area_sum)

    cv2.imwrite(os.path.join("results\\metrics\\droplet", "final_seg.png"), detected_image)
    predicted_stats = stats(vmd_value, rsf_value, coverage_percentage, num_pols, 0, 0)

    return sorted_polygons, width, height, predicted_stats
# Example usage
divide_image_with_labels('data\\testing\\synthetic_dataset\\image','data\\testing\\synthetic_dataset\\label', 'data\\testing\\synthetic_dataset_square')

# Example usage
# divide_image('data\\synthetic_normal_dataset_new\\wsp\\image\\0.png', 'data\\testing\\crop_test')
