import sys
import numpy as np
import cv2
from shapely import Polygon

sys.path.insert(0, 'src')
import Common.Util as FoldersUtil
import Common.config as config
import Segmentation.droplet.ccv.Segmentation_CCV as seg
from Common.Statistics import Statistics as stats

def calculate_predicted_statistics(predicted_shapes, width_px, height_px, width_mm):
    total_droplet_area = 0
    total_no_droplets = 0
    list_droplet_area = []
    list_polygons = []
    
    # get general information from the polygons
    predicted_shapes = [item[0] for item in predicted_shapes]
    for pred in predicted_shapes:
        polygon = Polygon(pred)
        if polygon.area < 10000:
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
            polygon = polygon.buffer(0)  
        for j, other_polygon in enumerate(list_polygons):
            if i != j:
                if not other_polygon.is_valid:
                    other_polygon = other_polygon.buffer(0)
                if polygon.intersects(other_polygon):
                    overlapping_polygons.append(i)
                    overlapping_polygons.append(j)

    no_droplets_overlapped = len(overlapping_polygons)
    overlaped_percentage = no_droplets_overlapped / total_no_droplets * 100 if total_no_droplets > 0 else 0

    # calculate statistics
    vmd_value, coverage_percentage, rsf_value, _ = stats.calculate_statistics(diameter_list, height_px * width_px, total_droplet_area)    
    predicted_stats = stats(vmd_value, rsf_value, coverage_percentage, total_no_droplets, no_droplets_overlapped, overlaped_percentage)
    
    return predicted_stats

def save_shapes_to_yolo_label(label_path, droplets_detected, width, height):
    with open(label_path, "w") as file:
        for droplet in droplets_detected:

            # write each one of the points in the label file
            normalized_points = []
            for point in droplet:
                x, y = point
                # Normalize coordinates by dividing by the image dimensions and converting to percentages
                x_normalized = x / width
                y_normalized = y / height
                normalized_points.append((x_normalized, y_normalized))

            yolo_line = f"0"
            for (x_norm, y_norm) in normalized_points:
                yolo_line += f" {x_norm:.10f} {y_norm:.10f}"
            file.write(yolo_line + "\n")      



def compute_ccv_segmentation(image_colors, image_gray, filename, label_path, width, height):
    predicted_seg:seg.Segmentation_CCV = seg.Segmentation_CCV(image_colors, image_gray, filename, 
                                                save_image_steps = False, 
                                                segmentation_method = 0, 
                                                dataset_results_folder = "")

    # save yolo labels
    droplets_detected = []
    for droplet in predicted_seg.droplets_data:
        mask = np.zeros_like(image_gray)

        # contour shape
        if droplet.overlappedIDs == []:
            cv2.drawContours(mask, [predicted_seg.droplet_shapes.get(droplet.id)], -1, (255), cv2.FILLED)
            contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                points = contour.reshape(-1, 2)
                droplets_detected.append(points)
        
        # perfect circle
        else:
            cv2.circle(mask, (droplet.center_x, droplet.center_y), droplet.radius, (255), cv2.FILLED)
            contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                points = contour.reshape(-1, 2)
                droplets_detected.append(points)
   
    save_shapes_to_yolo_label(label_path, droplets_detected, width, height)
