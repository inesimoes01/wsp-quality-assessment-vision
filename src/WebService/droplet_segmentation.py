import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for servers

import os
import sys
import cv2
import numpy as np
import time
import random
import colorsys
import skimage
from shapely import MultiPolygon, unary_union, Polygon
from matplotlib import pyplot as plt

from matplotlib.patches import Polygon as polygon_plot
from ultralytics import YOLO

from cellpose import models, core, io

sys.path.insert(0, 'src/')
import Segmentation.droplet.ccv.Segmentation_CCV as segmentation
import WebService.paper_segmentation as paper_segmentation
import Common.config as config
import Segmentation.dataset.DatasetUtil as dataset_util
from Common.Statistics import Statistics as stats

random.seed(42)

def load_and_convert_image(file_path):
    img = skimage.io.imread(file_path)
    if img.shape[-1] == 4:  # If the image has an alpha channel
        img = skimage.color.rgba2rgb(img)  # Convert RGBA to RGB
    return skimage.color.rgb2gray(img) 

def _calculate_stats(predicted_shapes, width_px, height_px, width_mm):
    total_droplet_area = 0
    total_no_droplets = 0
    list_droplet_area = []
    list_polygons = []
    
    # get general information from the polygons
    #predicted_shapes = [item[0] for item in predicted_shapes]
    for polygon in predicted_shapes:
      
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

    return vmd_value, rsf_value, coverage_percentage, total_no_droplets, diameter_list

def _random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def _handle_edge_cases(predicted_droplets, distance_threshold):
    joined_droplets = []
    final_droplet_polygons = []

    for (droplet, i, isEdge) in predicted_droplets:
        if i not in joined_droplets:
            # if the droplet is on the edge try to find the rest of the droplet
            if isEdge:
                for (droplet2, j, isEdge) in predicted_droplets:
                    if isEdge and i != j and j not in joined_droplets:
        
                        # check if the polygons are close enough to be considered the same
                        poly1 = Polygon(droplet)
                        poly2 = Polygon(droplet2)

                        if poly1.is_valid and poly2.is_valid:
                            if poly1.intersects(poly2) or poly1.distance(poly2) < distance_threshold:
                                # merge polygons
                                merged_polygon = unary_union([poly1, poly2])

                                if isinstance(merged_polygon, MultiPolygon): continue
                            
                                joined_droplets.append(i)
                                joined_droplets.append(j)

                                final_droplet_polygons.append(merged_polygon)
                                break  
                        else: 
                            break

                if i not in joined_droplets:
                    final_droplet_polygons.append(droplet)
                
            # if not save it as is
            else:
                final_droplet_polygons.append(droplet)

def _create_segmentation_image(filename, polygons, image, final_segmentation_file):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Display the image
    colors = _random_colors(len(polygons))

    for i, pol in enumerate(polygons):
        color = colors[i]
     
        if pol.area < 10000:   
            exterior_coords = list(pol.exterior.coords)
            poly_patch = polygon_plot(exterior_coords, edgecolor=color, facecolor=color, alpha=0.5)
            ax.add_patch(poly_patch)
        

    plt.axis('off')
    save_path = os.path.join(final_segmentation_file, "segmentation_" + filename + ".png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close(fig)  

# def _create_segmentation_image(filename, polygons, image, final_segmentation_file):
#     fig, ax = plt.subplots(1, figsize=(10, 10))
#     ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Display the image

#     colors = _random_colors(len(polygons))
    
#     polygons = []
#     for i, pol in enumerate(polygons):
#         color = colors[i]

#         if pol.area < 10000:
#             poly_patch = polygon_plot(pol, edgecolor=color, facecolor=color, alpha=0.5)
#             ax.add_patch(poly_patch)

#     plt.axis('off')
#     save_path = os.path.join(final_segmentation_file, "segmentation_" + filename + ".png")
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=600)

def _compute_cellpose_segmentation(filename, image_path, preannotation_path, diameter_median):
    print(image_path)
    use_GPU = core.use_gpu()

    files = [image_path, image_path]
    
    imgs = [load_and_convert_image(f) for f in files]
    nimg = len(imgs)
    imgs_2D = imgs[:-1]

    model = models.Cellpose(gpu = use_GPU, model_type = 'cyto3')

    channels = [[0, 0], [0, 0]]
    diameter = diameter_median
    masks, flows, styles, diams = model.eval(imgs_2D, diameter=diameter, flow_threshold=None, channels=channels)
    io.save_masks(imgs_2D, masks, flows, files, png=True, savedir = preannotation_path, save_txt = True)

    with open(os.path.join(preannotation_path, "undistorted_" + filename + "_cp_outlines.txt"), 'r') as file:
        lines = file.readlines()

    polygons = []

    for i, line in enumerate(lines):
        points = list(map(int, line.strip().split(',')))
        coordinates = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
        contour = np.array(coordinates, dtype=np.int32)

        polygons.append(Polygon(contour))

    return polygons            

def _compute_yolo_segmentation(image, image_path, height, width, yolo_model, last_index, x_offset, y_offset):
    predicted_droplets_adjusted = []
    predicted_droplets_adjusted_with_edges = []

    results = yolo_model(image_path, conf=0.3)

    if results[0].masks:
        segmentation_result = results[0].masks.xy
       
        detected_pts = []

        for polygon in segmentation_result:
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            detected_pts.append(pts)

        for coords in detected_pts:
            adjusted_coords = []
            for point in coords:
                x, y = point[0]
                adjusted_coords.append([x + x_offset, y + y_offset])
            if adjusted_coords != [] and len(adjusted_coords) >= 4:
                predicted_droplets_adjusted.append(np.array(adjusted_coords, dtype=np.int32))

        # check which droplets are on the edge
        edge_zone_width = 5
        for i, polygon in enumerate(predicted_droplets_adjusted):
            isEdge = False
            cv2.drawContours(image, [polygon], -1, (255, 0, 0), 1)

            for point in polygon:
                if (point[0] < edge_zone_width or 
                    point[0] > width - edge_zone_width or
                    point[1] < edge_zone_width or 
                    point[1] > height - edge_zone_width):

                    isEdge = True
        
            predicted_droplets_adjusted_with_edges.append((polygon, i + last_index, isEdge))     
            
    return predicted_droplets_adjusted_with_edges, len(predicted_droplets_adjusted_with_edges) + last_index

def droplet_segmentation_cellpose(image_colors, image_path, path_auxiliary, filename, paper_width, diameter_median, width, height):
    print("Detecting droplets...")
    droplets_detected = _compute_cellpose_segmentation(filename, image_path, path_auxiliary, diameter_median)
    
    print("Found", len(droplets_detected), "droplets. Calculating statistics...")
    vmd_value, rsf_value, coverage_percentage, total_no_droplets, diameter_list = _calculate_stats(droplets_detected, width, height, paper_width * 10)
    _create_segmentation_image(filename, droplets_detected, image_colors, path_auxiliary)

    print("Finished.")
    return vmd_value, rsf_value, coverage_percentage, total_no_droplets, diameter_list


def droplet_segmentation_yolo(image_colors, path_auxiliary, yolo_model, filename, paper_width):
    # divide into squares
    total_predicted_droplets = []
    squares = dataset_util.divide_image_into_squares_simple(image_colors)
    last_index = 0

    for (square, x_offset, y_offset, (x, y, _)) in squares:
        # save temporarly for yolo segmentation and apply the segmentation
        image_path = os.path.join(path_auxiliary, "temp.png")
        cv2.imwrite(image_path, square)

        square_image = cv2.imread(image_path)
        width, height = square_image.shape[:2]
        
        predicted_droplets, last_index = _compute_yolo_segmentation(square, image_path, height, width, yolo_model, last_index, x_offset, y_offset)
    
        total_predicted_droplets.extend(predicted_droplets)

    final_droplets = _handle_edge_cases(total_predicted_droplets, 5)

    _calculate_stats(final_droplets, width, height, paper_width)

    _create_segmentation_image(filename, final_droplets, image_colors, path_auxiliary)

def droplet_segmentation_ccv(image_colors, image_gray, filename, paper_width, width, height, path_auxiliary):
    print("Detecting droplets...")
    predicted_seg:segmentation.Segmentation_CCV = segmentation.Segmentation_CCV(image_colors, image_gray, filename, 
                                                save_image_steps = False, 
                                                segmentation_method = 0, 
                                                dataset_results_folder = path_auxiliary)
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
                if len(points) > 4:
                    droplets_detected.append(Polygon(points))
        
        # perfect circle
        else:
            cv2.circle(mask, (droplet.center_x, droplet.center_y), droplet.radius, (255), cv2.FILLED)
            contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                points = contour.reshape(-1, 2)
                if len(points) > 4:
                    droplets_detected.append(Polygon(points))

    print("Found", len(droplets_detected), "droplets. Calculating statistics...")
    vmd_value, rsf_value, coverage_percentage, total_no_droplets, diameter_list = _calculate_stats(droplets_detected, width, height, paper_width * 10)
    _create_segmentation_image(filename, droplets_detected, image_colors, path_auxiliary)

    print("Finished.")
    return vmd_value, rsf_value, coverage_percentage, total_no_droplets, diameter_list