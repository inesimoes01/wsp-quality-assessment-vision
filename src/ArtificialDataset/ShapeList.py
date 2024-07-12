import numpy as np
import cv2
import numpy as np
from DropletShape import DropletShape
from shapely.geometry import Point
from shapely import geometry
import shapes_variables
from itertools import combinations
import shape_list

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import translate

def convert_to_real_coordinates(points, image_width, image_height):
    real_coordinates = []
    for point in points:
        percentage_x, percentage_y = point
        real_x = (percentage_x / 100) * image_width
        real_y = (percentage_y / 100) * image_height
        real_coordinates.append([int(real_x), int(real_y)])
    return real_coordinates

def get_roi_points(points, x, y):
    roi_points = []
    for point in points:
        old_x, old_y = point
        new_x = old_x - x
        new_y = old_y - y
        roi_points.append([int(new_x), int(new_y)])
    return roi_points

# def get_max_distance(polygon):
#     points = list(polygon.exterior.coords)
#     max_distance = max(
#         point1.distance(point2)
#         for point1, point2 in combinations([Point(p) for p in points], 2)
#     )
#     return int(max_distance)

def create_new_shape_overlapped(index, polygon):
    #max_distance = get_max_distance(polygon)
    min_x, min_y, max_x, max_y = polygon.bounds
    max_distance = int(np.sqrt((max_y - min_y) ** 2 + (max_x - min_x) ** 2))

    new_coords = list(polygon.exterior.coords)
    new_coords = np.array(new_coords, np.int32)
    #new_coords = new_coords.reshape((-1, 1, 2))
    
    roi_points = np.array(get_roi_points(new_coords, min_x, min_y))
    roi_polygon = geometry.Polygon(roi_points)

    shape_for_official_list = DropletShape(index, max_x - min_x, max_y - min_y, max_distance, roi_polygon, roi_points)

    return shape_for_official_list

def save_all_shapes():
    list_roi_shapes = []
    polygons_by_size = {}
    
    rows, cols = 6, 5
    
    # fig, axes = plt.subplots(rows, cols, figsize=(15, 18))
    # axes = axes.flatten() 
    for i, shape in enumerate(shape_list.shapes):
    #for i, shape in enumerate(shapes_variables.shapes):
        # get the value of the points in a small area as to be able to easily scale the polygon
        scaled_points = np.array(convert_to_real_coordinates(shape, 512, 512))
        x, y, w, h = cv2.boundingRect(scaled_points)
        roi_points = np.array(get_roi_points(scaled_points, x, y))

        polygon = geometry.Polygon(roi_points)
        min_x, min_y, max_x, max_y = polygon.bounds
        max_distance = int(np.sqrt((max_y - min_y) ** 2 + (max_x - min_x) ** 2))
        
        #area = int(polygon.area)
        list_roi_shapes.append(DropletShape(i, w, h, max_distance, polygon, roi_points, roi_points))   

        if max_distance not in polygons_by_size:
            polygons_by_size[max_distance] = []
        polygons_by_size[max_distance].append(i)

        # x, y = polygon.exterior.xy
        # ax.fill(x, y, alpha=0.5, fc='blue', ec='black')
        
        # # Calculate the centroid for placing the ID
        # centroid = polygon.centroid
        # # Place the ID text slightly offset from the centroid
        # ax.text(centroid.x, centroid.y + 0.1, str(i), ha='center', fontsize=12, color='red')

        # # Set equal aspect ratio and remove axes for clarity
        # ax.set_aspect('equal', 'box')
        # ax.axis('off')

    # plt.tight_layout()
    #plt.show()
    
    return list_roi_shapes, polygons_by_size

