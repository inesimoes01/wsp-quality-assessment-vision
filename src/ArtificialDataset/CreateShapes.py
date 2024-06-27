import numpy as np   
import cv2
from matplotlib import pyplot as plt 
from shapely import geometry
import matplotlib.pyplot as plt

import ShapeList

class Shapes():
    def __init__(self):
        self.list_roi_shapes = []
        
        shapes = ShapeList.shapes

        for shape in shapes:
            # get the value of the points in a small area as to be able to easily scale the polygon
            scaled_points = np.array(self.convert_to_real_coordinates(shape, 512, 512))
            x, y, w, h = cv2.boundingRect(scaled_points)
            roi_points = np.array(self.get_roi_points(scaled_points, x, y))

            self.list_roi_shapes.append(roi_points)      
        

    def convert_to_real_coordinates(self, points, image_width, image_height):
        real_coordinates = []
        for point in points:
            percentage_x, percentage_y = point
            real_x = (percentage_x / 100) * image_width
            real_y = (percentage_y / 100) * image_height
            real_coordinates.append([int(real_x), int(real_y)])
        return real_coordinates

    def get_roi_points(self, points, x, y):
        roi_points = []
        for point in points:
            old_x, old_y = point
            new_x = old_x - x
            new_y = old_y - y
            roi_points.append([int(new_x), int(new_y)])
        return roi_points

    def resize_polygon(self, coords, target_diameter, isExpanding):
        _, _, w, h = cv2.boundingRect(coords)
        scale_factor = (1 - target_diameter / max(w, h)) / 2

        xs = [i[0] for i in coords]
        ys = [i[1] for i in coords]
        x_center = 0.5 * min(xs) + 0.5 * max(xs)
        y_center = 0.5 * min(ys) + 0.5 * max(ys)

        min_corner = geometry.Point(min(xs), min(ys))
        max_corner = geometry.Point(max(xs), max(ys))
        center = geometry.Point(x_center, y_center)
        distance = center.distance(min_corner) * scale_factor

        original_polygon = geometry.Polygon(coords)
        if isExpanding: new_polygon = original_polygon.buffer(distance)
        else: new_polygon = original_polygon.buffer(-distance)

        new_coords = list(new_polygon.exterior.coords)
        
        x, y = original_polygon.exterior.xy
        plt.plot(x,y)
        x, y = new_polygon.exterior.xy
        plt.plot(x,y)
        plt.axis('equal')
        plt.show()

        new_coords = np.array(new_coords, np.int32)
        new_coords = new_coords.reshape((-1, 1, 2))
        return new_coords









