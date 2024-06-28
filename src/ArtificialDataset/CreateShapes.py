import numpy as np   
import cv2
from matplotlib import pyplot as plt 
from shapely import geometry, Point
import matplotlib.pyplot as plt
import CreateBackground
from PIL import Image, ImageDraw
import ShapeList
import re
from shapely.ops import nearest_points


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    return rgb

brown_colors = [ '#190e00', '#1f150c', '#0e0f13', '#131514', '#1b0f19', '#14141c', '#1b141c', '#131522', '#0c0d22', '#181123', '#141325', '#1c1527', '#181729', '#1b1a2a']
dark_blue_colors=['#09082a', '#0f0c2b', '#181130', '#0a0a30', '#030430', '#160e33', '#000233', '#191935', '#0e0d35', '#0e0d35', '#140c35', '#181736', '#060838', '#060838', '#060a3a', '#060a3a', '#11143d', '#0d0b3d', '#0e1040', '#030444', '#060845', '#0d0b4a', '#0b094a', '#070654']
light_blue_color = ['#2c2bb7', '#2e2db5', '#2524ac', '#2d29a2', '#352ea0', '#2c2897', '#272595', '#221d91', '#18107f', '#181872']

brown_rgb = [hex_to_rgb(color) for color in brown_colors]
light_blue_rgb = [hex_to_rgb(color) for color in light_blue_color]
dark_blue_rgb = [hex_to_rgb(color) for color in dark_blue_colors]

def interpolate_color(color1, color2, t):
    r = int(color1[0] * (1 - t) + color2[0] * t)
    g = int(color1[1] * (1 - t) + color2[1] * t)
    b = int(color1[2] * (1 - t) + color2[2] * t)
    return (r, g, b)

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

            # yellow = (255, 255, 0)  # BGR for yellow
            # img = np.full((50, 50, 3), yellow, dtype=np.uint8)

            # cv2.fillPoly(img, roi_points, (255, 0, 0))
            # plt.imshow(img)
            # plt.show()





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
        scale_factor = (abs(target_diameter - max(w, h)) + 1)  / 10 
        print(scale_factor)
        xs = [i[0] for i in coords]
        ys = [i[1] for i in coords]
        x_center = 0.5 * min(xs) + 0.5 * max(xs)
        y_center = 0.5 * min(ys) + 0.5 * max(ys)

        min_corner = geometry.Point(min(xs), min(ys))
        center = geometry.Point(x_center, y_center)
        distance = center.distance(min_corner) *scale_factor

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




# yellow = (255, 255, 0)  # BGR for yellow
# img = np.full((50, 50, 3), yellow, dtype=np.uint8)

# Shapes()


    # def __init__(self, img):
    #     self.list_roi_shapes = []
        
    #     shapes = ShapeList.shapes

    #     for shape in shapes:
    #         # get the value of the points in a small area as to be able to easily scale the polygon
    #         scaled_points = np.array(self.convert_to_real_coordinates(shape, 512, 512))
    #         x, y, w, h = cv2.boundingRect(scaled_points)
    #         roi_points = np.array(self.get_roi_points(scaled_points, x, y))

    #         self.list_roi_shapes.append(roi_points)   

    #         #roi_points = self.resize_polygon(roi_points, 14, True)

    #         scale_factors = np.arange(0.1, 0.5, 0.02).tolist()
            
    #         xs = [i[0] for i in roi_points]
    #         ys = [i[1] for i in roi_points]
    #         x_center = 0.5 * min(xs) + 0.5 * max(xs)
    #         y_center = 0.5 * min(ys) + 0.5 * max(ys)

    #         min_corner = geometry.Point(min(xs), min(ys))
    #         center = geometry.Point(x_center, y_center)

    #         concentric_polygons = []
    #         original_polygon = geometry.Polygon(roi_points)

    #         # color the biggest polygon
    #         t = (max(h, w) - 0.9)
    #         color_index = int(t * (len(brown_rgb) - 1))
    #         color1 = brown_rgb[5 % len(brown_rgb)]
    #         color2 = (255, 255, 0)
    #         interpolated_color = interpolate_color(color1, color2, t)
    #         cv2.fillPoly(img, [roi_points], interpolated_color)
            

    #         for i, scale in enumerate(scale_factors):
    #             distance = center.distance(min_corner) * scale
    #             new_polygon = original_polygon.buffer(-distance)

    #             if new_polygon is not None:
    #                 concentric_polygons.append(new_polygon)
            
    #             if i < 3:
    #                 t = (distance - 0.8) / 0.2
    #                 color_index = int(t * (len(brown_rgb) - 1))
    #                 color1 = brown_rgb[color_index % len(brown_rgb)]
    #                 color2 = (255, 255, 0)
    #                 interpolated_color = interpolate_color(color1, color2, t)

    #             elif i < 5:
    #                 t = (distance - 0.7) / 0.2  # Scale t to [0, 1]
    #                 middle_color_index = len(dark_blue_rgb) - 1
    #                 outer_color_index = int(t * (len(brown_rgb) - 1))
    #                 t_outer = t * (len(brown_rgb) - 1) - outer_color_index

    #                 middle_color = dark_blue_rgb[middle_color_index]
    #                 outer_color1 = brown_rgb[outer_color_index % len(brown_rgb)]
    #                 outer_color2 = brown_rgb[(outer_color_index + 1) % len(brown_rgb)]

    #                 outer_interpolated_color = interpolate_color(outer_color1, outer_color2, t_outer)
    #                 interpolated_color = interpolate_color(middle_color, outer_interpolated_color, t)
                             
    #             elif i < 7:
    #                 t = (distance - 0.6) / 0.1  # Scale t to [0, 1]
    #                 inner_color_index = len(light_blue_rgb) - 1
    #                 middle_color_index = int(t * (len(dark_blue_rgb) - 1))
    #                 t_middle = t * (len(dark_blue_rgb) - 1) - middle_color_index

    #                 inner_color = light_blue_rgb[inner_color_index]
    #                 middle_color1 = dark_blue_rgb[middle_color_index % len(dark_blue_rgb)]
    #                 middle_color2 = dark_blue_rgb[(middle_color_index + 1) % len(dark_blue_rgb)]

    #                 middle_interpolated_color = interpolate_color(middle_color1, middle_color2, t_middle)
    #                 interpolated_color = interpolate_color(inner_color, middle_interpolated_color, t)
    
    #             else : # make sure to blend with background
    #                 t = distance / 0.6  
    #                 color_index = int(t * (len(light_blue_rgb) - 1))
    #                 t = (t * (len(light_blue_rgb) - 1)) - color_index
    #                 color1 = light_blue_rgb[color_index % len(light_blue_rgb)]
    #                 color2 = light_blue_rgb[(color_index + 1) % len(light_blue_rgb)]
    #                 interpolated_color = interpolate_color(color1, color2, t)


    #             pts = np.array(list(new_polygon.exterior.coords), np.int32)
    #             pts = pts.reshape((-1, 1, 2))

    #             cv2.fillPoly(img, [pts], interpolated_color)

    #         plt.imshow(img)
    #         plt.show()
