import numpy as np
import sys
sys.path.insert(0, 'src/common')
import config
from shapely import geometry
from matplotlib import pyplot as plt 
import cv2
import CreateBackground
from PIL import Image, ImageDraw
import ShapeList
import re
from shapely.ops import nearest_points
from shapely import geometry, Point
from DropletShape import DropletShape
import random

brown_colors = [ '#190e00', '#1f150c', '#0e0f13', '#131514', '#1b0f19', '#14141c', '#1b141c', '#131522', '#0c0d22', '#181123', '#141325', '#1c1527', '#181729', '#1b1a2a']
dark_blue_colors=['#09082a', '#0f0c2b', '#181130', '#0a0a30', '#030430', '#160e33', '#000233', '#191935', '#0e0d35', '#0e0d35', '#140c35', '#181736', '#060838', '#060838', '#060a3a', '#060a3a', '#11143d', '#0d0b3d', '#0e1040', '#030444', '#060845', '#0d0b4a', '#0b094a', '#070654']
light_blue_color = ['#2c2bb7', '#2e2db5', '#2524ac', '#2d29a2', '#352ea0', '#2c2897', '#272595', '#221d91', '#18107f', '#181872']

def create_concentric_polygons(coords, scale_factors, min_corner, center):
    original_polygon = geometry.Polygon(coords)
    polygons = []
    for scale in scale_factors:
        distance = center.distance(min_corner) * scale
        new_polygon = original_polygon.buffer(-distance)
        polygons.append(new_polygon)
    return polygons

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    return rgb

brown_rgb = [hex_to_rgb(color) for color in brown_colors]
light_blue_rgb = [hex_to_rgb(color) for color in light_blue_color]
dark_blue_rgb = [hex_to_rgb(color) for color in dark_blue_colors]

def interpolate_color(color1, color2, t):
    r = int(color1[0] * (1 - t) + color2[0] * t)
    g = int(color1[1] * (1 - t) + color2[1] * t)
    b = int(color1[2] * (1 - t) + color2[2] * t)
    return (r, g, b)


def draw_polygon(img, size, center_x, center_y, shape_instance:DropletShape):
    if size < config.DROPLET_COLOR_THRESHOLD_1:
        draw_three_layer_polygon(img, shape_instance, center_x, center_y)
    elif size < config.DROPLET_COLOR_THRESHOLD_2:
        draw_three_layer_polygon(img, shape_instance, center_x, center_y)
    else:
        draw_three_layer_polygon(img, shape_instance, center_x, center_y)

def draw_three_layer_polygon(img, shape:DropletShape, center_x, center_y):
    image_height, image_width = img.shape[:2]
    
    original_polygon = geometry.Polygon(shape.roi_points)
    xs = [i[0] for i in shape.roi_points]
    ys = [i[1] for i in shape.roi_points]
    roi_center_x = 0.5 * min(xs) + 0.5 * max(xs)
    roi_center_y = 0.5 * min(ys) + 0.5 * max(ys)
    center = geometry.Point(roi_center_x, roi_center_y)

    translation_vector_x = int(center_x - roi_center_x)
    translation_vector_y = int(center_y - roi_center_y)

    # check all the points in the roi 
    for y in range(shape.height):
        for x in range(shape.width):
            p = Point(x, y)
            if original_polygon.contains(p):

                # get distance to the nearest edge of the polygon
                nearest_edge = nearest_points(p, original_polygon.exterior)[1]

                # normalize the distance: distance of the p to the nearest edge 
                # divided by the distance of the nearest edge to the center
                distance = (p.distance(center)) / (center.distance(nearest_edge))

                # depending on the distance to the nearest edge, we color it differently
                if distance < 0.5:  # Inner part (0 to 60% of the radius)
                    t = distance / 0.5  # Scale t to [0, 1]
                    color_index = int(t * (len(light_blue_rgb) - 1))
                    t = (t * (len(light_blue_rgb) - 1)) - color_index
                    color1 = light_blue_rgb[color_index % len(light_blue_rgb)]
                    color2 = light_blue_rgb[(color_index + 1) % len(light_blue_rgb)]
                    interpolated_color = interpolate_color(color1, color2, t)
                
                elif distance < 0.7:  # middle part (60% to 70% of the radius)
                    t = (distance - 0.5) / 0.2  # Scale t to [0, 1]
                    inner_color_index = len(light_blue_rgb) - 1
                    middle_color_index = int(t * (len(dark_blue_rgb) - 1))
                    t_middle = t * (len(dark_blue_rgb) - 1) - middle_color_index

                    inner_color = light_blue_rgb[inner_color_index]
                    middle_color1 = dark_blue_rgb[middle_color_index % len(dark_blue_rgb)]
                    middle_color2 = dark_blue_rgb[(middle_color_index + 1) % len(dark_blue_rgb)]

                    middle_interpolated_color = interpolate_color(middle_color1, middle_color2, t_middle)
                    interpolated_color = interpolate_color(inner_color, middle_interpolated_color, t)
                            
                elif distance < 0.85:  # outer part (70% to 90% of the radius)
                    t = (distance - 0.7) / 0.15 # Scale t to [0, 1]
                    middle_color_index = len(dark_blue_rgb) - 1
                    outer_color_index = int(t * (len(brown_rgb) - 1))
                    t_outer = t * (len(brown_rgb) - 1) - outer_color_index

                    middle_color = dark_blue_rgb[middle_color_index]
                    outer_color1 = brown_rgb[outer_color_index % len(brown_rgb)]
                    outer_color2 = brown_rgb[(outer_color_index + 1) % len(brown_rgb)]

                    outer_interpolated_color = interpolate_color(outer_color1, outer_color2, t_outer)
                    interpolated_color = interpolate_color(middle_color, outer_interpolated_color, t)
                
                else : # make sure to blend with background
                    t = (distance - 0.85) / 0.15
                    color_index = int(t * (len(brown_rgb) - 1))
                    color1 = brown_rgb[color_index % len(brown_rgb)]
                    color2 = (255, 255, 0)
                    interpolated_color = interpolate_color(color1, color2, t)

                if y + translation_vector_y < image_height and x + translation_vector_x < image_width:
                    img[y + translation_vector_y, x + translation_vector_x] = np.array(interpolated_color).astype(np.uint8)

    
def choose_polygon(polygons_by_size, size):
    chosen_id = -1
    if size in polygons_by_size:
        chosen_id = random.choice(polygons_by_size[size])
    
    if chosen_id == -1:
        chosen_id = random.choice(polygons_by_size[8])

    return chosen_id


def draw_perfect_circle(img, center, radius, inner_colors, middle_colors, outer_colors, background_color):
    if radius < config.DROPLET_COLOR_THRESHOLD_1:
        draw_one_layer_circle(img, center, radius, outer_colors, background_color)
    elif radius < config.DROPLET_COLOR_THRESHOLD_2:
        draw_two_layer_circle(img, center, radius, middle_colors, outer_colors, background_color)
    else:
        draw_three_layer_circle(img, center, radius, inner_colors, middle_colors, outer_colors, background_color)   

def draw_three_layer_circle(img, center, radius, inner_colors, middle_colors, outer_colors, background_color):
    height, width = img.shape[:2]
    for y in range(center[1] - radius, center[1] + radius):
        for x in range(center[0] - radius, center[0] + radius):
            if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
                # Calculate distance from center normalized to [0, 1]
                distance = np.sqrt((x - center[0])**2 + (y - center[1])**2) / radius
                
                if distance < 0.6:  # Inner part (0 to 60% of the radius)
                    t = distance / 0.6  # Scale t to [0, 1]
                    color_index = int(t * (len(inner_colors) - 1))
                    t = (t * (len(inner_colors) - 1)) - color_index
                    color1 = inner_colors[color_index % len(inner_colors)]
                    color2 = inner_colors[(color_index + 1) % len(inner_colors)]
                    interpolated_color = interpolate_color(color1, color2, t)
                
                elif distance < 0.7:  # middle part (60% to 70% of the radius)
                    t = (distance - 0.6) / 0.1  # Scale t to [0, 1]
                    inner_color_index = len(inner_colors) - 1
                    middle_color_index = int(t * (len(middle_colors) - 1))
                    t_middle = t * (len(middle_colors) - 1) - middle_color_index

                    inner_color = inner_colors[inner_color_index]
                    middle_color1 = middle_colors[middle_color_index % len(middle_colors)]
                    middle_color2 = middle_colors[(middle_color_index + 1) % len(middle_colors)]

                    middle_interpolated_color = interpolate_color(middle_color1, middle_color2, t_middle)
                    interpolated_color = interpolate_color(inner_color, middle_interpolated_color, t)
                               
                elif distance < 0.9:  # outer part (70% to 90% of the radius)
                    t = (distance - 0.7) / 0.2  # Scale t to [0, 1]
                    middle_color_index = len(middle_colors) - 1
                    outer_color_index = int(t * (len(outer_colors) - 1))
                    t_outer = t * (len(outer_colors) - 1) - outer_color_index

                    middle_color = middle_colors[middle_color_index]
                    outer_color1 = outer_colors[outer_color_index % len(outer_colors)]
                    outer_color2 = outer_colors[(outer_color_index + 1) % len(outer_colors)]

                    outer_interpolated_color = interpolate_color(outer_color1, outer_color2, t_outer)
                    interpolated_color = interpolate_color(middle_color, outer_interpolated_color, t)
                
                else : # make sure to blend with background
                    t = (distance - 0.9) / 0.1  
                    color_index = int(t * (len(outer_colors) - 1))
                    color1 = outer_colors[color_index % len(outer_colors)]
                    color2 = background_color
                    interpolated_color = interpolate_color(color1, color2, t)
                if y < height and x < width:
                # Set pixel color in the image
                    img[y, x] = np.array(interpolated_color).astype(np.uint8)

def draw_two_layer_circle(img, center, radius, middle_colors, outer_colors, background_color):
    height, width = img.shape[:2]
    for y in range(center[1] - radius, center[1] + radius):
        for x in range(center[0] - radius, center[0] + radius):
            if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
                # calculate distance from center normalized to [0, 1]
                distance = np.sqrt((x - center[0])**2 + (y - center[1])**2) / radius
                
                if distance < 0.6:  # inner part (0 to 60% of the radius)
                    t = distance / 0.6  # Scale t to [0, 1]
                    color_index = int(t * (len(middle_colors) - 1))
                    t = (t * (len(middle_colors) - 1)) - color_index
                    color1 = middle_colors[color_index % len(middle_colors)]
                    color2 = middle_colors[(color_index + 1) % len(middle_colors)]
                    interpolated_color = interpolate_color(color1, color2, t)
                
                else : # make sure to blend with background
                    t = (distance - 0.6) / 0.4
                    color_index = int(t * (len(outer_colors) - 1))
                    color1 = outer_colors[color_index % len(outer_colors)]
                    color2 = background_color
                    interpolated_color = interpolate_color(color1, color2, t)

                if y < height and x < width:
                    img[y, x] = np.array(interpolated_color).astype(np.uint8)

def draw_one_layer_circle(img, center, radius, outer_colors, background_color):
    height, width = img.shape[:2]
    for y in range(center[1] - radius, center[1] + radius):
        for x in range(center[0] - radius, center[0] + radius):
            if (x - center[0])**2 + (y - center[1])**2 <= radius**2:

                distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                # Calculate color intensity based on distance
                intensity = 1 - distance / radius

                index = np.random.randint(0, len(outer_colors))
                varied_color = tuple(int(outer_colors[index][c] * intensity) for c in range(3))

                if y < height and x < width:
                # Set pixel color in the image
                    img[y, x] = np.array(varied_color).astype(np.uint8)