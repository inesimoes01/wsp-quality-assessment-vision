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
from shapely.affinity import translate
from shapely import geometry, Point
from DropletShape import DropletShape
import random
# #brown_colors = ['#1b1a2a', '#181729', '#1c1527', '#141325', '#181123', '#0c0d22', '#131522', '#1b141c', '#14141c', '#1b0f19', '#131514', '#0e0f13', '#1f150c', '#190e00']
# # from dark to light
brown_colors = [ '#190e00', '#1f150c', '#0e0f13', '#131514', '#1b0f19', '#14141c', '#1b141c', '#131522', '#0c0d22', '#181123', '#141325', '#1c1527', '#181729', '#1b1a2a']
# # from dark to light
dark_blue_colors=['#09082a', '#0f0c2b', '#181130', '#0a0a30', '#030430', '#160e33', '#000233', '#191935', '#0e0d35', '#0e0d35', '#140c35', '#181736', '#060838', '#060838', '#060a3a', '#060a3a', '#11143d', '#0d0b3d', '#0e1040', '#030444', '#060845', '#0d0b4a', '#0b094a', '#070654']
# #dark_blue_colors = ['#070654', '#0b094a', '#0d0b4a', '#060845', '#030444', '#0e1040', '#0d0b3d', '#11143d', '#060a3a', '#060a3a', '#060838', '#060838', '#181736', '#140c35', '#0e0d35', '#0e0d35', '#191935', '#000233', '#160e33', '#030430', '#0a0a30', '#181130', '#0f0c2b', '#09082a']
# # from light to dark
# #light_blue_color = ['#2c2bb7', '#2e2db5', '#2524ac', '#2d29a2', '#352ea0', '#2c2897', '#272595', '#221d91', '#18107f', '#181872']
# # from dark to light
light_blue_color = ['#181872', '#18107f', '#221d91', '#272595', '#2c2897', '#352ea0', '#2d29a2', '#2524ac', '#2e2db5', '#2c2bb7']

# brown_colors = [ '#220000', '#2a0700', '#330000', '#4a1800', '#4c2000', '#512600', '#542900', '#552809']
# dark_blue_colors = ['#5b1a30', '#4e0634', '#5e0b35', '#651f37', '#5e223b', '#65203f', '#65203f', '#571241']
# light_blue_color = ['#63219f',  '#642aa5',  '#7f24a5',  '#812ba6',  '#6a0fa8',  '#690eab',  '#7b25ac',  '#7530ad',  '#752eb0',  '#7e22b1',  '#8b27bb',  '#732ebd',  '#8e23bf',  '#7326c0',  '#853fc3',  '#812ec6',  '#8125c6',  '#6b1ec6',  '#8727c7',  '#892ace',  '#742dd1',  '#8038e4',  '#aa52e6',  '#a248e6']

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
    return [r, g, b]

def paint_polygon(img, polygon):
    min_x, min_y, max_x, max_y = polygon.bounds
    size = int(np.sqrt((max_y - min_y) ** 2 + (max_x - min_x) ** 2))

    points = list(polygon.exterior.coords)
    points = np.array(points, np.int32)

    image_height, image_width = img.shape[:2]
    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(points)

    # get the background color from the center coordinate to blend the droplet that is going to be drawn
    background_color = img[min(y_rect + h_rect, image_height-1), min(x_rect + w_rect, image_width-1)]

    # paint the shape pixel by pixel
    for x in range(max(x_rect, 0), min(x_rect + w_rect, image_width)):
        for y in range(max(y_rect, 0), min(y_rect + h_rect, image_height)):
            p = Point(x, y)
            if p.intersects(polygon):

                # get distance from the current point to the nearest edge of the polygon
                nearest_edge = nearest_points(p, polygon.exterior)[1]
                distance = abs((p.distance(nearest_edge)))
        

                # depending on the distance to the closest border, paint the pixel a different color
                if distance < 1:
                    t = (distance) / 1
                    color_index = int(t * (len(brown_rgb) - 1))
                    color1 = brown_rgb[color_index % len(brown_rgb)] 
                    color2 = background_color
                    interpolated_color = interpolate_color(color1, color2, t)
                    interpolated_color = tuple(int(interpolated_color[c] * 0.9) for c in range(3))
                    
                
                elif distance < 2:
                    t = (distance - 1) / 1
                    color_index = int(t * (len(brown_rgb) - 1))
                    color1 = brown_rgb[color_index % len(brown_rgb)]
                    color2 = brown_rgb[(color_index + 1) % len(brown_rgb)]
                    interpolated_color = interpolate_color(color1, color2, t)
                    (r, g, b) = interpolated_color
                    interpolated_color = tuple(int(interpolated_color[c] * 0.9) for c in range(3))

                elif distance < 3:
                    t = (distance - 2) / 1
                    middle_color_index = len(dark_blue_rgb) - 1
                    outer_color_index = int(t * (len(brown_rgb) - 1))
                    t_outer = t * (len(brown_rgb) - 1) - outer_color_index

                    middle_color = dark_blue_rgb[middle_color_index]
                    outer_color1 = brown_rgb[outer_color_index % len(brown_rgb)]
                    outer_color2 = brown_rgb[(outer_color_index + 1) % len(brown_rgb)]

                    outer_interpolated_color = interpolate_color(outer_color1, outer_color2, t_outer)
                    interpolated_color = interpolate_color(middle_color, outer_interpolated_color, t)
                    interpolated_color = tuple(int(interpolated_color[c] * 0.9) for c in range(3))

                elif distance < 4:
                    t = (distance - 3) / 1 
                    inner_color_index = len(light_blue_rgb) - 1
                    middle_color_index = int(t * (len(dark_blue_rgb) - 1))
                    t_middle = t * (len(dark_blue_rgb) - 1) - middle_color_index

                    inner_color = light_blue_rgb[inner_color_index]
                    middle_color1 = dark_blue_rgb[middle_color_index % len(dark_blue_rgb)]
                    middle_color2 = dark_blue_rgb[(middle_color_index + 1) % len(dark_blue_rgb)]

                    middle_interpolated_color = interpolate_color(middle_color1, middle_color2, t_middle)
                    interpolated_color = interpolate_color(inner_color, middle_interpolated_color, t)
                
                else:
                    t = (distance - 4) / size 
                    color_index = int(t * (len(light_blue_rgb) - 1))
                    t = (t * (len(light_blue_rgb) - 1)) - color_index
                    color1 = light_blue_rgb[color_index % len(light_blue_rgb)]
                    color2 = light_blue_rgb[(color_index + 1) % len(light_blue_rgb)]
                    interpolated_color = interpolate_color(color1, color2, t)
                

                if y < image_height and x < image_width:
                    img[y, x] = np.array(interpolated_color).astype(np.uint8)
    



def get_shape_translation(shape:DropletShape, center_x, center_y):
    xs = [i[0] for i in shape.roi_points]
    ys = [i[1] for i in shape.roi_points]
    roi_center_x = 0.5 * min(xs) + 0.5 * max(xs)
    roi_center_y = 0.5 * min(ys) + 0.5 * max(ys)
 
    translation_vector_x = int(center_x - roi_center_x)
    translation_vector_y = int(center_y - roi_center_y)
    
    translated_points = []
    for p in shape.roi_points:
        x, y = p
        x_translated = x + translation_vector_x
        y_translated = y + translation_vector_y
        translated_points.append((x_translated, y_translated))

    return translated_points


    
def choose_polygon(polygons_by_size, size):
    chosen_id = -1
    
    if size in polygons_by_size:
        chosen_id = random.choice(polygons_by_size[size])
    
    if chosen_id == -1:
        chosen_id = random.choice(polygons_by_size[15])

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






                # if distance < 0.5:  # Inner part (0 to 60% of the radius)
                #     t = distance / 0.5  # Scale t to [0, 1]
                #     color_index = int(t * (len(light_blue_rgb) - 1))
                #     t = (t * (len(light_blue_rgb) - 1)) - color_index
                #     color1 = light_blue_rgb[color_index % len(light_blue_rgb)]
                #     color2 = light_blue_rgb[(color_index + 1) % len(light_blue_rgb)]
                #     interpolated_color = interpolate_color(color1, color2, t)
                
                # elif distance < 0.7:  # middle part (60% to 70% of the radius)
                #     t = (distance - 0.5) / 0.2  # Scale t to [0, 1]
                #     inner_color_index = len(light_blue_rgb) - 1
                #     middle_color_index = int(t * (len(dark_blue_rgb) - 1))
                #     t_middle = t * (len(dark_blue_rgb) - 1) - middle_color_index

                #     inner_color = light_blue_rgb[inner_color_index]
                #     middle_color1 = dark_blue_rgb[middle_color_index % len(dark_blue_rgb)]
                #     middle_color2 = dark_blue_rgb[(middle_color_index + 1) % len(dark_blue_rgb)]

                #     middle_interpolated_color = interpolate_color(middle_color1, middle_color2, t_middle)
                #     interpolated_color = interpolate_color(inner_color, middle_interpolated_color, t)
                            
                # elif distance < 0.9:  # outer part (70% to 90% of the radius)
                #     t = (distance - 0.7) / 0.15 # Scale t to [0, 1]
                #     middle_color_index = len(dark_blue_rgb) - 1
                #     outer_color_index = int(t * (len(brown_rgb) - 1))
                #     t_outer = t * (len(brown_rgb) - 1) - outer_color_index

                #     middle_color = dark_blue_rgb[middle_color_index]
                #     outer_color1 = brown_rgb[outer_color_index % len(brown_rgb)]
                #     outer_color2 = brown_rgb[(outer_color_index + 1) % len(brown_rgb)]

                #     outer_interpolated_color = interpolate_color(outer_color1, outer_color2, t_outer)
                #     interpolated_color = interpolate_color(middle_color, outer_interpolated_color, t)
                
                # else : # make sure to blend with background
                #     t = (distance - 0.9) / 0.1
                #     color_index = int(t * (len(brown_rgb) - 1))
                #     color1 = brown_rgb[color_index % len(brown_rgb)]
                #     color2 = background_color
                #     interpolated_color = interpolate_color(color1, color2, t)

                # if y < image_height and x < image_width:
                #     img[y, x] = np.array(interpolated_color).astype(np.uint8)