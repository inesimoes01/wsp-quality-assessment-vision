import numpy as np
import cv2

def normalize_polygon_points(points, original_width, original_height):
    normalized_points = []
    for point in points:
        percentage_x, percentage_y = point
        normalized_x = percentage_x / 100
        normalized_y = percentage_y / 100
        normalized_points.append((normalized_x, normalized_y))
    return normalized_points

def scale_polygon_points(normalized_points, diameter):
    scaled_points = []
    for point in normalized_points:
        normalized_x, normalized_y = point
        scaled_x = normalized_x * diameter
        scaled_y = normalized_y * diameter
        scaled_points.append((scaled_x, scaled_y))
    return scaled_points

def translate_polygon_points(scaled_points, center_x, center_y):
    translated_points = []
    for point in scaled_points:
        scaled_x, scaled_y = point
        translated_x = center_x + scaled_x - (scaled_x / 2)
        translated_y = center_y + scaled_y - (scaled_y / 2)
        translated_points.append([int(translated_x), int(translated_y)])
    return translated_points

# Example usage:
points = [
    [46.875, 20.8984375],
    [46.484375, 21.2890625],
    [45.703125, 21.2890625],
    [45.5078125, 21.484375],
    [45.5078125, 21.6796875],
    [45.3125, 21.875],
    [45.3125, 22.4609375],
    [45.5078125, 22.65625],
    [45.5078125, 22.8515625],
    [45.703125, 23.046875],
    [46.2890625, 23.046875],
    [46.484375, 23.2421875],
    [47.265625, 23.2421875],
    [47.4609375, 23.046875],
    [47.65625, 23.046875],
    [47.8515625, 22.8515625],
    [47.8515625, 22.4609375],
    [48.046875, 22.265625],
    [48.046875, 21.875],
    [47.8515625, 21.6796875],
    [47.8515625, 21.484375],
    [47.4609375, 21.09375],
    [47.265625, 21.09375],
    [47.0703125, 20.8984375]
]

original_width = 1920  # original width of the image in pixels
original_height = 1080  # original height of the image in pixels

# Normalize the polygon points
#normalized_points = normalize_polygon_points(points, original_width, original_height)

# Desired size and position
diameter = 100  # new diameter for the polygon
center_x = 500  # x-coordinate of the new center position
center_y = 500  # y-coordinate of the new center position

# Scale and translate the polygon points
scaled_points = scale_polygon_points(points, diameter)
translated_points = np.array(translate_polygon_points(scaled_points, center_x, center_y))

mask_change = np.zeros((512*2, 512*2), dtype=np.uint8)
poly = cv2.fillPoly(mask_change, [translated_points], 255)


import matplotlib.pyplot as plt
plt.imshow(mask_change)
plt.show()
