import cv2
import numpy as np
from matplotlib import pyplot as plt 

from cairo import CairoPolygonDrawer



def convert_to_real_coordinates(points, image_width, image_height):
    real_coordinates = []
    for point in points:
        percentage_x, percentage_y = point
        real_x = (percentage_x / 100) * image_width
        real_y = (percentage_y / 100) * image_height
        real_coordinates.append([int(real_x), int(real_y)])
    return real_coordinates

def get_roi_points(points, image_width, image_height, x, y):
    roi_points = []
    for point in points:
        old_x, old_y = point
        new_x = old_x - x
        new_y = old_y - y
        roi_points.append([int(new_x), int(new_y)])
    return roi_points

points=np.array([[46.875,20.8984375], 
                    [46.484375,21.2890625], 
                    [45.703125,21.2890625],                  
                    [ 45.5078125, 21.484375],
                    [ 45.5078125, 21.6796875 ],
                    [ 45.3125, 21.875 ],
                    [ 45.3125, 22.4609375 ], 
                    [ 45.5078125, 22.65625 ], 
                    [ 45.5078125, 22.8515625 ], 
                    [ 45.703125, 23.046875 ], 
                    [ 46.2890625, 23.046875 ], 
                    [ 46.484375, 23.2421875 ], 
                    [ 47.265625, 23.2421875 ], 
                    [ 47.4609375, 23.046875 ], 
                    [ 47.65625, 23.046875 ], 
                    [ 47.8515625, 22.8515625 ], 
                    [ 47.8515625, 22.4609375 ], 
                    [ 48.046875, 22.265625 ], 
                    [ 48.046875, 21.875 ], 
                    [ 47.8515625, 21.6796875 ], 
                    [ 47.8515625, 21.484375 ], 
                    [ 47.4609375, 21.09375 ], 
                    [ 47.265625, 21.09375 ], 
                    [ 47.0703125, 20.8984375 ] ],dtype=np.float32)

# get the value of the points in a small area as to be able to easily scale the polygon
scaled_points = np.array(convert_to_real_coordinates(points, 512, 512))
x, y, w, h = cv2.boundingRect(scaled_points)
roi_points = np.array(get_roi_points(scaled_points, 512, 512, x, y))
   

CairoPolygonDrawer.draw(points)

# Create a mask image to draw the polygon
height, width = 25, 25
mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillPoly(mask, [roi_points], 255)

# Create a blank image
image = np.zeros((height, width, 3), dtype=np.uint8)

# Define the colors
dark_blue = (139, 0, 0)  # RGB for dark blue
light_blue = (0, 0, 255) # RGB for light blue
brown = (42, 42, 165) # RGB for brown

# Apply the gradient inside the polygon
for y in range(height):
    alpha = y / height
    color = cv2.addWeighted(dark_blue, 1 - alpha, light_blue, alpha, 0).astype(np.uint8)
    mask_row_indices = np.where(mask[y, :] == 255)[0]
    image[y, mask_row_indices] = dark_blue

# Add the brown color to the last row of the polygon
last_row_y = np.max(np.where(mask == 255)[0])
image[last_row_y, mask[last_row_y] == 255] = brown


plt.imshow(image)
plt.show()
