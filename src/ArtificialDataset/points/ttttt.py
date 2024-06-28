import cv2
import numpy as np
from shapely import geometry
from matplotlib import pyplot as plt 

# Function to create concentric polygons
def create_concentric_polygons(coords, scale_factors,min_corner):
    original_polygon = geometry.Polygon(coords)
    polygons = []
    for scale in scale_factors:
        distance = center.distance(min_corner) * scale
        new_polygon = original_polygon.buffer(-distance)
        polygons.append(new_polygon)
    return polygons

# Function to blend colors
def blend_colors(img, polygon, color):
    # overlay = img.copy()
    pts = np.array(list(polygon.exterior.coords), np.int32)
    pts = pts.reshape((-1, 1, 2))

    cv2.fillPoly(img, [pts], color)
    # alpha = 0.4
    # cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# Coordinates of the polygon (example)
coords = [(100, 150), (200, 50), (300, 150), (250, 250), (150, 250)]

# Compute bounding box and center
target_diameter = 50
_, _, w, h = cv2.boundingRect(np.array(coords))
#scale_factor = (1 - target_diameter / max(w, h)) / 2
xs = [i[0] for i in coords]
ys = [i[1] for i in coords]
x_center = 0.5 * min(xs) + 0.5 * max(xs)
y_center = 0.5 * min(ys) + 0.5 * max(ys)

min_corner = geometry.Point(min(xs), min(ys))
max_corner = geometry.Point(max(xs), max(ys))
center = geometry.Point(x_center, y_center)

# Define scale factors for layers
scale_factors = [0.1, 0.3, 0.5]

# Create concentric polygons
polygons = create_concentric_polygons(coords, scale_factors, min_corner)

# Define colors for each layer (BGR format)
colors = [(77, 47, 6), (12, 6, 77), (54, 130, 158)]  # Light blue, dark blue, brown

# Create an empty image
img = np.zeros((400, 400, 3), dtype=np.uint8)

# Blend each polygon layer with corresponding color
for polygon, color in zip(polygons, colors):
    blend_colors(img, polygon, color)

plt.imshow(img)
plt.show()