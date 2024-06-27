import numpy as np
import matplotlib.pyplot as plt
import cv2

def draw_varied_color_circle(img, center, radius, color):
    for y in range(center[1] - radius, center[1] + radius):
        for x in range(center[0] - radius, center[0] + radius):
            if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
                # Calculate distance from center
                distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                # Calculate color intensity based on distance
                intensity = 1 - distance / radius
                # Apply intensity to color channels
                varied_color = tuple(int(color[c] * intensity) for c in range(3))
                # Set pixel color in the image
                img[y, x] = varied_color

# Create a blank image
img = np.zeros((512, 512, 3), dtype=np.uint8)
img = cv2.rectangle(img, (0, 0), (512, 512), (255, 255, 0), cv2.FILLED)

# Define circle parameters
center = (256, 256)
radius = 10


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    return rgb

color = hex_to_rgb('#181872')
#color = (0, 255, 0)  # BGR color tuple (here, green)

# Draw the varied color circle
draw_varied_color_circle(img, center, radius, color)

plt.imshow(img)
plt.show()