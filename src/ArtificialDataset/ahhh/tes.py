import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
def draw_varied_color_circle(img, center, radius, colors):
    color_count = len(colors)
    color_index = 0
    for y in range(center[1] - radius, center[1] + radius):
        for x in range(center[0] - radius, center[0] + radius):
            if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
                # Calculate distance from center
                distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                # Calculate color index based on distance
                color_index = int(distance / radius * (color_count - 1))
                # Get color from the list
                color = colors[color_index]
                # Set pixel color in the image
                img[y, x] = color
# Create a blank image
img = np.zeros((512, 512, 3), dtype=np.uint8)

# Define circle parameters
center = (256, 256)
radius = 5
colors = [
    (255, 0, 0),   # Red
    (255, 255, 0), # Yellow
    (0, 255, 0),   # Green
    (0, 255, 255), # Cyan
    (0, 0, 255),   # Blue
    (255, 0, 255)  # Magenta
]

# Draw the varied color circle
draw_varied_color_circle(img, center, radius, colors)
plt.imshow(img)
plt.show()
