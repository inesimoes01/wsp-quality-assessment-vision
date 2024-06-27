import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to create a circular gradient
def create_circular_gradient(radius, colors):
    # Create gradient image
    gradient = np.zeros((radius, radius, 3), dtype=np.uint8)
    
    # Define the center of the circle
    center = (radius, radius)
    
    # Create a circular mask
    for y in range(2*radius):
        for x in range(2*radius):
            distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if distance <= radius:
                ratio = distance / radius
                color = [int(colors[0][i] * (1 - ratio) + colors[1][i] * ratio) for i in range(3)]
                gradient[y, x] = color
            else:
                gradient[y, x] = [255, 255, 255]  # Background color

    # Apply circular mask to gradient
    mask = np.zeros((3*radius, 3*radius), dtype=np.uint8)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
    gradient = cv2.bitwise_and(gradient, gradient, mask=mask)
    
    return gradient

# Set the radius and colors for the gradient
radius = 10
colors = [(0, 0, 128), (255, 255, 0)]  # From blue to yellow

# Create the gradient
gradient_image = create_circular_gradient(radius, colors)
plt.imshow(gradient_image)
plt.show()