import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

img = cv2.imread("C:\\Users\\mines\\OneDrive\\Ambiente de Trabalho\\sem.png", cv2.IMREAD_GRAYSCALE)

_, threshold = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(threshold, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

final_image = np.ones_like(img)
cv2.drawContours(final_image, contours, -1, 0, 5)
cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
(center_x, center_y), radius = cv2.minEnclosingCircle(cntsSorted[0])


# Parameters for circle drawing
max_radius = radius -10
num_circles = 1000

# Function to draw circles on edges
def draw_circles_on_edges(image, edges, num_circles, max_radius):
    height, width = image.shape[:2]
    circles = []
    
    # Get the coordinates of the edge points
    edge_points = np.column_stack(np.where(edges > 0))
    
    for _ in range(num_circles):
        if len(edge_points) == 0:
            break
            
        # Randomly select an edge point
        idx = np.random.randint(0, len(edge_points))
        y, x = edge_points[idx]
        
        # Randomly select a radius
        r = max_radius
        
        circles.append((x, y, r))
    
    return circles

# Draw the circles

 
circles = draw_circles_on_edges(final_image, edges, num_circles, max_radius)
plt.figure(figsize=(10, 10))
plt.imshow(final_image, cmap='gray')

for (x, y, r) in circles:
    circle = plt.Circle((x, y), r, color='red', fill=False, alpha=0.1)
    plt.gca().add_patch(circle)

plt.axis('off')

plt.savefig("results\\latex\\hough_circles.png", dpi = 1200)
plt.show()