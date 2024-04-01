import numpy as np
import cv2
from matplotlib import pyplot as plt 
import copy

class Circle:
    def __init__(self, xCenter, yCenter, radius):
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.radius = radius

def circle_ransac(edge_points, iterations, radius_threshold, edge_points_threshold):


    best_circles = []
    for _ in range(iterations):
        # Select three points at random among edge points
        indices = np.random.choice(len(edge_points), 3, replace=False)
        A, B, C = [np.array(edge_points[i]) for i in indices]

        # Calculate midpoints
        midpt_AB = (A + B) * 0.5
        midpt_BC = (B + C) * 0.5

        # Calculate slopes and intercepts
        slope_AB = (B[1] - A[1]) / (B[0] - A[0] + 0.000000001)
        intercept_AB = A[1] - slope_AB * A[0]
        slope_BC = (C[1] - B[1]) / (C[0] - B[0] + 0.000000001)
        intercept_BC = C[1] - slope_BC * C[0]

        # Calculate perpendicular slopes and intercepts
        slope_midptAB = -1.0 / slope_AB
        slope_midptBC = -1.0 / slope_BC
        intercept_midptAB = midpt_AB[1] - slope_midptAB * midpt_AB[0]
        intercept_midptBC = midpt_BC[1] - slope_midptBC * midpt_BC[0]

        # Calculate intersection of perpendiculars to find center of circle and radius
        centerX = (intercept_midptBC - intercept_midptAB) / (slope_midptAB - slope_midptBC)
        centerY = slope_midptAB * centerX + intercept_midptAB
        center = (centerX, centerY)
        radius = np.linalg.norm(center - A)
        circumference = 2.0 * np.pi * radius

        on_circle = []
        not_on_circle = []
        

        # Find edge points that fit on circle radius
        for i, point in enumerate(edge_points):
            distance_to_center = np.linalg.norm(np.array(point) - np.array(center))
            if abs(distance_to_center - radius) < radius_threshold:
                on_circle.append(i)
            else:
                not_on_circle.append(i)
      
        # If number of edge points more than circumference, we found a correct circle
        if len(on_circle) >= circumference:
            print("yes")
            circle_found = Circle(centerX, centerY, radius)
            best_circles.append(circle_found)

            # Remove edge points if circle found (only keep non-voting edge points)
            edge_points = [edge_points[i] for i in not_on_circle]

        # Stop iterations when there are not enough edge points
        if len(edge_points) < edge_points_threshold:
            break

    return best_circles

iterations = 1000
radius_threshold = 16
edge_points_threshold = 5

# Load image
image_path = 'images\\artificial_dataset\\outputs\\overlapped\\2024-03-25_0\\51.png'
image = cv2.imread(image_path)

# process image
image1 = copy.copy(image)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
edges1 = cv2.Canny(gray1, 50, 150)
_, thresh1 = cv2.threshold(edges1, 127, 255, cv2.THRESH_BINARY)
contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
edge_points1 = [point[0] for contour in contours1 for point in contour]

# RANSAC circle detection
best_circles1 = circle_ransac(edge_points1, iterations, radius_threshold, edge_points_threshold)

# circle image
for circle in best_circles1:
    cv2.circle(image1, (int(circle.xCenter), int(circle.yCenter)), int(circle.radius), (0, 255, 0), 1)

# remove circles detected
mask1 = np.zeros_like(gray1)
for circle in best_circles1:
    for i in range(int(circle.radius)):
        cv2.circle(mask1, (int(circle.xCenter), int(circle.yCenter)), int(circle.radius)-i, (255), 2)
mask_inv1 = cv2.bitwise_not(mask1)
result1 = cv2.bitwise_and(thresh1, thresh1, mask=mask_inv1)


# RUN IT AGAIN
image2 = copy.copy(result1)
edges2 = cv2.Canny(image2, 50, 150)
_, thresh2 = cv2.threshold(edges2, 127, 255, cv2.THRESH_BINARY)
contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
edge_points2 = [point[0] for contour in contours2 for point in contour]

# RANSAC circle detection on the result image
best_circles2 = circle_ransac(edge_points2, iterations, radius_threshold, edge_points_threshold)

# circle image
image3 = copy.copy(image1)
for circle in best_circles2:
    cv2.circle(image3, (int(circle.xCenter), int(circle.yCenter)), int(circle.radius), (255, 0, 0), 1)

# mask2 = np.zeros_like(result1)
# for circle in best_circles2:
#     cv2.circle(mask2, (int(circle.xCenter), int(circle.yCenter)), int(circle.radius), (255), -1)
# mask_inv2 = cv2.bitwise_not(mask2)
# result2 = cv2.bitwise_and(thresh2, thresh2, mask=mask_inv2)


fig = plt.figure(figsize=(10, 7)) 
fig.add_subplot(2, 2, 1)
plt.imshow(image1)
fig.add_subplot(2, 2, 2)
plt.imshow(result1)
# fig.add_subplot(2, 2, 3)
# plt.imshow(result2)
fig.add_subplot(2, 2, 4)
plt.imshow(image3)
plt.show()

# cv2.imshow('Result', result)
# cv2.imshow('Result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Display the result
# cv2.imwrite('images\\artificial_dataset\\outputs\\overlapped\\2024-03-25_0\\51_ransac.png', image)
# cv2.imshow('Detected Circles', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
