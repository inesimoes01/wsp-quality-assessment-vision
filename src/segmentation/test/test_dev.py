import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_droplets(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Image not found or could not be read.")
        return

    # Apply a binary threshold to the image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour (for example)
    contour = max(contours, key=cv2.contourArea)
    
    # Compute Derivatives
    def compute_derivative(points, i):
        prev_point = points[i - 1] if i > 0 else points[-1]
        next_point = points[i + 1] if i < len(points) - 1 else points[0]
        dx = next_point[0][0] - prev_point[0][0]
        dy = next_point[0][1] - prev_point[0][1]
        return np.array([dx, dy])

    derivatives = [compute_derivative(contour, i) for i in range(len(contour))]

    # Check for intersections
    def check_intersections(derivatives):
        intersections = []
        for i in range(len(derivatives) - 1):
            current_dir = np.arctan2(derivatives[i][1], derivatives[i][0])
            next_dir = np.arctan2(derivatives[i + 1][1], derivatives[i + 1][0])
            if current_dir * next_dir < 0:  # Check for sign change indicating intersection
                intersections.append((i, i + 1))
        return intersections

    intersections = check_intersections(derivatives)

    # Plot the results
    plt.figure(figsize=(10, 8))
    plt.imshow(binary_image, cmap='gray')
    
    # Plot the contour
    contour_points = contour
    plt.imshow(contour_points)
    plt.show()
    plt.plot(contour_points[:, 0], contour_points[:, 1], 'b-')
    
    # # Highlight intersection points
    # for idx in intersections:
    #     plt.plot(contour[idx[0]][0][0], contour[idx[0]][0][1], 'ro')
    #     plt.plot(contour[idx[1]][0][0], contour[idx[1]][0][1], 'ro')

    plt.title('Contour with Intersection Points')
    plt.axis('off')
    plt.show()

# Path to the image
image_path = 'images\\inesc_dataset\\0.png'

# Detect droplets and plot the results
detect_droplets(image_path)
