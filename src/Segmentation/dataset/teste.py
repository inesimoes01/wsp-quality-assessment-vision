import cv2
import numpy as np

def find_unclosed_contours(contours):
    unclosed_contours = []
    for contour in contours:
        points_set = set()
        for point in contour:
            point_tuple = tuple(point[0])
            if point_tuple in points_set:
                unclosed_contours.append(contour)
                break
            points_set.add(point_tuple)
    return unclosed_contours

def remove_duplicate_lines(contour):
    unique_points = []
    points_set = set()
    for point in contour:
        point_tuple = tuple(point[0])
        if point_tuple not in points_set:
            unique_points.append(point)
            points_set.add(point_tuple)
    return np.array(unique_points)

def find_open_endpoints(contour):
    point_count = {}
    for point in contour:
        point_tuple = tuple(point[0])
        if point_tuple in point_count:
            point_count[point_tuple] += 1
        else:
            point_count[point_tuple] = 1
    open_endpoints = [point for point, count in point_count.items() if count == 1]
    return open_endpoints

def close_contour(contour):
    open_endpoints = find_open_endpoints(contour)
    if len(open_endpoints) == 2:
        contour = np.append(contour, [[open_endpoints[0]]], axis=0)
        contour = np.append(contour, [[open_endpoints[1]]], axis=0)
        return contour
    return contour

def process_contours(contours):
    unclosed_contours = find_unclosed_contours(contours)
    closed_contours = []
    for contour in unclosed_contours:
        contour = remove_duplicate_lines(contour)
        contour = close_contour(contour)
        closed_contours.append(contour)
    return closed_contours

# Example usage:
# Load image, convert to grayscale, and find edges
image = cv2.imread('C:\\Users\\mines\\AppData\\Local\\label-studio\\label-studio\\media\\upload\\1\\3ed68956-1_V1_A1_square25.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Process contours
closed_contours = process_contours(contours)

# Draw closed contours on the image
for contour in closed_contours:
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)

# Save the result
cv2.imwrite('closed_contours.jpg', image)
