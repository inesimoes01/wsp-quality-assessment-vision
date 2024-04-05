import cv2
from matplotlib import pyplot as plt 

image = cv2.imread("images\\inesc_dataset\\1_V1_A1.jpg")
edges = cv2.GaussianBlur(image, (5, 5), 1.5, 1.5)
edges = cv2.erode(edges, None)
edges = cv2.dilate(edges, None)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)
edge_image= cv2.Canny(edges, 100, 150)

# try to remove most of the edges of the droplet
kernel =cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel)

# RETR_EXTERNAL - retrieves all of the contours without establishing any hierarchical relationships.
# CHAIN_APPROX_SIMPLE - compresses horizontal, vertical, and diagonal segments and leaves only their end points.
contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

# assuming the biggest contour will be the rectangle
convexHull = cv2.convexHull(contours[0])
cv2.drawContours(image, [convexHull], -1, (255, 255, 0), 2)

plt.imshow(image)
plt.show()


