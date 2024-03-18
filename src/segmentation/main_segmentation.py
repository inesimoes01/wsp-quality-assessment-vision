import cv2
from matplotlib import pyplot as plt 
import os
import copy

# read image
in_image = cv2.imread('images\\artificial_images\\image\\wsp_2024-03-18_0.png')
in_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
out_image = copy.copy(in_image)
stats_file_path = ('images\\artificial_images\\statistic\\statistics_2024-03-18_0.txt')

# grayscale
gray = cv2.cvtColor(out_image, cv2.COLOR_RGB2GRAY)
gray_image = copy.copy(gray)

# canny edge detectio + thresholding + contours
edges = cv2.Canny(gray, 50, 150)
ret, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw contours
cv2.drawContours(out_image, contours, -1, (0, 255, 0), 2)
contour_image = copy.copy(out_image)

# number of contours
object_count = len(contours)
print("Number of objects detected:", object_count)

# ground truth of number of contours
with open(stats_file_path, 'r') as f:
    for line in f:
        if "Number of droplets: " in line:
            print(line)
            number_of_droplets = int(line.split(":")[1].strip())


roi_image = copy.copy(in_image)
enumerate_image = copy.copy(in_image)
diameter_image = copy.copy(in_image)

# measure diameters + crop ROI
for i, contour in enumerate(contours):    
    # find minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(contour)
    diameter = radius * 2

    # annotate image
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(diameter_image, center, radius, (255, 0, 0), 2)
    cv2.putText(diameter_image, f'{diameter:.2f}', (int(x-radius), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # crop region of interest
    x, y, w, h = cv2.boundingRect(contour)
    expansion_factor = 2
    expanded_w = int(w * expansion_factor)
    expanded_h = int(h * expansion_factor)
    x -= int((expanded_w - w) / 2)
    y -= int((expanded_h - h) / 2)
    x = max(x, 0)
    y = max(y, 0)
    object_roi = roi_image[y:y+expanded_h, x:x+expanded_w]
    (x, y), radius = cv2.minEnclosingCircle(contour)
    cv2.putText(enumerate_image, f'{i}', (int(x-radius), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # delete old outputs
    for filename in os.listdir('images\\outputs'):
        file_path_statistic = os.path.join('images\\outputs', filename)
        if os.path.isfile('images\\outputs'):
            os.remove(file_path_statistic)

    # save outputs
    object_roi = cv2.cvtColor(object_roi, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'images\\outputs\\output_{i}.jpg', object_roi)
   

# display the result
fig = plt.figure(figsize = (10, 7))
fig.add_subplot(2, 1, 1)
plt.imshow(in_image)
plt.title("IN Number of droplets: {}".format(number_of_droplets))
fig.add_subplot(2, 1, 2)
plt.imshow(contour_image)
plt.title("OUT Number of droplets: {}".format(object_count))

plt.show()
