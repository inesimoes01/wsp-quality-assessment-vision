import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_yolo_annotation_file(file_path, image_height, image_width):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    polygon_coordinates = []

    for line in lines:
        data = line.strip().split()
        if len(data) < 2:
            continue
        
        coordinates = []
        for i in range(1, len(data), 2):
            x = (float(data[i])) * image_width
            y = (float(data[i + 1])) * image_height
            coordinates.append((int(x), int(y)))
        polygon_coordinates.append(coordinates)

    # for line in lines:

    #     values = line.strip().split()
    #     if len(values) < 2:
    #         continue  

    #     # extract coordinates (ignoring the class number, assuming it's always at the start)
    #     coordinates = [(int(float(values[i])*image_width), int(float(values[i+1])* image_height)) for i in range(1, len(values)-1, 2)]
    #     polygon_coordinates.append(coordinates)

    return np.array(polygon_coordinates, dtype=np.int32)


#for file in os.listdir(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR):
    # load images and mask
    # background_image = cv2.imread('path_to_original_image.jpg')
    # mask = cv2.imread('path_to_mask_image.png', cv2.IMREAD_GRAYSCALE)
    # wsp_image = cv2.imread(os.path.join(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR, file))

background_image = cv2.imread('data\\artificial_dataset_versions\\s4_06_jpg.rf.408d920a0920b27f57e08e2c620015a0.jpg')
image_height, image_width = background_image.shape[:2]
#mask = cv2.imread('path_to_mask_image.png', cv2.IMREAD_GRAYSCALE)
wsp_image = cv2.imread('data\\artificial_dataset_versions\\0.png')
wsp_image = cv2.rotate(wsp_image, cv2.ROTATE_90_CLOCKWISE)

polygon_points = read_yolo_annotation_file('data\\artificial_dataset_versions\\s4_06_jpg.rf.408d920a0920b27f57e08e2c620015a0.txt', image_height, image_width)

# find the contours in the mask
polygon_mask = np.zeros_like(background_image)
print(polygon_points)
cv2.fillPoly(polygon_mask, polygon_points, (255, 255, 255))
x, y, w, h = cv2.boundingRect(polygon_points)

plt.imshow(polygon_mask)
plt.show()

# Resize new image to fit the bounding box
resized_new_image = cv2.resize(wsp_image, (w, h))

# Place the resized new image into the polygon area
warped_new_image = np.zeros_like(background_image)
warped_new_image[y:y+h, x:x+w] = resized_new_image

# Combine the warped new image with the original image using the mask
result_image = cv2.bitwise_and(background_image, 255 - polygon_mask) + cv2.bitwise_and(warped_new_image, polygon_mask)

# Save or display the result
plt.imshow(result_image)
plt.show()
