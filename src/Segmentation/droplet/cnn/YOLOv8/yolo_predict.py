from ultralytics import YOLO
import os 
import numpy as np
from matplotlib import pyplot as plt
import random
import cv2

train_model_path = "results\\yolo_droplet\\50epc_droplet2\\weights\\best.pt"
model = YOLO(train_model_path)

import cv2
import numpy as np
from ultralytics import YOLO

def plot_results(image, mask, segmentation_result):
#     # Create an overlay to display the results
    overlay = image.copy()
    
    # Apply mask to the overlay
    overlay[mask == 255] = (0, 255, 0)  # Green color for the mask

    # Add the detected polygons to the overlay
    for polygon in segmentation_result:
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 255), thickness=1)  # Red color for the edges

    # Combine original image with overlay
    result = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    # Display the result
    plt.imshow(result)
    plt.show()

def create_binary_mask(segmentation_result, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for polygon in segmentation_result:
        if len(polygon) > 0:
            cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask

def draw_contours(image, masks):
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return image

def main(image_path):

    # Load the image
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Perform inference
    results = model(image_path, conf=0.2)

    segmentation_result = results[0].masks.xy
    mask = create_binary_mask(segmentation_result, image.shape)

    # Draw contours on the original image
    result_image = plot_results(original_image, mask, segmentation_result)

    plt.imshow(result_image)
    plt.show()

if __name__ == "__main__":
    
    image_path = ("data\\synthetic_normal_dataset\\yolo\\images\\test\\131_6.png")        # Update with your image path
    main(image_path)



# def create_binary_mask(segmentation_result, image_shape):
#     mask = np.zeros(image_shape[:2], dtype=np.uint8)
#     for polygon in segmentation_result:
#         cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
#     return mask

# def plot_results(image, mask, segmentation_result):
#     # Create an overlay to display the results
#     overlay = image.copy()
    
#     # Apply mask to the overlay
#     overlay[mask == 255] = (0, 255, 0)  # Green color for the mask

#     # Add the detected polygons to the overlay
#     for polygon in segmentation_result:
#         pts = np.array(polygon, np.int32)
#         pts = pts.reshape((-1, 1, 2))
#         cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 255), thickness=2)  # Red color for the edges

#     # Combine original image with overlay
#     result = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

#     # Display the result
#     plt.imshow(result)
#     plt.show()

# def find_droplets_yolo(image_path):

#     image = cv2.imread(image_path)

#     results = model(image, conf = 0.5)
#     segmentation_result = results[0].masks.xy

#     mask = create_binary_mask(segmentation_result, image.shape)

#     plot_results(image, mask, segmentation_result)



# find_droplets_yolo("data\\synthetic_normal_dataset\\yolo\\images\\test\\0_1.png")