from ultralytics import YOLO
import os 
import numpy as np
from matplotlib import pyplot as plt
import random
import cv2

train_model_path = "results\\yolo_rectangle\\200epc_rectangle2"
model = YOLO(os.path.join(train_model_path, "weights", "last.pt"))

def create_binary_mask(segmentation_result, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for polygon in segmentation_result:
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask

def plot_results(image, mask, segmentation_result):
    # Create an overlay to display the results
    overlay = image.copy()
    
    # Apply mask to the overlay
    overlay[mask == 255] = (0, 255, 0)  # Green color for the mask

    # Add the detected polygons to the overlay
    for polygon in segmentation_result:
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 255), thickness=2)  # Red color for the edges

    # Combine original image with overlay
    result = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    # Display the result
    plt.imshow(result)
    plt.show()

def find_polygon_yolo(image_path):

    image = cv2.imread(image_path)

    results = model(image)
    segmentation_result = results[0].masks.xy

    mask = create_binary_mask(segmentation_result, image.shape)

    plot_results(image, mask, segmentation_result)



find_polygon_yolo("data\\real_rectangle_dataset\\test\\image\\2_V1_A3_jpg.rf.3ff6c061dd2d3f7239d33e83352914b1.jpg")