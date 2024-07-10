import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained YOLOv8 model (segmentation version)
#model = YOLO('results\\yolo\\100epc_7iou_0001lr_0005wd_2drp\\weights\\best.pt')  # you can replace with 'yolov8m-seg.pt', 'yolov8l-seg.pt', etc.
model = YOLO("src\\Segmentation_AI\\yolo_models\\yolov8n-seg.pt")
# Load an image
image_path = 'data\\artificial_dataset\\yolo_data\\images\\test\\61_0.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predict_results = model.predict(source=image_path, 
                                conf=0.2, 
                                iou=0.5, 
                                imgsz=512, 
                                visualize=True, 
                                show=True, 
                                save=True, 
                                save_txt=True
                                )


# # Perform inference
# results = model.predict(image_rgb,
#                         conf  =0.2)

# # Extract segmentation masks
# masks = results[0].masks.data  # assuming single image batch

# # Convert masks to numpy arrays
# masks = masks.cpu().numpy()

# # Visualize results
# fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# ax.imshow(image_rgb)

# # Overlay masks
# for mask in masks:
#     mask = mask.squeeze()
#     ax.imshow(mask, alpha=0.5)

# plt.axis('off')
# plt.show()
