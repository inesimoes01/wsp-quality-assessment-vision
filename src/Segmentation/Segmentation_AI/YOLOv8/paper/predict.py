from ultralytics import YOLO
import os 
import numpy as np
from matplotlib import pyplot as plt
import random
import cv2

train_model_path = "results\\yolo_rectangle\\200epc_rectangle2"
model = YOLO(os.path.join(train_model_path, "weights", "last.pt"))

def find_polygon_yolo(img):

    results = model.predict(img, conf = 0.1, save = True, save_txt = True)

    for result in results:
        for mask in result.masks.xy:
            points = np.int32([mask])
            cv2.polylines(img, points, True, (255, 255, 0), 2)

    