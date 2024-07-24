import cv2
import os
import numpy as np
import sys

from label_studio_converter.brush import encode_rle, image2annotation
import json

from matplotlib import pyplot as plt 

sys.path.insert(0, 'src/segmentation')
sys.path.insert(0, 'src/common')
import config as config
from Util import * 

def segment_droplets(image_path, output_folder_single, output_folder_overlapped, filename, output_label):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binary threshold
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)   
    edges = cv2.Canny(threshold, 170, 200)
   
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = threshold.shape


annotation = image2annotation(
    'C:/Users/mines/OneDrive - Universidade do Porto/documentos/00 Uni/5º ANO/Tese/00 GitHub/vision-quality-assessment-opencv/data/real_dataset/processed/image/1_V1_A1_square25.jpg',
    label_name='droplet',
    from_name='label',
    to_name='image',
    model_version='v1',
    type = 'polygonlabels',
    score=0.5,
)

task = {
    'data': {'image': '/data/upload/1/3ed68956-1_V1_A1_square25.jpg'},
    'predictions': [annotation],
}


json.dump(task, open('task.json', 'w'))
