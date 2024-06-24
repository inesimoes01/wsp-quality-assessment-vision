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
    # Load the image
    image = cv2.imread(image_path)
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # initialize masks for single and overlapped droplets
    single_mask = np.zeros_like(image)
    overlapped_mask = np.zeros_like(image)


    # binary threshold
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)   
    edges = cv2.Canny(threshold, 170, 200)
    #_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = threshold.shape



    
    # # cv2.drawContours(image, contours, -1, (255, 255, 255), 1)
    # polygons_single = []
    # polygons_overlapped = []

    # for contour in contours:

    #     contour = cv2.convexHull(contour)
      
    #     area = cv2.contourArea(contour)
    #     perimeter = cv2.arcLength(contour, True)

    #     if perimeter == 0:
    #         cv2.drawContours(overlapped_mask, [contour], -1, 255, thickness=cv2.FILLED)
    #         continue

    #     # calculate circularity and apply threshold
    #     circularity = 4 * np.pi * (area / (perimeter * perimeter))
    #     if circularity > 0.8: 
    #         cv2.drawContours(single_mask, [contour], -1, 255, thickness=cv2.FILLED)
    #         polygon = []
    #         for point in contour:
    #             x, y = point[0]
    #             polygon.append(x / W)
    #             polygon.append(y / H)
    #         polygons_single.append(polygon)

    #     else:
    #         cv2.drawContours(overlapped_mask, [contour], -1, 255, thickness=cv2.FILLED)
    #         polygon = []
    #         for point in contour:
    #             x, y = point[0]
    #             polygon.append(x / W)
    #             polygon.append(y / H)
    #         polygons_overlapped.append(polygon)

    # cv2.imwrite(os.path.join(output_folder_single, filename + ".png"), single_mask)
    # cv2.imwrite(os.path.join(output_folder_overlapped, filename + ".png"), overlapped_mask)

    # with open('{}.txt'.format(os.path.join(output_label, filename)), 'a') as f:
    #     for polygon in polygons_single:
    #         f.write(f"0 {' '.join(map(str, polygon))}\n")
    #     for polygon_over in polygons_overlapped:
    #         f.write(f"1 {' '.join(map(str, polygon))}\n")

    #brush.image2annotation(output_folder_single, label_name, from_name, to_name, ground_truth=False, model_version=None, score=None)


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

# def mask_to_label(output_dir, class_id, filename):
#     # load mask and get its contours
#     mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

#     H, W = mask.shape
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # convert the contours to polygons
#     polygons = []
#     for cnt in contours:
#         if cv2.contourArea(cnt) > 5:
#             polygon = []
#             for point in cnt:
#                 x, y = point[0]
#                 polygon.append(x / W)
#                 polygon.append(y / H)
#             polygons.append(polygon)
    
#     # print the polygons
#     with open('{}.txt'.format(os.path.join(output_dir, filename)), 'a') as f:
#         for polygon in polygons:
#             f.write(f"{class_id} {' '.join(map(str, polygon))}\n")


# input_folder = config.DATA_REAL_PROC_IMAGE_DIR
# output_folder_single = config.DATA_REAL_PROC_MASK_SIN_DIR
# output_folder_overlapped = config.DATA_REAL_PROC_MASK_OV_DIR

# output_label = config.DATA_REAL_PROC_LABEL_DIR



# if not os.path.exists(output_folder_single):
#     os.makedirs(output_folder_single)
# if not os.path.exists(output_folder_overlapped):
#     os.makedirs(output_folder_overlapped)

# delete_folder_contents(output_folder_overlapped)
# delete_folder_contents(output_folder_single)
# delete_folder_contents(output_label)

# for filename in os.listdir(input_folder):
#     if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
#         image_path = os.path.join(input_folder, filename)
#         parts = filename.split(".")
#         file_name_noend = parts[0]
#         segment_droplets(image_path, output_folder_single, output_folder_overlapped, file_name_noend, output_label)




# # for file in os.listdir(output_folder_single):
# #     parts = file.split(".")
# #     filename = parts[0]
    
# #     image_path = os.path.join(output_folder_single, file)
# #     mask_to_label(output_label, 0, filename)

# # for file in os.listdir(output_folder_overlapped):
# #     parts = file.split(".")
# #     filename = parts[0]

# #     image_path = os.path.join(output_folder_single, file)
# #     mask_to_label(output_label, 1, filename)

    


