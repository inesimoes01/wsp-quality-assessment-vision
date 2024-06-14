import os
import cv2
import sys

sys.path.insert(0, 'src/segmentation')
sys.path.insert(0, 'src/common')

from Util import * 


def mask_to_label(input_dir , output_dir, class_id):
    for j in os.listdir(input_dir):
        image_path = os.path.join(input_dir, j)
        # load the binary mask and get its contours
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        H, W = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # convert the contours to polygons
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 5:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)
        
        # print the polygons
        with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'a') as f:
            for polygon in polygons:
                f.write(f"{class_id} {' '.join(map(str, polygon))}\n")
       
        # with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
        #     for polygon in polygons:
        #         for p_, p in enumerate(polygon):
        #             if p_ == len(polygon) - 1:
        #                 f.write('{}\n'.format(p))
        #             elif p_ == 0:
        #                 f.write('0 {} '.format(p))
        #             else:
        #                 f.write('{} '.format(p))

            




input_dir = 'images\\dataset_rectangle\\masks\\single'
output_dir = 'images\\dataset_rectangle\\yolo'

delete_folder_contents(output_dir)

mask_to_label(input_dir , output_dir, 0)

input_dir = 'images\\dataset_rectangle\\masks\\overlapped'
output_dir = 'images\\dataset_rectangle\\yolo'

mask_to_label(input_dir , output_dir, 1)