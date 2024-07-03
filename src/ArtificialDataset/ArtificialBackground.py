import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image

sys.path.insert(0, 'src/common')
import config

def resize_image(image, target_width, target_height):
    current_height, current_width = image.shape[:2]
    if current_width != target_width or current_height != target_height:
        resized_image = cv2.resize(image, (int(target_width), int(target_height)))
    else:
        resized_image = image
    return resized_image

def read_yolo_label_wsp(lines, image_height_original, image_width_original, w, h, dx, dy):
    polygon_coordinates = []

    for line in lines:
        data = line.strip().split()
        if len(data) < 2:
            continue
        
        coordinates = []
        for i in range(1, len(data), 2):
            # switched because the labels are like the original image
            x = (float(data[i])) * image_height_original
            y = (float(data[i + 1])) * image_width_original

            # rotation
            rot_x = int(image_width_original - y)
            rot_y = int(x)

            # # trans_x = w / image_width_original
            # # trans_y = h / image_height_original
            
            # # translation
            trans_x = rot_x + dx
            trans_y = rot_y + dy
            
            coordinates.append((trans_x, trans_y))
            
            #new_coords = np.array([[img_height - y, x] for x, y in coords])
        polygon_coordinates.append(np.array(coordinates, dtype=np.int32))

    return polygon_coordinates


def read_yolo_label_background(lines, image_height, image_width):
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

    return np.array(polygon_coordinates, dtype=np.int32)
 


def translate_coordinates(coords, dx, dy):
    return coords + np.array([dx, dy])

def scale_coordinates(coords, scale_x, scale_y):
    return coords * np.array([scale_x, scale_y])

def compress_to_resolution(image_path, resolution):
    img = Image.open(image_path)

    original_width, original_height = img.size
    aspect_ratio = original_width / original_height

    # determine the new dimensions based on the target resolution
    new_width = resolution[0]
    new_height = int(new_width / aspect_ratio)
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img.save(image_path)


last_index = 0
wsp_image_width = 468
wsp_image_height = 1368
for background_file in os.listdir(config.DATA_ARTIFICIAL_BG_IMAGE_DIR):
    #compress_to_resolution(os.path.join(config.DATA_ARTIFICIAL_BG_IMAGE_DIR, background_file), )
    bg_filename = os.path.splitext(background_file)[0]
        
    background_image = cv2.imread(os.path.join(config.DATA_ARTIFICIAL_BG_IMAGE_DIR, background_file))
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    image_height, image_width = background_image.shape[:2]

    with open(os.path.join(config.DATA_ARTIFICIAL_BG_LABEL_DIR, bg_filename + ".txt"), 'r') as file:
        lines = file.readlines()

    rectangle_points = read_yolo_label_background(lines, image_height, image_width)
    #rectangle_mask = np.zeros_like(background_image)
    rectangle_points = np.array(rectangle_points, dtype=np.int32)
    #cv2.fillPoly(rectangle_mask, rectangle_points, (255, 255, 255))
    x, y, w, h = cv2.boundingRect(rectangle_points)

    resized_background = resize_image(background_image, image_width * wsp_image_width / w, image_height * wsp_image_height / h)
    image_height_rez, image_width_rez = resized_background.shape[:2]

    rectangle_points = read_yolo_label_background(lines, image_height_rez, image_width_rez)
    rectangle_mask = np.zeros_like(resized_background)
    rectangle_points = np.array(rectangle_points, dtype=np.int32)
    cv2.fillPoly(rectangle_mask, rectangle_points, (255, 255, 255))
    x, y, w, h = cv2.boundingRect(rectangle_points)


    # plt.imshow(rectangle_mask)
    # plt.show()

    
    for index in range(10):
        index += last_index + 1000
        
        # load wsp_image created previously
        wsp_image = cv2.imread(os.path.join(config.DATA_ARTIFICIAL_WSP_IMAGE_DIR, str(index) + ".png"))
        wsp_image = cv2.cvtColor(wsp_image, cv2.COLOR_BGR2RGB)
        wsp_image = cv2.rotate(wsp_image, cv2.ROTATE_90_CLOCKWISE)
        wsp_image_height, wsp_image_width = wsp_image.shape[:2]

        
        # resize the wsp image to fit the rectangle and place it on the background image
        resized_new_image = cv2.resize(wsp_image, (w, h))
        warped_new_image = np.zeros_like(resized_background)
        warped_new_image[y:y+h, x:x+w] = resized_new_image
        wsp_image_height_rez, wsp_image_width_rez = wsp_image.shape[:2]
        
        result_image = cv2.bitwise_and(resized_background, 255 - rectangle_mask) + cv2.bitwise_and(warped_new_image, rectangle_mask)

        # get new coordinates of the droplets
        wsp_label_file = os.path.join(config.DATA_ARTIFICIAL_WSP_LABEL_DIR, str(index) + ".txt")
        with open(wsp_label_file, 'r') as file:
            wsp_lines = file.readlines()

        wsp_annotations = read_yolo_label_wsp(wsp_lines, wsp_image_height, wsp_image_width, w, h, x, y)
        
        new_label_file = os.path.join(config.DATA_ARTIFICIAL_RAW_LABEL_DIR, f"{index}.txt")
        with open(new_label_file, 'w') as file:
            for coords in wsp_annotations:
                coord_str = " ".join([f"{p[0] / image_width_rez} {p[1] / image_height_rez}" for p in coords])
                file.write(f"{0} {coord_str}\n")


        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR, str(index) + ".png"), result_image)


    
    last_index += 10