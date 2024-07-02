import os

import sys
sys.path.insert(0, 'src/common')
import config

def polygon_to_yolo_label(points, image_width, image_height):
    label = []
    for p in points:
        x, y = p
        new_x = max(x, 0) / image_width
        new_y = max(y, 0) / image_height
        if (new_x > 1):
            new_x = 1
        if new_y > 1:
            new_y = 1
        label.append(new_x)
        label.append(new_y)
    return label

def write_label_file(filepath, labels):
    classid = 0

    with open(filepath, 'a') as f:
        for label in labels:
            f.write(f"{classid} {' '.join(map(str, label))}\n")