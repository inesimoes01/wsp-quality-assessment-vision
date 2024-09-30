import os
import sys
import colorsys
import random
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as polygon_plot
from shapely import Polygon

sys.path.insert(0, 'src')
import Common.config as config 
import evaluate_algorithms_config
from Common.Statistics import Statistics as stats

random.seed(42)

def _random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def create_segmentation_image(filename, path_results, path_dataset, algorithm):
    label_path = os.path.join(path_results, config.RESULTS_GENERAL_LABEL_FOLDER_NAME, filename + ".txt")
    image_path = os.path.join(path_dataset, config.DATA_GENERAL_IMAGE_FOLDER_NAME, filename + ".png")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[:2]
    
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Display the image


    with open(label_path, 'r') as file:
        lines = file.readlines()
        colors = _random_colors(len(lines))
    
    polygons = []
    with open(label_path, 'r') as file:
        for i, line in enumerate(file):
            parts = line.strip().split()            
            coordinates = list(map(float, parts[1:]))
            polygon = [(coordinates[i] * width, coordinates[i+1] * height) for i in range(0, len(coordinates), 2)]
            polygons.append(polygon)

            color = colors[i]

            if Polygon(polygon).area < 10000:

                poly_patch = polygon_plot(polygon, edgecolor=color, facecolor=color, alpha=0.5)
                ax.add_patch(poly_patch)


    plt.axis('off')
    save_path = os.path.join("results\\latex\\segmentation results", filename + "_" + algorithm + ".png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=600)

# yolo_results = "results\\droplet_detection\\yolo_algorithm\\real_dataset\\full"
# mrcnn_results = "results\\droplet_detection\\mrcnn_algorithm\\real_dataset\\full"
# ccv_results = "results\\droplet_detection\\computer_vision_algorithm\\real_dataset\\full"
# cellpose_results = "results\\droplet_detection\\cellpose_algorithm\\real_dataset\\full"

# create_segmentation_image("2_Z1_cut", 
#                           yolo_results,
#                           "data\\droplets\\real_dataset_droplets\\full",
#                           "yolo")

# create_segmentation_image("2_Z1_cut", 
#                           ccv_results,
#                           "data\\droplets\\real_dataset_droplets\\full",
#                           "ccv")

# create_segmentation_image("2_Z1_cut", 
#                           cellpose_results,
#                           "data\\droplets\\real_dataset_droplets\\full",
#                           "cellpose")

path_dataset = "data\\droplets\\synthetic_dataset_droplets\\full\divided\\test"
yolo_results = "results\\droplet_detection\\yolo_algorithm\\synthetic_dataset\\full"
mrcnn_results = "results\\droplet_detection\\mrcnn_algorithm\\synthetic_dataset\\full"
ccv_results = "results\\droplet_detection\\computer_vision_algorithm\\synthetic_dataset\\full"
cellpose_results = "results\\droplet_detection\\cellpose_algorithm\\synthetic_dataset\\full"


create_segmentation_image("165", 
                          mrcnn_results,
                          path_dataset,
                          "mrcn")

# create_segmentation_image("40", 
#                           yolo_results,
#                           path_dataset,
#                           "yolo")

# create_segmentation_image("40", 
#                           ccv_results,
#                           path_dataset,
#                           "ccv")

# create_segmentation_image("40", 
#                           cellpose_results,
#                           path_dataset,
#                           "cellpose")
