from shapely.geometry import Polygon, Point
import os
import cv2
import sys 

sys.path.insert(0, 'src/common')
from Statistics import Statistics
from Droplet import Droplet
import config
from GroundTruth_Statistics import GroundTruth_Statistics
from Accuracy import Accuracy

sys.path.insert(0, 'src/Segmentation_CV')

def cellpose_label_to_droplet(file_path, width_px, width_mm, image_area):

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    polygons = []
    total_area_number = 0
    total_area_list = []
    droplet_list = []

    # save all information from each polygon to then calculate the statistics and the accuracy
    for i, line in enumerate(lines):
        points = list(map(int, line.strip().split(',')))
        coordinates = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
        polygon = Polygon(coordinates)
        polygons.append(polygon)
        pol_area = polygon.area

        total_area_number += pol_area
        total_area_list.append(polygon.area)
    
        center_x, center_y = Point(polygon.centroid).x, Point(polygon.centroid).y
        droplet_list.append(Droplet(center_x, center_y, pol_area, i, []))

    volume_list = sorted(Statistics.area_to_volume(total_area_list, width_px, width_mm))

    # calculate the statistics
    vmd_value, coverage_percentage, rsf_value = Statistics.calculate_statistics(volume_list, image_area, total_area_number)
    predicted_stats = Statistics(vmd_value, rsf_value, coverage_percentage, len(droplet_list), droplet_list)

    return predicted_stats, droplet_list

def calculate_cellpose_accuracy(filename, image, width_mm):

    width, height = image.shape[:2]
    base_file_gt = os.path.join("data", "artificial_dataset_2", "wsp")

    # get the groundtruth
    groundtruth:GroundTruth_Statistics = GroundTruth_Statistics(filename, base_file_gt)
    droplets_groundtruth_dict = {droplet.id: droplet for droplet in groundtruth.droplets}
    stats_groundtruth:Statistics = groundtruth.stats

    # get the predicted statistics
    label_file_path = os.path.join(config.RESULTS_CELLPOSE_DIR, config.RESULTS_GENERAL_LABEL_FOLDER_NAME, filename + ".txt")

    stats_predicted, predicted_droplets = cellpose_label_to_droplet(label_file_path, width, width_mm, (width * height))
    droplets_predicted_dict = {droplet.id: droplet for droplet in predicted_droplets}
    
    Accuracy(droplets_predicted_dict, droplets_groundtruth_dict, filename, stats_predicted, stats_groundtruth, config.RESULTS_CELLPOSE_DIR)


img = cv2.imread("data\\artificial_dataset_2\\wsp\\image\\0.png")
calculate_cellpose_accuracy("0", img, 76)