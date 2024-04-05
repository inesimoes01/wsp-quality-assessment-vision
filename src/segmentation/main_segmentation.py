import sys
import cv2
from matplotlib import pyplot as plt 
import os
import copy
import numpy as np

from Calculated_Statistics import Calculated_Statistics
from GroundTruth_Statistics import GroundTruth_Statistics
from Accuracy import Accuracy

sys.path.insert(0, 'src/others')
from util import *
from variables import *

sys.path.insert(0, 'src')
from Droplet import *

# delete old outputs
delete_folder_contents(path_to_separation_folder)
delete_folder_contents(path_to_outputs_folder)

# for each one of the images of the dataset
for file in os.listdir(path_to_images_folder):
    # get name of the file
    parts = file.split(".")
    filename = parts[0]

    # treat image
    in_image = cv2.imread(os.path.join(path_to_images_folder, filename + ".png"))
    in_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
    out_image = copy.copy(in_image)

    # calculate statistics
    stats_calculated:list[Droplet] = Calculated_Statistics(out_image, filename).droplets_data

    # save ground truth
    stats_groundtruth:list[Droplet] = GroundTruth_Statistics(filename, out_image).droplets

    Accuracy(stats_calculated, stats_groundtruth, filename)