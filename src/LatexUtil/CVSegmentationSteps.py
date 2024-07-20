import cv2
import os
import sys

sys.path.insert(0, 'src/common')
import config
import Util
sys.path.insert(0, 'src/Segmentation_CV')
from Segmentation import Segmentation
from GroundTruth_Statistics import GroundTruth_Statistics
from Accuracy import Accuracy
from Statistics import Statistics
from Distortion import Distortion


file = "results\\latex\\pipeline\\original.png"
# read image
image_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
image_colors = cv2.imread(file)  

image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)

# calculate statistics
calculated:Segmentation = Segmentation(image_colors, image_gray, "original", True, True, 0, config.WIDTH_MM, config.HEIGHT_MM)
droplets_calculated_dict = {droplet.id: droplet for droplet in calculated.droplets_data}
stats_calculated:Statistics = calculated.stats


# # save ground truth
# groundtruth:GroundTruth_Statistics = GroundTruth_Statistics("Convencional_Exterior_Arv1_25m", config.DATA_ARTIFICIAL_WSP_DIR)
# droplets_groundtruth_dict = {droplet.id: droplet for droplet in groundtruth.droplets}
# stats_groundtruth:Statistics = groundtruth.stats


# # calculate accuracy values
# acc:Accuracy = Accuracy(droplets_calculated_dict, droplets_groundtruth_dict, "Convencional_Exterior_Arv1_25m", stats_calculated, stats_groundtruth, config.DATA_ARTIFICIAL_WSP_DIR)





