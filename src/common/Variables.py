import math
import os

## PATHS
path_main_dataset = os.path.normpath('images\\artificial_dataset')
path_to_images_folder = os.path.normpath('images\\artificial_dataset\\image')
path_to_statistics_gt_folder = os.path.normpath('images\\artificial_dataset\\statistic\\gt')
path_to_statistics_pred_folder = os.path.normpath('images\\artificial_dataset\\statistic\\c')
path_to_dropletinfo_gt_folder = os.path.normpath('images\\artificial_dataset\\dropletinfo\\gt')
path_to_dropletinfo_pred_folder = os.path.normpath('images\\artificial_dataset\\dropletinfo\\c')
path_to_outputs_folder = os.path.normpath('images\\artificial_dataset\\outputs')
path_to_masks_overlapped_pred_folder = os.path.normpath('images\\artificial_dataset\\masks\\c\\overlapped')
path_to_masks_overlapped_gt_folder = os.path.normpath('images\\artificial_dataset\\masks\\gt\\overlapped')
path_to_masks_single_pred_folder = os.path.normpath('images\\artificial_dataset\\masks\\c\\single')
path_to_masks_single_gt_folder = os.path.normpath('images\\artificial_dataset\\masks\\gt\\single')
path_to_real_dataset_inesc = os.path.normpath('images\\inesc_dataset')
path_to_real_dataset_inesc_original = os.path.normpath('images\\inesc_dataset\\original')
path_to_real_dataset_inesc_undistorted = os.path.normpath('images\\inesc_dataset\\undistorted')
path_to_real_dataset = os.path.normpath('images\\real_images')
path_to_labels_yolo = os.path.normpath('images\\artificial_dataset\\labels_yolo')



### VALUES
num_wsp = 500
max_num_spots = 1000
min_num_spots = 50

characteristic_particle_size = 7.0  # Characteristic particle size
uniformity_constant = 2.0    # Uniformity constant

# Parameters for the normal distribution
mean_droplets = 500   # Mean number of droplets per image
std_droplets = 200  # Standard deviation of droplets per image


width_mm, height_mm = 76, 26
resolution = int(600*0.039)
max_radius = 15 * math.ceil(resolution / 30)
min_radius = math.ceil(resolution * 0.05)

### accuracy
distance_threshold = 5
diameter_threshold = 5

### distinguish between overlapped and singular
circularity_threshold = 0.8

### border for ROI
border_expand = 1

