## PATHS
path_main_dataset = 'images\\artificial_dataset'
path_to_images_folder = 'images\\artificial_dataset\\image'
path_to_statistics_gt_folder = 'images\\artificial_dataset\\statistic\\gt'
path_to_statistics_pred_folder = 'images\\artificial_dataset\\statistic\\c'
path_to_outputs_folder = 'images\\artificial_dataset\\outputs'
path_to_masks_overlapped_pred_folder = 'images\\artificial_dataset\\masks\\c\\overlapped'
path_to_masks_overlapped_gt_folder = 'images\\artificial_dataset\\masks\\gt\\overlapped'
path_to_masks_single_pred_folder = 'images\\artificial_dataset\\masks\\c\\single'
path_to_masks_single_gt_folder = 'images\\artificial_dataset\\masks\\gt\\single'
path_to_real_dataset_inesc = 'images\\inesc_dataset'
path_to_real_dataset_inesc_original = 'images\\inesc_dataset\\original'
path_to_real_dataset_inesc_undistorted = 'images\\inesc_dataset\\undistorted'
path_to_real_dataset = 'images\\real_images'


### VALUES
max_num_spots = 1000
min_num_spots = 300
max_radius = 15
width_mm, height_mm = 76, 26
resolution = 30
num_wsp = 20

### accuracy
distance_threshold = 5
diameter_threshold = 5

### distinguish between overlapped and singular
circularity_threshold = 0.8

### border for ROI
border_expand = 1

