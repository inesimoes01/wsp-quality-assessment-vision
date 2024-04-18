## PATHS
path_main_dataset = 'images\\artificial_dataset'
path_to_images_folder = 'images\\artificial_dataset\\image'
path_to_statistics_gt_folder = 'images\\artificial_dataset\\statistic\\gt'
path_to_statistics_c_folder = 'images\\artificial_dataset\\statistic\\c'
path_to_outputs_folder = 'images\\artificial_dataset\\outputs'
path_to_masks_overlapped_folder = 'images\\artificial_dataset\\masks\\overlapped'
path_to_masks_single_circle_folder = 'images\\artificial_dataset\\masks\\single_circle'
path_to_masks_single_ellipse_folder = 'images\\artificial_dataset\\masks\\single_ellipse'
path_to_real_dataset_inesc = 'images\\inesc_dataset'
path_to_real_dataset_inesc_original = 'images\\inesc_dataset\\original'
path_to_real_dataset_inesc_undistorted = 'images\\inesc_dataset\\undistorted'
path_to_real_dataset = 'images\\real_images'


### VALUES
max_num_spots = 1000
min_num_spots = 300
max_radius = 15
width, height = 76, 26
resolution = 30
num_wsp = 20

### accuracy
distance_threshold = 5
radius_threshold = 5

### distinguish between overlapped and singular
circularity_threshold = 0.8

### border for ROI
border_expand = 2

