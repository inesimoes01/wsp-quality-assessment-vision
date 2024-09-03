from pathlib import Path
import math
import os

PROJ_ROOT = Path(__file__).resolve().parents[2]

DATA_SYNTHETIC_NORMAL_WSP_TESTING_DIR = Path("data") / "droplets" / "synthetic_dataset_normal_droplets" / "test_yolo"
DATA_REAL_WSP_TESTING_DIR = Path("data") / "droplets" / "real_dataset_droplets" / "test"

DATA_SYNTHETIC_NORMAL_WSP_MRCNN_DIR = Path("data") / "droplets" / "synthetic_dataset_normal_droplets" / "mrcnn_ready"

#DATA_SYNTHETIC_NORMAL_WSP_DIR = Path("data") / "synthetic_normal_dataset" / "wsp" 
DATA_SYNTHETIC_NORMAL_WSP_DIR = Path("data") / "synthetic_normal_dataset_new" / "wsp" 
DATA_SYNTHETIC_SIMPLE_WSP_DIR = Path("data") / "synthetic_simple_dataset" / "wsp" 
DATA_REAL_RECTANGLE_DIR = Path("data") / "real_rectangle_dataset" / "wsp" 

DATA_SYNTHETIC_AUGMENTED_DIR = Path("data") / "artificial_dataset" / "augmentation" 
DATA_SYNTHETIC_BG_DIR = Path("data") / "artificial_dataset" / "background" 
DATA_SYNTHETIC_RAW_DIR = Path("data") / "artificial_dataset" / "raw" 
DATA_SYNTHETIC_WSP_BACKGROUND_IMG = os.path.join("data", "synthetic_normal_dataset_new", "wsp" , "background.png")

DATA_REAL_RAW_DIR = Path("data") / "real_rectangle_dataset" / "test"  
DATA_REAL_PROC_DIR = Path("data") / "real_dataset" / "processed" 
DATA_REAL_PAPER_DIR = Path("data") / "real_dataset" / "paper" /"raw" 

# RESULTS_REAL_CCV_DIR = Path("results") / "computer_vision_algorithm" / "real_dataset"
# RESULTS_SYNTHETIC_CCV_DIR = Path("results") / "computer_vision_algorithm" / "synthetic_dataset"
# RESULTS_ACCURACY_DIR = Path("results") / "metrics" 

RESULTS_REAL_MRCNN_DIR = Path("results") / "droplet_detection" / "mrcnn_algorithm" / "real_dataset"
RESULTS_SYNTHETIC_MRCNN_DIR = Path("results") / "droplet_detection" / "mrcnn_algorithm" / "synthetic_dataset"
RESULTS_REAL_YOLO_DIR = Path("results") / "droplet_detection" / "yolo_algorithm" / "real_dataset"
RESULTS_SYNTHETIC_YOLO_DIR = Path("results") / "droplet_detection" / "yolo_algorithm" / "synthetic_dataset"
RESULTS_REAL_CCV_DIR = Path("results") / "droplet_detection" / "computer_vision_algorithm" / "real_dataset"
RESULTS_SYNTHETIC_CCV_DIR = Path("results") / "droplet_detection" / "computer_vision_algorithm" / "synthetic_dataset"

RESULTS_CELLPOSE_DIR = Path("results") / "cellpose"
RESULTS_LATEX_DIR = Path("results") / "latex"
RESULTS_LATEX_PIP_DIR = Path("results") / "latex" / "pipeline"
RESULTS_LATEX_PIP_ROI_DIR = Path("results") / "latex" / "pipeline" / "rois"
YOLO_MODEL_DIR = Path("models") / "yolo_droplet" / "50epc_droplet4"


DATA_GENERAL_STATS_FOLDER_NAME = "statistics"
DATA_GENERAL_LABEL_FOLDER_NAME = "label"
DATA_GENERAL_IMAGE_FOLDER_NAME = "image"
DATA_GENERAL_INFO_FOLDER_NAME = "info"
DATA_GENERAL_MASK_SIN_FOLDER_NAME = "mask\\single" 
DATA_GENERAL_MASK_OV_FOLDER_NAME = "mask\\overlapped"
DATA_GENERAL_MASK_FOLDER_NAME = "mask"

RESULTS_GENERAL_STATS_FOLDER_NAME = "statistcs"
RESULTS_GENERAL_ACC_FOLDER_NAME = "accuracy"
RESULTS_GENERAL_LABEL_FOLDER_NAME = "label"
RESULTS_GENERAL_INFO_FOLDER_NAME = "info"
RESULTS_GENERAL_DROPLETCLASSIFICATION_FOLDER_NAME = "droplet_visual_classification"
RESULTS_GENERAL_UNDISTORTED_FOLDER_NAME = "undistorted"
RESULTS_GENERAL_MASK_SIN_FOLDER_NAME = "mask\\single" 
RESULTS_GENERAL_MASK_OV_FOLDER_NAME = "mask\\overlapped"
RESULTS_GENERAL_SEGMENTATIONTIME_FOLDER_NAME = "segmentation_time"

CODE_MRCNN_DIR = Path("src") / "Segmentation" / "droplet" / "cnn" / "MaskRCNN"
MODELS_MRCNN_DIR = Path("models") / "mrcnn"

### CREATE SYNTHETIC DATASET VALUES

NUM_WSP = 300                                    # how many images to create
MAX_NUM_SPOTS = 7000                            # maximum number of spots per image
MIN_NUM_SPOTS = 300                             # minimum number of spots per image

WIDTH_MM, HEIGHT_MM = 76, 26                    # width and height trying to emulate with the artificial dataset
RESOLUTION = 18                                 # resolution of the image based on a minimum value previously agreed with professor
MAX_RADIUS = 10 * math.ceil(RESOLUTION / 30)    # maximum radius for a droplet given the resolution of the image
MIN_RADIUS = math.ceil(RESOLUTION * 0.05)       # minimum radius for a droplet given the resolution of the image

CHARACTERISTIC_PARTICLE_SIZE = 15                # characteristic particle size for distribution of droplet values
UNIFORMITY_CONSTANT = 5                         # uniformity constant for distribution of droplet values (smaller values make radius less uniform)

MEAN_DROPLETS = 2000                            # mean number of droplets per image for the normal distribution
STD_DROPLETS = 750                              # standard deviation of droplets per image for the normal distribution

DROPLET_COLOR_THRESHOLD_1 = 3                     # threshold for the radius for the droplet to be brown if lower or blue if higher
DROPLET_COLOR_THRESHOLD_2 = 5                     # threshold for the radius for the droplet to be brown if lower or blue if higher

ELIPSE_MAJOR_AXE_VALUE = 3                      # value to add to the spot radius to create the major axis of an elipse 

OVERLAPPING_THRESHOLD = 2                       # value to add to the distance between centers to make sure the droplets are actually overlapping

### COMPUTER VISION ALGORITHM ACCURACY VALUES

ACCURACY_DISTANCE_THRESHOLD =  30                # maximum distance of the centers of the droplets when trying to find the pairs
ACCURACY_AREA_THRESHOLD = 100                 # maximum diameter difference between the droplets when trying to find the pairs 

### COMPUTER VISION SEGMENTATION VALUES
DISTANCE_THRESHOLD = 4/500                      # distance between radius for the droplets to be considered too close when eliminating
CIRCULARITY_THRESHOLD = 0.75                     # threshold for the circularity of a contour that separates singular circular droplets from elipse and overlapped droplets
ELIPSE_THRESHOLD = 0.90                         # threshold for the aspect ratio of the contour that indicates if the droplet is an elipse
ELIPSE_AREA_THRESHOLD = 30                      # difference between the total area of the contour and the area of the elipse calculated

BORDER_EXPAND = 10                              # maximum value for the border when cutting the roi in hough transform


# PURPLE_SHADE
outside_color1 = [ '#220000', '#2a0700', '#330000', '#4a1800', '#4c2000', '#512600', '#542900', '#552809']
dark_color1 = ['#5b1a30', '#4e0634', '#5e0b35', '#651f37', '#5e223b', '#65203f', '#65203f', '#571241']
light_color1 = ['#63219f',  '#642aa5',  '#7f24a5',  '#812ba6',  '#6a0fa8',  '#690eab',  '#7b25ac',  '#7530ad',  '#752eb0',  '#7e22b1',  '#8b27bb',  '#732ebd',  '#8e23bf',  '#7326c0',  '#853fc3',  '#812ec6',  '#8125c6',  '#6b1ec6',  '#8727c7',  '#892ace',  '#742dd1',  '#8038e4',  '#aa52e6',  '#a248e6']

# BLUE_SHADE
outside_color2 = [ '#190e00', '#1f150c', '#0e0f13', '#131514', '#1b0f19', '#14141c', '#1b141c', '#131522', '#0c0d22', '#181123', '#141325', '#1c1527', '#181729', '#1b1a2a']
dark_color2 =['#09082a', '#0f0c2b', '#181130', '#0a0a30', '#030430', '#160e33', '#000233', '#191935', '#0e0d35', '#0e0d35', '#140c35', '#181736', '#060838', '#060838', '#060a3a', '#060a3a', '#11143d', '#0d0b3d', '#0e1040', '#030444', '#060845', '#0d0b4a', '#0b094a', '#070654']
light_color2 = ['#181872', '#18107f', '#221d91', '#272595', '#2c2897', '#352ea0', '#2d29a2', '#2524ac', '#2e2db5', '#2c2bb7']


background_colors=['#ffff16', '#f7e52d', '#fff60a', '#fffb0b', '#fee900', '#ffdc1f', '#fffe18', '#fff200', '#fdf41b', '#f7e715', '#ffe827', '#f9d400', '#ebcb00', '#eab742', '#faeb0a', '#ffd600', '#fff808', '#f8e80c', '#f9fa09', '#fff212', '#fff316', '#fbf227', '#e3d22b', '#ddd02b', '#e2d033', '#e2eb67', '#eaf388', '#e2e20f', '#efef04', '#e6eb2d', '#ecf56d', '#ffe700', '#ffe800', '#ffe800', '#f8e600', '#fce600', '#f6df00', '#ffe500', '#fee53f', '#f9e039', '#fdf0aa', '#f8eba5', '#fff59d', '#dac82b', '#e1d52a', '#e2d026', '#e1ca22', '#e3d02f', '#ddd124', '#d59d00', '#f8c10f', '#ffd63d', '#fbad27', '#ffd52e', '#fff50a', '#ffed3b', '#ffbd00', '#ffda25', '#f5c847', '#ffe652', '#f4c223', '#e8ef8d', '#e6ed86', '#e6ed8b', '#e9ef92', '#dfe57b', '#ebf28e', '#edf780', '#e2e753', '#e4ed6a', '#f2f765', '#fdff77', '#e3ea58', '#e5ea65', '#edf272']

background_color_1 = (255, 244, 137)
background_color_2 = (159, 127, 19)