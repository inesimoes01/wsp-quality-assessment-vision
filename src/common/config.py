from pathlib import Path
import math
import os

PROJ_ROOT = Path(__file__).resolve().parents[2]


# DATA_ARTIFICIAL_RAW_DIR = Path("data") / "artificial_dataset_versions" / "only_single_circles" / "raw" 
# DATA_ARTIFICIAL_RAW_IMAGE_DIR = DATA_ARTIFICIAL_RAW_DIR / "image"
# DATA_ARTIFICIAL_RAW_INFO_DIR = DATA_ARTIFICIAL_RAW_DIR / "info"
# DATA_ARTIFICIAL_RAW_LABEL_DIR = DATA_ARTIFICIAL_RAW_DIR / "label"
# DATA_ARTIFICIAL_RAW_MASK_SIN_DIR = DATA_ARTIFICIAL_RAW_DIR / "mask" / "single"
# DATA_ARTIFICIAL_RAW_MASK_OV_DIR = DATA_ARTIFICIAL_RAW_DIR / "mask" / "overlapped"
# DATA_ARTIFICIAL_RAW_STATISTICS_DIR = DATA_ARTIFICIAL_RAW_DIR / "statistics"
# DATA_ARTIFICIAL_RAW_BACKGROUND_IMG = os.path.join("data", "artificial_dataset_versions", "only_single_circles", "raw" , "background.png")

DATA_ARTIFICIAL_WSP_DIR = Path("data") / "artificial_dataset" / "wsp" 
DATA_ARTIFICIAL_WSP_IMAGE_DIR = DATA_ARTIFICIAL_WSP_DIR / "image"
DATA_ARTIFICIAL_WSP_INFO_DIR = DATA_ARTIFICIAL_WSP_DIR / "info"
DATA_ARTIFICIAL_WSP_LABEL_DIR = DATA_ARTIFICIAL_WSP_DIR / "label"
DATA_ARTIFICIAL_WSP_MASK_SIN_DIR = DATA_ARTIFICIAL_WSP_DIR / "mask" / "single"
DATA_ARTIFICIAL_WSP_MASK_OV_DIR = DATA_ARTIFICIAL_WSP_DIR / "mask" / "overlapped"
DATA_ARTIFICIAL_WSP_STATISTICS_DIR = DATA_ARTIFICIAL_WSP_DIR / "statistics"
DATA_ARTIFICIAL_WSP_BACKGROUND_IMG = os.path.join("data", "artificial_dataset_main", "raw" , "background.png")

DATA_ARTIFICIAL_BG_DIR = Path("data") / "artificial_dataset" / "background" 
DATA_ARTIFICIAL_BG_IMAGE_DIR = DATA_ARTIFICIAL_BG_DIR / "image"
DATA_ARTIFICIAL_BG_LABEL_DIR = DATA_ARTIFICIAL_BG_DIR / "label"

DATA_ARTIFICIAL_RAW_DIR = Path("data") / "artificial_dataset" / "raw" 
DATA_ARTIFICIAL_RAW_IMAGE_DIR = DATA_ARTIFICIAL_RAW_DIR / "image"
DATA_ARTIFICIAL_RAW_LABEL_DIR = DATA_ARTIFICIAL_RAW_DIR / "label"



DATA_REAL_RAW_DIR = Path("data") / "real_dataset" / "raw" 
DATA_REAL_RAW_IMAGE_DIR = DATA_REAL_RAW_DIR / "image"
DATA_REAL_RAW_INFO_DIR = DATA_REAL_RAW_DIR / "info"
DATA_REAL_RAW_LABEL_DIR = DATA_REAL_RAW_DIR / "label"
DATA_REAL_RAW_MASK_SIN_DIR = DATA_REAL_RAW_DIR / "mask" / "single"
DATA_REAL_RAW_MASK_OV_DIR = DATA_REAL_RAW_DIR / "mask" / "overlapped"
DATA_REAL_RAW_STATISTICS_DIR = DATA_REAL_RAW_DIR / "statistics"


DATA_REAL_PROC_DIR = Path("data") / "real_dataset" / "processed" 
DATA_REAL_PROC_IMAGE_DIR = DATA_REAL_PROC_DIR / "image"
DATA_REAL_PROC_INFO_DIR = DATA_REAL_PROC_DIR / "info"
DATA_REAL_PROC_LABEL_DIR = DATA_REAL_PROC_DIR / "label"
DATA_REAL_PROC_MASK_SIN_DIR = DATA_REAL_PROC_DIR / "mask" / "single"
DATA_REAL_PROC_MASK_OV_DIR = DATA_REAL_PROC_DIR / "mask" / "overlapped"
DATA_REAL_PROC_STATISTICS_DIR = DATA_REAL_PROC_DIR / "statistics"

RESULTS_CV_DIR = Path("results") / "computer_vision_algorithm"
RESULTS_CV_ACCURACY_DIR = RESULTS_CV_DIR / "accuracy"
RESULTS_CV_DROPLETCLASSIFICATION_DIR = RESULTS_CV_DIR / "droplet_visual_classification"
RESULTS_CV_INFO_DIR = RESULTS_CV_DIR / "info"
RESULTS_CV_MASK_SIN_DIR = RESULTS_CV_DIR / "mask" / "single"
RESULTS_CV_MASK_OV_DIR = RESULTS_CV_DIR / "mask" / "overlapped"
RESULTS_CV_STATISTICS_DIR = RESULTS_CV_DIR / "statistics"
RESULTS_CV_UNDISTORTED_DIR = RESULTS_CV_DIR / "undistorted"

RESULTS_LATEX = Path("results") / "latex"

YOLO_MODEL_DIR = Path("models") 


### CREATE ARTIFICIAL DATASET VALUES

NUM_WSP = 1000                                    # how many images to create
MAX_NUM_SPOTS = 7000                            # maximum number of spots per image
MIN_NUM_SPOTS = 300                             # minimum number of spots per image

WIDTH_MM, HEIGHT_MM = 76, 26                    # width and height trying to emulate with the artificial dataset
RESOLUTION = int(465*0.039)                     # resolution of the image based on a minimum value previously agreed with professor
MAX_RADIUS = 10 * math.ceil(RESOLUTION / 30)    # maximum radius for a droplet given the resolution of the image
MIN_RADIUS = math.ceil(RESOLUTION * 0.05)       # minimum radius for a droplet given the resolution of the image

CHARACTERISTIC_PARTICLE_SIZE = 15                # characteristic particle size for distribution of droplet values
UNIFORMITY_CONSTANT = 3                         # uniformity constant for distribution of droplet values (smaller values make radius less uniform)

MEAN_DROPLETS = 2000                            # mean number of droplets per image for the normal distribution
STD_DROPLETS = 500                              # standard deviation of droplets per image for the normal distribution

DROPLET_COLOR_THRESHOLD_1 = 3                     # threshold for the radius for the droplet to be brown if lower or blue if higher
DROPLET_COLOR_THRESHOLD_2 = 5                     # threshold for the radius for the droplet to be brown if lower or blue if higher

ELIPSE_MAJOR_AXE_VALUE = 3                      # value to add to the spot radius to create the major axis of an elipse 

OVERLAPPING_THRESHOLD = 2                       # value to add to the distance between centers to make sure the droplets are actually overlapping

### COMPUTER VISION ALGORITHM ACCURACY VALUES

ACCURACY_DISTANCE_THRESHOLD = 5                 # maximum distance of the centers of the droplets when trying to find the pairs
ACCURACY_DIAMETER_THRESHOLD = 5                 # maximum diameter difference between the droplets when trying to find the pairs 

### COMPUTER VISION SEGMENTATION VALUES
DISTANCE_THRESHOLD = 4/500                      # distance between radius for the droplets to be considered too close when eliminating
CIRCULARITY_THRESHOLD = 0.8                     # threshold for the circularity of a contour that separates singular circular droplets from elipse and overlapped droplets
ELIPSE_THRESHOLD = 0.90                         # threshold for the aspect ratio of the contour that indicates if the droplet is an elipse
ELIPSE_AREA_THRESHOLD = 30                      # difference between the total area of the contour and the area of the elipse calculated

BORDER_EXPAND = 10                              # maximum value for the border when cutting the roi in hough transform
