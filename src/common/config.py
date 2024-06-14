from pathlib import Path
import math

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_ARTIFICIAL_RAW_DIR = PROJ_ROOT / "data" / "artificial_dataset" / "raw" 
DATA_ARTIFICIAL_RAW_IMAGE_DIR = DATA_ARTIFICIAL_RAW_DIR / "image"
DATA_ARTIFICIAL_RAW_INFO_DIR = DATA_ARTIFICIAL_RAW_DIR / "info"
DATA_ARTIFICIAL_RAW_LABEL_DIR = DATA_ARTIFICIAL_RAW_DIR / "label"
DATA_ARTIFICIAL_RAW_MASK_SIN_DIR = DATA_ARTIFICIAL_RAW_DIR / "mask" / "single"
DATA_ARTIFICIAL_RAW_MASK_OV_DIR = DATA_ARTIFICIAL_RAW_DIR / "mask" / "overlapped"
DATA_ARTIFICIAL_RAW_STATISTICS_DIR = DATA_ARTIFICIAL_RAW_DIR / "statistics"

DATA_REAL_RAW_DIR = PROJ_ROOT / "data" / "real_dataset" / "raw" 
DATA_REAL_RAW_IMAGE_DIR = DATA_REAL_RAW_DIR / "image"
DATA_REAL_RAW_INFO_DIR = DATA_REAL_RAW_DIR / "info"
DATA_REAL_RAW_LABEL_DIR = DATA_REAL_RAW_DIR / "label"
DATA_REAL_RAW_MASK_SIN_DIR = DATA_REAL_RAW_DIR / "mask" / "single"
DATA_REAL_RAW_MASK_OV_DIR = DATA_REAL_RAW_DIR / "mask" / "overlapped"
DATA_REAL_RAW_STATISTICS_DIR = DATA_REAL_RAW_DIR / "statistics"

RESULTS_CV_DIR = PROJ_ROOT / "results" / "computer_vision_algorithm"
RESULTS_CV_ACCURACY_DIR = RESULTS_CV_DIR / "accuracy"
RESULTS_CV_DROPLETCLASSIFICATION_DIR = RESULTS_CV_DIR / "droplet_visual_classification"
RESULTS_CV_INFO_DIR = RESULTS_CV_DIR / "info"
RESULTS_CV_MASK_SIN_DIR = RESULTS_CV_DIR / "mask" / "single"
RESULTS_CV_MASK_OV_DIR = RESULTS_CV_DIR / "mask" / "overlapped"
RESULTS_CV_STATISTICS_DIR = RESULTS_CV_DIR / "statistics"
RESULTS_CV_UNDISTORTED_DIR = RESULTS_CV_DIR / "undistorted"


### CREATE ARTIFICIAL DATASET VALUES

NUM_WSP = 500                                   # how many images to create
MAX_NUM_SPOTS = 1000                            # maximum number of spots per image
MIN_NUM_SPOTS = 50                              # minimum number of spots per image

WIDTH_MM, HEIGHT_MM = 76, 26                    # width and height trying to emulate with the artificial dataset
RESOLUTION = int(600*0.039)                     # resolution of the image based on a minimum value previously agreed with professor
MAX_RADIUS = 15 * math.ceil(RESOLUTION / 30)    # maximum radius for a droplet given the resolution of the image
MIN_RADIUS = math.ceil(RESOLUTION * 0.05)       # minimum radius for a droplet given the resolution of the image

CHARACTERISTIC_PARTICLE_SIZE = 7.0              # characteristic particle size for distribution of droplet values
UNIFORMITY_CONSTANT = 2.0                       # uniformity constant for distribution of droplet values

MEAN_DROPLETS = 500                             # mean number of droplets per image for the normal distribution
STD_DROPLETS = 200                              # standard deviation of droplets per image for the normal distribution

### COMPUTER VISION ALGORITHM ACCURACY VALUES

ACCURACY_DISTANCE_THRESHOLD = 5                 # maximum distance of the centers of the droplets when trying to find the pairs
ACCURACY_DIAMETER_THRESHOLD = 5                 # maximum diameter difference between the droplets when trying to find the pairs 

### COMPUTER VISION SEGMENTATION VALUES
DISTANCE_THRESHOLD = 4/500                      # distance between radius for the droplets to be considered too close when eliminating
CIRCULARITY_THRESHOLD = 0.8                     # threshold for the circularity of a contour that separates singular circular droplets from elipse and overlapped droplets
ELIPSE_THRESHOLD = 0.90                         # threshold for the aspect ratio of the contour that indicates if the droplet is an elipse
ELIPSE_AREA_THRESHOLD = 30                      # difference between the total area of the contour and the area of the elipse calculated

BORDER_EXPAND = 10                              # maximum value for the border when cutting the roi in hough transform
