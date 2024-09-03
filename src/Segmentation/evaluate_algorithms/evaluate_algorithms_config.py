from pathlib import Path
import os

# FIELDNAMES
FIELDNAMES_RECTANGLE = ["file", "iou", "segmentation_time"]
FIELDNAMES_RECTANGLE_GENERAL = ["method", "iou", "segmentation_time"]
FIELDNAMES_DROPLET_STATISTICS = ["file", 
                                "VMD_pred", "VMD_gt", "VMD_error", 
                                "RSF_pred", "RSF_gt", "RSF_error", 
                                "CoveragePercentage_pred", "CoveragePercentage_gt", "CoveragePercentage_error", 
                                "NoDroplets_pred", "NoDroplets_gt", "NoDroplets_error", 
                                "NoOverlappedDroplets_pred", "NoOverlappedDroplets_gt", "NoOverlappedDroplets_error",
                                "OverlappedDropletsPercentage_pred", "OverlappedDropletsPercentage_gt", "OverlappedDropletsPercentage_error"]
FIELDNAMES_DROPLET_SEGMENTATION = ["file", "precision", "recall", "f1_score", "map50", "map50-95", "tp", "fp", "fn", "segmentation_time"]
FIELDNAMES_DROPLET_GENERAL_STATISTICS = ["method", 
                                "VMD_error", "RSF_error", "CoveragePercentage_error", "NoDroplets_error", "OtherCoveragePercentage_error",
                                "VMD_median", "RSF_median", "CoveragePercentage_median", "NoDroplets_median", "OtherCoveragePercentage_median",
                                "VMD_std", "RSF_std", "CoveragePercentage_std", "NoDroplets_std", "OtherCoveragePercentage_std",
                                "VMD_max", "RSF_max", "CoveragePercentage_max", "NoDroplets_max", "OtherCoveragePercentage_max",]
FIELDNAMES_DROPLET_GENERAL_SEGMENTATION = ['method', 'precision', 'recall', 'f1-score', 'map50', 'map50-95', 'tp', 'fp', 'fn', 'segmentation_time']

FIELDNAMES_SEGMENTATION_TIME = ['Filename', 'Segmentation Time (seconds)']
FIELDNAMES_PREDICTED_STATISTICS = ['VMD', 'RSF', 'Coverage %', 'Nº Droplets', 'Overlapped Droplets %', 'Number of overlapped droplets']

# YOLO MODELS
PAPER_YOLO_MODEL = os.path.join("models\\yolo_rectangle\\30epc_rectangle7", "weights", "best.pt")
DROPLET_YOLO_MODEL = os.path.join("models\\droplets\\yolo_droplet\\50epc_droplet4", "weights", "best.pt")

# MRCNN MODELS
DROPLET_MRCNN_MODEL = "models\\droplets\\mrcnn\\mask_rcnn_droplet_dataset_0050.h5"

EVAL_MAIN_DROPLET_REAL_PATH = Path("results") / "evaluation" / "droplet" / "real_dataset"
EVAL_MAIN_DROPLET_SYNTHETIC_PATH = Path("results") / "evaluation" / "droplet" / "synthetic_dataset" 
EVAL_MAIN_DROPLET_GENERAL_PATH =  Path("results") / "evaluation" / "droplet" / "general"
EVAL_MAIN_PAPER_PATH = Path("results") / "evaluation" / "paper" 

# DROPLET REAL DATASET WITH CV AND YOLO
EVAL_DROPLET_SEGM_REAL_DATASET_CV = os.path.join(EVAL_MAIN_DROPLET_REAL_PATH, "droplet_real_segmentation_cv.csv")
EVAL_DROPLET_STATS_REAL_DATASET_CV = os.path.join(EVAL_MAIN_DROPLET_REAL_PATH, "droplet_real_statistics_cv.csv")
EVAL_DROPLET_SEGM_REAL_DATASET_YOLO = os.path.join(EVAL_MAIN_DROPLET_REAL_PATH, "droplet_real_segmentation_yolo.csv")
EVAL_DROPLET_STATS_REAL_DATASET_YOLO = os.path.join(EVAL_MAIN_DROPLET_REAL_PATH, "droplet_real_statistics_yolo.csv")
EVAL_DROPLET_SEGM_REAL_DATASET_MRCNN = os.path.join(EVAL_MAIN_DROPLET_REAL_PATH, "droplet_real_segmentation_mrcnn.csv")
EVAL_DROPLET_STATS_REAL_DATASET_MRCNN = os.path.join(EVAL_MAIN_DROPLET_REAL_PATH, "droplet_real_statistics_mrcnn.csv")
EVAL_DROPLET_STATS_REAL_DATASET_DROPLEAF = os.path.join(EVAL_MAIN_DROPLET_REAL_PATH, "droplet_real_statistics_dropleaf.csv")

# DROPLET SYNTHETIC DATASET WITH CV AND YOLO
EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_CV = os.path.join(EVAL_MAIN_DROPLET_SYNTHETIC_PATH, "droplet_synthetic_segmentation_cv.csv")
EVAL_DROPLET_STATS_SYNTHETIC_DATASET_CV = os.path.join(EVAL_MAIN_DROPLET_SYNTHETIC_PATH, "droplet_synthetic_statistics_cv.csv")
EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO = os.path.join(EVAL_MAIN_DROPLET_SYNTHETIC_PATH, "droplet_synthetic_segmentation_yolo.csv")
EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO = os.path.join(EVAL_MAIN_DROPLET_SYNTHETIC_PATH, "droplet_synthetic_statistics_yolo.csv")
EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_MRCNN = os.path.join(EVAL_MAIN_DROPLET_SYNTHETIC_PATH, "droplet_synthetic_segmentation_mrcnn.csv")
EVAL_DROPLET_STATS_SYNTHETIC_DATASET_MRCNN = os.path.join(EVAL_MAIN_DROPLET_SYNTHETIC_PATH, "droplet_synthetic_statistics_mrcnn.csv")
EVAL_DROPLET_STATS_SYNTHETIC_DATASET_DROPLEAF = os.path.join(EVAL_MAIN_DROPLET_SYNTHETIC_PATH, "droplet_synthetic_statistics_dropleaf.csv")

# DROPLET GENERAL EVAL
EVAL_DROPLET_SEGM_GENERAL = os.path.join(EVAL_MAIN_DROPLET_GENERAL_PATH, "droplet_real_segmentation_general.csv")
EVAL_DROPLET_STATS_GENERAL = os.path.join(EVAL_MAIN_DROPLET_GENERAL_PATH, "droplet_real_statistics_general.csv")


# PAPER DATASET
EVAL_PAPER_SEGM_CV = os.path.join(EVAL_MAIN_PAPER_PATH, "paper_cv.csv")
EVAL_PAPER_SEGM_YOLO = os.path.join(EVAL_MAIN_PAPER_PATH, "paper_yolo.csv")

# PAPER GENERAL EVAL
EVAL_PAPER_SEGM_GENERAL = os.path.join(EVAL_MAIN_PAPER_PATH, "paper_general.csv")