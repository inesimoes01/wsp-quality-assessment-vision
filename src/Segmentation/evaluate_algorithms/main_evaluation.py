from pathlib import Path
import csv 
import sys
import pandas as pd
import json
import os
from ultralytics import YOLO
from shapely import Polygon

sys.path.insert(0, 'src')
import Segmentation.evaluate_algorithms.evaluate_paper as evaluate_paper
import Segmentation.evaluate_algorithms.evaluate_droplet as evaluate_droplet
# import Segmentation.evaluate_algorithms.evaluate_droplet_real_dataset as evaluate_droplet_real_dataset
import Common.config as config
from Statistics import Statistics as stats

# WHAT EVALUATIONS TO UPDATE
isDropletCCV, isDropletYOLO, isDropletMRCNN, isPaperCCV, isPaperYOLO = False, True, False, False, False



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
FIELDNAMES_DROPLET_GENERAL_SEGMENTATION = ['method', 'precision', 'recall', 'f1-score', 'map50', 'map50-95', 'tp', 'fp', 'fn', 'segmentation_time', 'iou_mask']


# YOLO MODELS
PAPER_YOLO_MODEL = YOLO(os.path.join("models\\yolo_rectangle\\30epc_rectangle7", "weights", "best.pt"))
DROPLET_YOLO_MODEL = YOLO(os.path.join("models\\yolo_droplet\\50epc_droplet4", "weights", "best.pt"))

# MRCNN MODELS
DROPLET_MRCNN_MODEL = os.path.join("models\\yolo_droplet\\50epc_droplet4", "weights", "best.pt")

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


def new_csv_file(path_to_new_csv, new_csv_fieldnames):
    with open(path_to_new_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=new_csv_fieldnames)
        writer.writeheader()

def update_general_evaluation_droplet_stats(path_general_evaluation, path_individual_evaluation, method):
    df = pd.read_csv(path_individual_evaluation)

    average_df = pd.DataFrame([{
        'method': method,
        'VMD_error': df['VMD_error'].mean(),
        'RSF_error': df['RSF_error'].mean(),
        'CoveragePercentage_error': df['CoveragePercentage_error'].mean(),
        'NoDroplets_error': df['NoDroplets_error'].mean(),
        #'OtherCoveragePercentage_error': df['OtherCoveragePercentage_error'].mean(),
    
        'VMD_median': df['VMD_error'].median(),
        'RSF_median': df['RSF_error'].median(),
        'CoveragePercentage_median': df['CoveragePercentage_error'].median(),
        'NoDroplets_median': df['NoDroplets_error'].median(),
        #'OtherCoveragePercentage_median': df['OtherCoveragePercentage_error'].median(),
        
        'VMD_std': df['VMD_error'].std(),
        'RSF_std': df['RSF_error'].std(),
        'CoveragePercentage_std': df['CoveragePercentage_error'].std(),
        'NoDroplets_std': df['NoDroplets_error'].std(),
        #'OtherCoveragePercentage_std': df['OtherCoveragePercentage_error'].std(),

        'VMD_max': df['VMD_error'].max(),
        'RSF_max': df['RSF_error'].max(),
        'CoveragePercentage_max': df['CoveragePercentage_error'].max(),
        'NoDroplets_max': df['NoDroplets_error'].max(),
        #'OtherCoveragePercentage_max': df['OtherCoveragePercentage_error'].max(),
    }])

    df_gen = pd.read_csv(path_general_evaluation)
    df_gen = df_gen._append(average_df, ignore_index=True)
    df_gen.to_csv(path_general_evaluation, index=False)

def update_general_evaluation_droplet_segm(path_general_evaluation, path_individual_evaluation, method):
    df = pd.read_csv(path_individual_evaluation)

    tp = df['tp'].sum()
    fp = df['fp'].sum()
    fn = df['fn'].sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    average_df = pd.DataFrame([{
        'method': method,
        'precision': precision,
        'recall': recall,
        'f1-score': f1_score,
        'map50': df['map50'].mean(),
        'map50-95': df['map50-95'].mean(),
        'segmentation_time': df['segmentation_time'].mean()
    }])

    df_gen = pd.read_csv(path_general_evaluation)

    df_gen = df_gen._append(average_df, ignore_index=True)
    df_gen.to_csv(path_general_evaluation, index=False)

def update_general_evaluation_paper(path_general_evaluation, path_individual_evaluation, method):
    df = pd.read_csv(path_individual_evaluation)

    average_df = pd.DataFrame([{
        'method': method,
        'iou_mask': df['iou'].median(),
        'segmentation_time': df['segmentation_time'].mean()
    }])

    df_gen = pd.read_csv(path_general_evaluation)

    df_gen = df_gen._append(average_df, ignore_index=True)
    df_gen.to_csv(path_general_evaluation, index=False)

def aux_save_yolo_labels(filename, studio_annotations, path_yolo_labels):
    yolo_annotations = []
    polygons = []

    with open(os.path.join(path_yolo_labels, filename + ".txt"), 'w') as file: 

        for result in studio_annotations:

            points = result['value']['points']

            coordinates = [(point[0]/ 100, point[1] / 100) for point in points]

            polygon = Polygon(coordinates)
            polygons.append(polygon)
            line = '0 ' + ' '.join(f'{x} {y}' for x, y in coordinates) + "\n"
            file.write(line)
        
            yolo_annotations.append(coordinates)

    return polygons

def save_labels_real_dataset(path_yolo_labels, path_studio_labels, path_images):
    polygons = []
    for file in os.listdir(path_images):

        image_name = file.split(".")[0]
        f = open(os.path.join(path_studio_labels, image_name + ".json"))
        data = json.load(f)

        annotations = data['annotations'][0]['result']
        predictions = data['predictions'][0]['result']

        polygons.append(aux_save_yolo_labels(image_name, annotations, path_yolo_labels))
        polygons.append(aux_save_yolo_labels(image_name, predictions, path_yolo_labels))

def check_folders():
    if not os.path.exists(EVAL_MAIN_PAPER_PATH):
        os.makedirs(EVAL_MAIN_PAPER_PATH)
    if not os.path.exists(EVAL_MAIN_DROPLET_GENERAL_PATH):
        os.makedirs(EVAL_MAIN_DROPLET_GENERAL_PATH)
    if not os.path.exists(EVAL_MAIN_DROPLET_REAL_PATH):
        os.makedirs(EVAL_MAIN_DROPLET_REAL_PATH)
    if not os.path.exists(EVAL_MAIN_DROPLET_SYNTHETIC_PATH):
        os.makedirs(EVAL_MAIN_DROPLET_SYNTHETIC_PATH)



def compute_evaluations():
    check_folders()

    if isDropletCCV:
        # REAL DATASET
        evaluate_droplet.main_ccv(FIELDNAMES_DROPLET_SEGMENTATION, FIELDNAMES_DROPLET_STATISTICS, EVAL_DROPLET_SEGM_REAL_DATASET_CV, EVAL_DROPLET_STATS_REAL_DATASET_CV, config.DATA_REAL_WSP_TESTING_DIR, config.RESULTS_REAL_CCV_DIR, 0.5, 10)
        update_general_evaluation_droplet_segm(EVAL_DROPLET_SEGM_GENERAL, EVAL_DROPLET_SEGM_REAL_DATASET_CV, "droplet_real_dataset_ccv")
        update_general_evaluation_droplet_stats(EVAL_DROPLET_STATS_GENERAL, EVAL_DROPLET_STATS_REAL_DATASET_CV, "droplet_real_dataset_ccv")

        # SYNTHETIC DATASET
        evaluate_droplet.main_ccv(FIELDNAMES_DROPLET_SEGMENTATION, FIELDNAMES_DROPLET_STATISTICS, EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_CV, EVAL_DROPLET_STATS_SYNTHETIC_DATASET_CV, config.DATA_SYNTHETIC_NORMAL_WSP_TESTING_DIR, config.RESULTS_SYNTHETIC_CCV_DIR, 0.5, 10)
        update_general_evaluation_droplet_segm(EVAL_DROPLET_SEGM_GENERAL, EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_CV, "droplet_synthetic_dataset_ccv")
        update_general_evaluation_droplet_stats(EVAL_DROPLET_STATS_GENERAL, EVAL_DROPLET_STATS_SYNTHETIC_DATASET_CV, "droplet_synthetic_dataset_ccv") 
    
    if isDropletYOLO:
        # SYNTHETIC DATASET
        evaluate_droplet.main_yolo(FIELDNAMES_DROPLET_SEGMENTATION, FIELDNAMES_DROPLET_STATISTICS, EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO, EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO, config.DATA_SYNTHETIC_NORMAL_WSP_TESTING_DIR, config.RESULTS_SYNTHETIC_CCV_DIR, DROPLET_YOLO_MODEL, 0.5, 10, 76)
        update_general_evaluation_droplet_segm(EVAL_DROPLET_SEGM_GENERAL, EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO, "droplet_synthetic_dataset_yolo")
        update_general_evaluation_droplet_stats(EVAL_DROPLET_STATS_GENERAL, EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO, "droplet_synthetic_dataset_yolo")
        
        # REAL DATASET
        evaluate_droplet.main_yolo(FIELDNAMES_DROPLET_SEGMENTATION, FIELDNAMES_DROPLET_STATISTICS, EVAL_DROPLET_SEGM_REAL_DATASET_YOLO, EVAL_DROPLET_STATS_REAL_DATASET_YOLO, config.DATA_REAL_WSP_TESTING_DIR, config.RESULTS_REAL_CCV_DIR, DROPLET_YOLO_MODEL, 0.5, 10, 76)
        update_general_evaluation_droplet_segm(EVAL_DROPLET_SEGM_GENERAL, EVAL_DROPLET_SEGM_REAL_DATASET_YOLO, "droplet_real_dataset_yolo")
        update_general_evaluation_droplet_stats(EVAL_DROPLET_STATS_GENERAL, EVAL_DROPLET_STATS_REAL_DATASET_YOLO, "droplet_real_dataset_yolo")
        
    if isDropletMRCNN:
        # SYNTHETIC DATASET
        evaluate_droplet.main_mrcnn(FIELDNAMES_DROPLET_SEGMENTATION, FIELDNAMES_DROPLET_STATISTICS, EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO, EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO, config.DATA_SYNTHETIC_WSP_TESTING_DIR, config.RESULTS_SYNTHETIC_CCV_DIR, DROPLET_MRCNN_MODEL, 0.5, 10, 76)
        update_general_evaluation_droplet_segm(EVAL_DROPLET_SEGM_GENERAL, EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO, "droplet_synthetic_dataset_mrcnn")
        update_general_evaluation_droplet_stats(EVAL_DROPLET_STATS_GENERAL, EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO, "droplet_synthetic_dataset_mrcnn")

        # REAL DATASET
        evaluate_droplet.main_mrcnn(FIELDNAMES_DROPLET_SEGMENTATION, FIELDNAMES_DROPLET_STATISTICS, EVAL_DROPLET_SEGM_REAL_DATASET_YOLO, EVAL_DROPLET_STATS_REAL_DATASET_YOLO, config.DATA_REAL_WSP_TESTING_DIR, config.RESULTS_REAL_CCV_DIR, DROPLET_MRCNN_MODEL, 0.5, 10, 76)
        update_general_evaluation_droplet_segm(EVAL_DROPLET_SEGM_GENERAL, EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO, "droplet_real_dataset_mrcnn")
        update_general_evaluation_droplet_stats(EVAL_DROPLET_STATS_GENERAL, EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO, "droplet_real_dataset_mrcnn")

    if isPaperCCV:
        evaluate_paper.main_ccv(EVAL_PAPER_SEGM_CV, FIELDNAMES_RECTANGLE, config.DATA_REAL_PAPER_DIR)
        update_general_evaluation_paper(EVAL_PAPER_SEGM_GENERAL, EVAL_PAPER_SEGM_CV, "paper_classicalcomputervision")
    
    if isPaperYOLO:
        evaluate_paper.main_yolo(EVAL_PAPER_SEGM_YOLO, FIELDNAMES_RECTANGLE, config.DATA_REAL_PAPER_DIR, PAPER_YOLO_MODEL)
        update_general_evaluation_paper(EVAL_PAPER_SEGM_GENERAL, EVAL_PAPER_SEGM_YOLO, "paper_yolo")


compute_evaluations()

