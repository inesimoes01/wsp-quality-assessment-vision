
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

import evaluate_algorithms_config

# WHAT EVALUATIONS TO UPDATE
isDropletCCV, isDropletYOLO, isDropletMRCNN, isPaperCCV, isPaperYOLO = False, False, True, False, False




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
    if not os.path.exists(evaluate_algorithms_config.EVAL_MAIN_PAPER_PATH):
        os.makedirs(evaluate_algorithms_config.EVAL_MAIN_PAPER_PATH)
    if not os.path.exists(evaluate_algorithms_config.EVAL_MAIN_DROPLET_GENERAL_PATH):
        os.makedirs(evaluate_algorithms_config.EVAL_MAIN_DROPLET_GENERAL_PATH)
    if not os.path.exists(evaluate_algorithms_config.EVAL_MAIN_DROPLET_REAL_PATH):
        os.makedirs(evaluate_algorithms_config.EVAL_MAIN_DROPLET_REAL_PATH)
    if not os.path.exists(evaluate_algorithms_config.EVAL_MAIN_DROPLET_SYNTHETIC_PATH):
        os.makedirs(evaluate_algorithms_config.EVAL_MAIN_DROPLET_SYNTHETIC_PATH)



def compute_evaluations():
    check_folders()

    if isDropletCCV:
        # REAL DATASET
        evaluate_droplet.main_ccv(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_CV, evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_CV, config.DATA_REAL_WSP_TESTING_DIR, config.RESULTS_REAL_CCV_DIR, 0.5, 10)
        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_CV, "droplet_real_dataset_ccv")
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_CV, "droplet_real_dataset_ccv")

        # SYNTHETIC DATASET
        evaluate_droplet.main_ccv(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_CV, evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_CV, config.DATA_SYNTHETIC_NORMAL_WSP_TESTING_DIR, config.RESULTS_SYNTHETIC_CCV_DIR, 0.5, 10)
        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_CV, "droplet_synthetic_dataset_ccv")
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_CV, "droplet_synthetic_dataset_ccv") 
    
    if isDropletYOLO:
        model = YOLO(evaluate_algorithms_config.DROPLET_YOLO_MODEL)
        # SYNTHETIC DATASET
        evaluate_droplet.main_yolo(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO, evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO, config.DATA_SYNTHETIC_NORMAL_WSP_TESTING_DIR, config.RESULTS_SYNTHETIC_CCV_DIR, model, 0.5, 10, 76)
        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO, "droplet_synthetic_dataset_yolo")
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO, "droplet_synthetic_dataset_yolo")
        
        # REAL DATASET
        evaluate_droplet.main_yolo(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_YOLO, evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_YOLO, config.DATA_REAL_WSP_TESTING_DIR, config.RESULTS_REAL_CCV_DIR, model, 0.5, 10, 76)
        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_YOLO, "droplet_real_dataset_yolo")
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_YOLO, "droplet_real_dataset_yolo")
        
    if isDropletMRCNN:
        # SYNTHETIC DATASET
        evaluate_droplet.main_mrcnn(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO, evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO, config.DATA_SYNTHETIC_NORMAL_WSP_TESTING_DIR, config.RESULTS_SYNTHETIC_CCV_DIR, evaluate_algorithms_config.DROPLET_MRCNN_MODEL, 0.5, 10, 76)
        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO, "droplet_synthetic_dataset_mrcnn")
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO, "droplet_synthetic_dataset_mrcnn")

        # REAL DATASET
        evaluate_droplet.main_mrcnn(evaluate_algorithms_config.FIELDNAMES_DROPLET_SEGMENTATION, evaluate_algorithms_config.FIELDNAMES_DROPLET_STATISTICS, evaluate_algorithms_config.EVAL_DROPLET_SEGM_REAL_DATASET_YOLO, evaluate_algorithms_config.EVAL_DROPLET_STATS_REAL_DATASET_YOLO, config.DATA_REAL_WSP_TESTING_DIR, config.RESULTS_REAL_CCV_DIR, evaluate_algorithms_config.DROPLET_MRCNN_MODEL, 0.5, 10, 76)
        update_general_evaluation_droplet_segm(evaluate_algorithms_config.EVAL_DROPLET_SEGM_GENERAL, evaluate_algorithms_config.EVAL_DROPLET_SEGM_SYNTHETIC_DATASET_YOLO, "droplet_real_dataset_mrcnn")
        update_general_evaluation_droplet_stats(evaluate_algorithms_config.EVAL_DROPLET_STATS_GENERAL, evaluate_algorithms_config.EVAL_DROPLET_STATS_SYNTHETIC_DATASET_YOLO, "droplet_real_dataset_mrcnn")

    if isPaperCCV:
        evaluate_paper.main_ccv(evaluate_algorithms_config.EVAL_PAPER_SEGM_CV, evaluate_algorithms_config.FIELDNAMES_RECTANGLE, config.DATA_REAL_PAPER_DIR)
        update_general_evaluation_paper(evaluate_algorithms_config.EVAL_PAPER_SEGM_GENERAL, evaluate_algorithms_config.EVAL_PAPER_SEGM_CV, "paper_classicalcomputervision")
    
    if isPaperYOLO:
        model = YOLO(evaluate_algorithms_config.PAPER_YOLO_MODEL)
        evaluate_paper.main_yolo(evaluate_algorithms_config.EVAL_PAPER_SEGM_YOLO, evaluate_algorithms_config.FIELDNAMES_RECTANGLE, config.DATA_REAL_PAPER_DIR, model)
        update_general_evaluation_paper(evaluate_algorithms_config.EVAL_PAPER_SEGM_GENERAL, evaluate_algorithms_config.EVAL_PAPER_SEGM_YOLO, "paper_yolo")


compute_evaluations()

