from ultralytics import YOLO
#from matplotlib import pyplot as plt
#import torch
import glob
import os
import numpy as np

from ultralytics.models.yolo.segment import SegmentationValidator
import pandas as pd


yaml_file = "data\\rectangle_dataset\\yolo\\configuration.yaml"
train_model_name = "results\\yolo_rectangle\\200epc_rectangle2\\weights\\best.pt"
results_folder = "results\\latex\\yolo_rectangle_val"

## saves the csv files for yolov8 validation curves
def save_val_csv():

    arguments = dict(model=train_model_name, data=yaml_file, split='val')
    validator = SegmentationValidator(args=arguments)
    validator()
    
    valid = validator.metrics

    precision_recall_box = valid.curves_results[0]
    f1_score_box = valid.curves_results[1]
    precision_confidence_box = valid.curves_results[2]
    recall_confidence_box = valid.curves_results[3]

    precision_recall_mask = valid.curves_results[4]
    f1_score_mask = valid.curves_results[5]
    precision_confidence_mask = valid.curves_results[6]
    recall_confidence_mask = valid.curves_results[7]

    pr_data = np.column_stack((precision_recall_box[0], precision_recall_box[1][0]))
    pr_df = pd.DataFrame(pr_data, columns=['Precision', 'Recall'])
    pr_df.to_csv(os.path.join(results_folder, 'precision_recall_box.csv'), index=False)

    pr_data = np.column_stack((f1_score_box[0], f1_score_box[1][0]))
    pr_df = pd.DataFrame(pr_data, columns=['F1_Score', 'Confidence'])
    pr_df.to_csv(os.path.join(results_folder, 'f1score_confidence_box.csv'), index=False)

    pr_data = np.column_stack((precision_confidence_box[0], precision_confidence_box[1][0]))
    pr_df = pd.DataFrame(pr_data, columns=['Precision', 'Confidence'])
    pr_df.to_csv(os.path.join(results_folder, 'precision_confidence_box.csv'), index=False)

    pr_data = np.column_stack((recall_confidence_box[0], recall_confidence_box[1][0]))
    pr_df = pd.DataFrame(pr_data, columns=['Recall', 'Confidence'])
    pr_df.to_csv(os.path.join(results_folder, 'recall_confidence_box.csv'), index=False)

    
    pr_data = np.column_stack((precision_recall_mask[0], precision_recall_mask[1][0]))
    pr_df = pd.DataFrame(pr_data, columns=['Precision', 'Recall'])
    pr_df.to_csv(os.path.join(results_folder, 'precision_recall_mask.csv'), index=False)

    pr_data = np.column_stack((f1_score_mask[0], f1_score_mask[1][0]))
    pr_df = pd.DataFrame(pr_data, columns=['F1_Score', 'Confidence'])
    pr_df.to_csv(os.path.join(results_folder, 'f1_score_mask.csv'), index=False)

    pr_data = np.column_stack((precision_confidence_mask[0], precision_confidence_mask[1][0]))
    pr_df = pd.DataFrame(pr_data, columns=['Precision', 'Confidence'])
    pr_df.to_csv(os.path.join(results_folder, 'precision_confidence_mask.csv'), index=False)

    pr_data = np.column_stack((recall_confidence_mask[0], recall_confidence_mask[1][0]))
    pr_df = pd.DataFrame(pr_data, columns=['Recall', 'Confidence'])
    pr_df.to_csv(os.path.join(results_folder, 'recall_confidence_mask.csv'), index=False)


    # plt.imshow(img)
    # plt.show()
    # cv2.waitKey(0)


img = cv2.imread("data\\real_dataset\\raw\\image\\1_V1_A1.jpg")
find_polygon_yolo(img)