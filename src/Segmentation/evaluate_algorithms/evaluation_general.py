import csv
import pandas as pd

filename = "results\\metrics\\general_avg_values.csv"
columns = [
        'method', 'precision', 'recall', 'f1-score', 'map50', 'map50-95',
        'tp', 'fp', 'fn', 'segmentation_time', 'iou_mask'
    ]
def start_file():
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
    
        # Write the column headers
        writer.writerow(columns)

def drop_eval(csv_file, method_name):

    df = pd.read_csv(csv_file)
    # average_iou = df['iou'].median()
    # average_segmentation_time = df['segmentation_time'].mean()
    # average_segmentation_time = df['segmentation_time'].mean()

    tp = df['tp'].sum()
    fp = df['fp'].sum()
    fn = df['fn'].sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    average_df = pd.DataFrame([{
        'method': method_name,
        'precision': precision,
        'recall': recall,
        'f1-score': f1_score,
        'map50': df['map50'].mean(),
        'map50-95': df['map50-95'].mean(),
        'segmentation_time': df['segmentation_time'].mean()
    }])

    df_gen = pd.read_csv(filename)

    df_gen = df_gen._append(average_df, ignore_index=True)
    df_gen.to_csv(filename, index=False)

drop_eval("results\\metrics\\droplet\\real_dataset\\eval_real_dataset_cv.csv", "droplet_real_dataset_cv")