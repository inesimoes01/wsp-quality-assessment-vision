import csv
import pandas as pd

filename = "results\\metrics\\general_avg_values.csv"
columns = [
        'method', 'precision', 'recall', 'f1-score', 'map0.5', 'map0.5-0.95',
        'tp', 'fp', 'fn', 'segmentation_time', 'iou_mask'
    ]
def start_file():
        

    with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the column headers
            writer.writerow(columns)

def drop_eval(csv_file):

        df = pd.read_csv(csv_file)
        # average_iou = df['iou'].median()
        # average_segmentation_time = df['segmentation_time'].mean()
        # average_segmentation_time = df['segmentation_time'].mean()

        average_df = pd.DataFrame([{
            'method': 'droplet_segmentation_cv',
            'precision': df['precision'].mean(),
            'recall': df['recall'].mean(),
            'f1-score': df['f1_score'].mean(),
            'map0.5': df['map0.5'].mean(),
            'map0.5-0.95': df['map0.5-0.95'].mean(),
            'segmentation_time': df['segmentation_time'].mean()
        }])

        df_gen = pd.read_csv(filename)

        df_gen = df_gen._append(average_df, ignore_index=True)
        df_gen.to_csv(filename, index=False)


drop_eval("results\\metrics\\droplet_evaluation_cv.csv", )