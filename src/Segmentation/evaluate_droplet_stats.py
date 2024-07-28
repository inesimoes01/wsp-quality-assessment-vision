import pandas as pd
import os

predictions_df = pd.read_csv("results\\metrics\\dropleaf_stats.csv")

id_list = predictions_df['id'].tolist() 

for id in id_list:

    ground_truth_df = pd.read_csv(os.path.join("data\\synthetic_normal_dataset_new\\wsp\\statistics", str(id) + ".csv"),  index_col=0)
    predictions = predictions_df.loc[predictions_df['id'] == id]

    ground_truth = {
        'VMD': ground_truth_df.loc['VMD', 'GroundTruth'],
        'RSF': ground_truth_df.loc['RSF', 'GroundTruth'],
        'Coverage %': ground_truth_df.loc['Coverage %', 'GroundTruth'],
        'Nº Droplets': ground_truth_df.loc['Nº Droplets', 'GroundTruth'],
    }

    prediction_values = {
        'VMD': predictions['vmd'],
        'RSF': predictions['rsf'],
        'Coverage %': predictions['coverage_percentage'],
        'Nº Droplets': predictions['no_droplets'],
    }

    errors = {key: abs(float(prediction_values[key]) - float(ground_truth[key])) for key in ground_truth}


    
    for key, error in errors.items():
        print(f"Error in {key}: {error}")

    # Optionally, you can calculate Mean Absolute Error (MAE) if you want an aggregated error metric
    mae = sum(errors.values()) / len(errors)
    print(f"Mean Absolute Error: {mae}")