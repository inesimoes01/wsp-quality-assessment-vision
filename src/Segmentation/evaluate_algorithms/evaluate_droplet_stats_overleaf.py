import pandas as pd
import os
import sys
import csv

sys.path.insert(0, 'src')
import Common.config as config 

results_files = "results\\evaluation\\droplet\\dropleaf_statistics.csv"

def write_stats_csv(filename, coverage_percentage, vmd, rsf, no_droplets, coverage_percentage_gt, vmd_gt, rsf_gt, no_droplets_gt):
    
    with open(results_files, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file", "VMD_pred", "VMD_gt", "VMD_error","VMD_abs_error",
                                                  "RSF_pred", "RSF_gt", "RSF_error", "RSF_abs_error", 
                                                  "CoveragePercentage_pred", "CoveragePercentage_gt", "CoveragePercentage_error", "CoveragePercentage_abs_error", 
                                                  "NoDroplets_pred", "NoDroplets_gt", "NoDroplets_error","NoDroplets_abs_error",
                                                  "OtherCoveragePercentage_pred", "OtherCoveragePercentage_gt", "OtherCoveragePercentage_error",  "OtherCoveragePercentage_abs_error"])
        
        new_row = {
            "file": filename, 
            "VMD_pred": vmd, "VMD_gt": vmd_gt, "VMD_error": abs((vmd - vmd_gt) / vmd_gt), "VMD_abs_error": abs((vmd - vmd_gt)), 
            "RSF_pred": rsf, "RSF_gt": rsf_gt, "RSF_error": abs((rsf - rsf_gt) / rsf_gt), "RSF_abs_error": abs((rsf - rsf_gt)), 
            "CoveragePercentage_pred": coverage_percentage, "CoveragePercentage_gt": coverage_percentage_gt, "CoveragePercentage_error": abs((coverage_percentage - coverage_percentage_gt) / coverage_percentage_gt),"CoveragePercentage_abs_error": abs((coverage_percentage - coverage_percentage_gt)), 
            "NoDroplets_pred": no_droplets, "NoDroplets_gt": no_droplets_gt, "NoDroplets_error": abs((no_droplets - no_droplets_gt) / no_droplets_gt), "NoDroplets_abs_error": abs((no_droplets - no_droplets_gt)),
            "OtherCoveragePercentage_pred": 100 - coverage_percentage, "OtherCoveragePercentage_gt": coverage_percentage_gt, "OtherCoveragePercentage_error":abs((100-coverage_percentage - coverage_percentage_gt) / coverage_percentage_gt), "OtherCoveragePercentage_abs_error":abs((100-coverage_percentage - coverage_percentage_gt)),
            }
        writer.writerow(new_row)

predictions_df = pd.read_csv("results\\metrics\\droplet\\real_dataset\\dropleaf_stats.csv")

id_list = predictions_df['id'].tolist() 

with open(results_files, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["file", "VMD_pred", "VMD_gt", "VMD_error","VMD_abs_error",
                                                  "RSF_pred", "RSF_gt", "RSF_error", "RSF_abs_error", 
                                                  "CoveragePercentage_pred", "CoveragePercentage_gt", "CoveragePercentage_error", "CoveragePercentage_abs_error", 
                                                  "NoDroplets_pred", "NoDroplets_gt", "NoDroplets_error","NoDroplets_abs_error",
                                                  "OtherCoveragePercentage_pred", "OtherCoveragePercentage_gt", "OtherCoveragePercentage_error",  "OtherCoveragePercentage_abs_error"])
    writer.writeheader()

for id in id_list:

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

    write_stats_csv(id, float(predictions['coverage_percentage']), float(predictions['vmd']), float(predictions['rsf']), float(predictions['no_droplets']),
                    ground_truth_df.loc['Coverage %', 'GroundTruth'], ground_truth_df.loc['VMD', 'GroundTruth'], ground_truth_df.loc['RSF', 'GroundTruth'], ground_truth_df.loc['Nº Droplets', 'GroundTruth']
                    )
    
    for key, error in errors.items():
        print(f"Error in {key}: {error}")


df = pd.read_csv(results_files)

average_df = pd.DataFrame([{
    'method': 'dropleaf',
    'VMD_error': df['VMD_error'].mean(),
    'RSF_error': df['RSF_error'].mean(),
    'CoveragePercentage_error': df['CoveragePercentage_error'].mean(),
    'NoDroplets_error': df['NoDroplets_error'].mean(),
    'OtherCoveragePercentage_error': df['OtherCoveragePercentage_error'].mean(),
   
    'VMD_median': df['VMD_error'].median(),
    'RSF_median': df['RSF_error'].median(),
    'CoveragePercentage_median': df['CoveragePercentage_error'].median(),
    'NoDroplets_median': df['NoDroplets_error'].median(),
    'OtherCoveragePercentage_median': df['OtherCoveragePercentage_error'].median(),
    
    'VMD_std': df['VMD_error'].std(),
    'RSF_std': df['RSF_error'].std(),
    'CoveragePercentage_std': df['CoveragePercentage_error'].std(),
    'NoDroplets_std': df['NoDroplets_error'].std(),
    'OtherCoveragePercentage_std': df['OtherCoveragePercentage_error'].std(),

    'VMD_max': df['VMD_error'].max(),
    'RSF_max': df['RSF_error'].max(),
    'CoveragePercentage_max': df['CoveragePercentage_error'].max(),
    'NoDroplets_max': df['NoDroplets_error'].max(),
    'OtherCoveragePercentage_max': df['OtherCoveragePercentage_error'].max(),
}])

general_file = "results\\metrics\\general_avg_values_stats.csv"
df_gen = pd.read_csv(general_file)

df_gen = df_gen._append(average_df, ignore_index=True)
df_gen.to_csv(general_file, index=False)
