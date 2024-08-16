import csv
import pandas as pd


general_file = "results\\metrics\\general_avg_values_stats.csv"
columns = ["method", "VMD_error", 
                    "RSF_error", 
                    "CoveragePercentage_error", 
                    "NoDroplets_error",
                    "OtherCoveragePercentage_error",
                    
                    "VMD_median", 
                    "RSF_median", 
                    "CoveragePercentage_median", 
                    "NoDroplets_median",
                    "OtherCoveragePercentage_median",

                    "VMD_std", 
                    "RSF_std", 
                    "CoveragePercentage_std", 
                    "NoDroplets_std",
                    "OtherCoveragePercentage_std",
                    
                    "VMD_max", 
                    "RSF_max", 
                    "CoveragePercentage_max", 
                    "NoDroplets_max",
                    "OtherCoveragePercentage_max",
                    ]

def new_file():
    with open(general_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["method", "VMD_error", 
                                                    "RSF_error", 
                                                    "CoveragePercentage_error", 
                                                    "NoDroplets_error",
                                                    "OtherCoveragePercentage_error",
                                                    
                                                    "VMD_median", 
                                                    "RSF_median", 
                                                    "CoveragePercentage_median", 
                                                    "NoDroplets_median",
                                                    "OtherCoveragePercentage_median",

                                                    "VMD_std", 
                                                    "RSF_std", 
                                                    "CoveragePercentage_std", 
                                                    "NoDroplets_std",
                                                    "OtherCoveragePercentage_std",
                                                    
                                                    "VMD_max", 
                                                    "RSF_max", 
                                                    "CoveragePercentage_max", 
                                                    "NoDroplets_max",
                                                    "OtherCoveragePercentage_max",
                                                    ])
        writer.writeheader()

def update_results(results_files, method):
    df = pd.read_csv(results_files)

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


    df_gen = pd.read_csv(general_file)
    df_gen = df_gen._append(average_df, ignore_index=True)
    df_gen.to_csv(general_file, index=False)


update_results("results\\metrics\\droplet\\real_dataset\\droplet_stats_real_dataset_cv.csv", "real_dataset_cv")