import pandas as pd
import os

# Specify the folder containing the CSV files
folder_path = 'data\\droplets\\synthetic_dataset_droplets\\full\\divided\\test\\statistics'  # Replace with the path to your folder

# Loop through all the files in the specified folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file into a DataFrame with correct format
        df = pd.read_csv(file_path, header=None, names=["Metric", "Value"])

        # Ensure the required rows exist in the DataFrame
        if 'Overlapped Droplets %' in df['Metric'].values and 'Number of overlapped droplets' in df['Metric'].values:
            # Locate the indices of the rows to be swapped
            overlapped_idx = df[df['Metric'] == 'Overlapped Droplets %'].index[0]
            number_overlapped_idx = df[df['Metric'] == 'Number of overlapped droplets'].index[0]

            # Swap the values between 'Overlapped Droplets %' and 'Number of overlapped droplets'
            temp = df.at[overlapped_idx, 'Value']
            df.at[overlapped_idx, 'Value'] = df.at[number_overlapped_idx, 'Value']
            df.at[number_overlapped_idx, 'Value'] = temp

            # Save the modified DataFrame back to the CSV file
            df.to_csv(file_path, index=False, header=False)
            print(f"Processed file: {filename}")
        else:
            print(f"Required rows not found in file: {filename}")
