import pandas as pd
import os

# Specify the folder containing the CSV files
folder_path = 'data\\droplets\\synthetic_dataset_normal_droplets\\raw\\statistics'  # Replace with the path to your folder

# Loop through all the files in the specified folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Ensure the columns exist in the DataFrame
        if 'Overlapped Droplets %' in df.columns and 'Number of overlapped droplets' in df.columns:
            # Swap the values of 'Overlapped Droplets %' and 'Number of overlapped droplets'
            temp = df['Overlapped Droplets %'].copy()
            df['Overlapped Droplets %'] = df['Number of overlapped droplets']
            df['Number of overlapped droplets'] = temp

            # Save the modified DataFrame back to the CSV file
            df.to_csv(file_path, index=False)
            print(f"Processed file: {filename}")
        else:
            print(f"Columns not found in file: {filename}")
