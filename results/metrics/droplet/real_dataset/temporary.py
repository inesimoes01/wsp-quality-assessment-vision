import pandas as pd

# Load the CSV file
df = pd.read_csv('your_file.csv')

# Substitute VMD_error with VMD_gt - VMD_pred
df['VMD_error'] = df['VMD_gt'] - df['VMD_pred']

# Substitute VMD_abs_error with the absolute value of VMD_error
df['VMD_abs_error'] = df['VMD_error'].abs()

# Substitute RSF_error with RSF_gt - RSF_pred
df['RSF_error'] = df['RSF_gt'] - df['RSF_pred']

# Substitute RSF_abs_error with the absolute value of RSF_error
df['RSF_abs_error'] = df['RSF_error'].abs()

# Substitute CoveragePercentage_error with CoveragePercentage_gt - CoveragePercentage_pred
df['CoveragePercentage_error'] = df['CoveragePercentage_gt'] - df['CoveragePercentage_pred']

# Substitute CoveragePercentage_abs_error with the absolute value of CoveragePercentage_error
df['CoveragePercentage_abs_error'] = df['CoveragePercentage_error'].abs()

# Substitute NoDroplets_error with NoDroplets_gt - NoDroplets_pred (if not null)
if 'NoDroplets_gt' in df.columns and 'NoDroplets_pred' in df.columns:
    df['NoDroplets_error'] = df['NoDroplets_gt'] - df['NoDroplets_pred']
    df['NoDroplets_abs_error'] = df['NoDroplets_error'].abs()

# Substitute OtherCoveragePercentage_error with OtherCoveragePercentage_gt - OtherCoveragePercentage_pred (if not null)
if 'OtherCoveragePercentage_gt' in df.columns and 'OtherCoveragePercentage_pred' in df.columns:
    df['OtherCoveragePercentage_error'] = df['OtherCoveragePercentage_gt'] - df['OtherCoveragePercentage_pred']
    df['OtherCoveragePercentage_abs_error'] = df['OtherCoveragePercentage_error'].abs()

# Save the updated dataframe to a new CSV file
df.to_csv('updated_file.csv', index=False)

print("Errors have been substituted with actual values and the file has been saved as 'updated_file.csv'.")
