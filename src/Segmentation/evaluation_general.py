import csv

filename = 'results\\metrics\\general_avg_values.csv'

# Define the column headers
columns = [
    'method', 'precision', 'recall', 'f1-score', 'map0.5', 'map0.5-0.95',
    'tp', 'fp', 'fn', 'segmentation_time', 'iou_mask'
]

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the column headers
    writer.writerow(columns)
    