import pandas as pd

metrics_table_path = "results\\latex\\metrics_table.tex"
csv_file = "results\\evaluation\\droplet\\general\\droplet_real_segmentation_general.csv"
data = pd.read_csv(csv_file)

# Filter the data based on the dataset
def get_data(method, dataset):
    if dataset == 'RD':
        dataset_name = 'droplet_real_dataset_' + method.lower()
    else:
        dataset_name = 'droplet_synthetic_dataset_' + method.lower()
    row = data[data['method'] == dataset_name]
    return row[['precision', 'recall', 'f1-score', 'map50', 'segmentation_time']].values[0]

# LaTeX table template
latex_template = """
\\begin{{table}}[h!]
    \\centering
    \\caption{{Results of algorithms for calculating droplet segmentation metrics on water-sensitive paper.}}
    \\label{{tab:segmentation-metrics-drop}}
    \\begin{{tabular}}{{ccccccc }}
        \\hline
        \\textbf{{Method}} & \\textbf{{Dataset}} & \\textbf{{P}} & \\textbf{{R}} & \\textbf{{F1}} & \\textbf{{mAP50}} & \\textbf{{Time (ms)}}\\\\
        \\hline
        \\multirow{{2}}{{2cm}}[0em]{{YOLOv8}} & RD & {yolo_rd_p} & {yolo_rd_r} & {yolo_rd_f1} & {yolo_rd_map50} & {yolo_rd_time}\\\\
         & SD & {yolo_sd_p} & {yolo_sd_r} & {yolo_sd_f1} & {yolo_sd_map50} & {yolo_sd_time} \\\\
        
        \\multirow{{2}}{{2cm}}[0em]{{MRCNN}} & RD & {mrcnn_rd_p} & {mrcnn_rd_r} & {mrcnn_rd_f1} & {mrcnn_rd_map50} & {mrcnn_rd_time}\\\\
         & SD & {mrcnn_sd_p} & {mrcnn_sd_r} & {mrcnn_sd_f1} & {mrcnn_sd_map50} & {mrcnn_sd_time}\\\\
       
        \\multirow{{2}}{{2cm}}[0em]{{CCV}} & RD & {ccv_rd_p} & {ccv_rd_r} & {ccv_rd_f1} & {ccv_rd_map50} & {ccv_rd_time}\\\\
         & SD & {ccv_sd_p} & {ccv_sd_r} & {ccv_sd_f1} & {ccv_sd_map50} & {ccv_sd_time}\\\\
        \\hline
    \\end{{tabular}}
\\end{{table}}
"""

# Extract values for each method and dataset
yolo_rd = get_data('yolo', 'RD')
yolo_sd = get_data('yolo', 'SD')
mrcnn_rd = get_data('mrcnn', 'RD')
mrcnn_sd = get_data('mrcnn', 'SD')
ccv_rd = get_data('ccv', 'RD')
ccv_sd = get_data('ccv', 'SD')

# Fill in the LaTeX template with the extracted data
filled_latex = latex_template.format(
    yolo_rd_p=round(yolo_rd[0], 4), yolo_rd_r=round(yolo_rd[1], 4), yolo_rd_f1=round(yolo_rd[2], 4), yolo_rd_map50=round(yolo_rd[3], 4), yolo_rd_time=round(yolo_rd[4], 4),
    yolo_sd_p=round(yolo_sd[0], 4), yolo_sd_r=round(yolo_sd[1], 4), yolo_sd_f1=round(yolo_sd[2], 4), yolo_sd_map50=round(yolo_sd[3], 4), yolo_sd_time=round(yolo_sd[4], 4),
    mrcnn_rd_p=round(mrcnn_rd[0], 4), mrcnn_rd_r=round(mrcnn_rd[1], 4), mrcnn_rd_f1=round(mrcnn_rd[2], 4), mrcnn_rd_map50=round(mrcnn_rd[3], 4), mrcnn_rd_time=round(mrcnn_rd[4], 4),
    mrcnn_sd_p=round(mrcnn_sd[0], 4), mrcnn_sd_r=round(mrcnn_sd[1], 4), mrcnn_sd_f1=round(mrcnn_sd[2], 4), mrcnn_sd_map50=round(mrcnn_sd[3], 4), mrcnn_sd_time=round(mrcnn_sd[4], 4),
    ccv_rd_p=round(ccv_rd[0], 4), ccv_rd_r=round(ccv_rd[1], 4), ccv_rd_f1=round(ccv_rd[2], 4), ccv_rd_map50=round(ccv_rd[3], 4), ccv_rd_time=round(ccv_rd[4], 4),
    ccv_sd_p=round(ccv_sd[0], 4), ccv_sd_r=round(ccv_sd[1], 4), ccv_sd_f1=round(ccv_sd[2], 4), ccv_sd_map50=round(ccv_sd[3], 4), ccv_sd_time=round(ccv_sd[4], 4)
)

# Write the filled LaTeX table to a file
with open(metrics_table_path, 'w') as f:
    f.write(filled_latex)

print("LaTeX table has been generated and saved to 'filled_latex_table.tex'")
