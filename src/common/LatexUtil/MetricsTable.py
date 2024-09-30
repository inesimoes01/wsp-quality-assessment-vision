import pandas as pd

# Read the CSV file
csv_file = 'results\\evaluation\\droplet\\general\\droplet_real_segmentation_general.csv'  # Update this path to your actual CSV file location
tex_file = "results\\latex\\metrics_table.tex"
data = pd.read_csv(csv_file)

# Map dataset abbreviations to full names for proper matching
def get_data(method, dataset):
    dataset_name = f'droplet_{dataset}_dataset_{method.lower()}'
    row = data[data['method'] == dataset_name]
    return row[['precision', 'recall', 'f1-score', 'map50', 'map50-95', 'segmentation_time']].values[0]

# Updated LaTeX table template
latex_template = """
\\begin{{table}}[h!]
\\small
    \\centering
    \\caption{{Results of algorithms for calculating droplet segmentation metrics on water-sensitive paper.}}
    \\label{{tab:segmentation-metrics-drop}}
    \\begin{{tabular}}{{cccccccc}}
        \\hline
        \\textbf{{Method}} & \\textbf{{Dataset}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1 Score}} & \\textbf{{mAP50}} & \\textbf{{mAP50-95}} & \\textbf{{Time (ms)}}\\\\
        \\hline
        \\multirow{{4}}{{2cm}}[0em]{{YOLOv8}} & RDS & {yolo_rds_p} & {yolo_rds_r} & {yolo_rds_f1} & {yolo_rds_map50} & {yolo_rds_map5095} & {yolo_rds_time}\\\\
        & RDF & {yolo_rdf_p} & {yolo_rdf_r} & {yolo_rdf_f1} & {yolo_rdf_map50} & {yolo_rdf_map5095} & {yolo_rdf_time}\\\\
        & SDS & {yolo_sds_p} & {yolo_sds_r} & {yolo_sds_f1} & {yolo_sds_map50} & {yolo_sds_map5095} & {yolo_sds_time} \\\\
        & SDF & {yolo_sdf_p} & {yolo_sdf_r} & {yolo_sdf_f1} & {yolo_sdf_map50} & {yolo_sdf_map5095} & {yolo_sdf_time} \\\\

        \\vspace{{0.2cm}}\\\\
        
        \\multirow{{4}}{{2cm}}[0em]{{MRCNN}} & RDS & {mrcnn_rds_p} & {mrcnn_rds_r} & {mrcnn_rds_f1} & {mrcnn_rds_map50} & {mrcnn_rds_map5095} & {mrcnn_rds_time}\\\\
        & RDF & {mrcnn_rdf_p} & {mrcnn_rdf_r} & {mrcnn_rdf_f1} & {mrcnn_rdf_map50} & {mrcnn_rdf_map5095} & {mrcnn_rdf_time}\\\\
        & SDS & {mrcnn_sds_p} & {mrcnn_sds_r} & {mrcnn_sds_f1} & {mrcnn_sds_map50} & {mrcnn_sds_map5095} & {mrcnn_sds_time}\\\\
        & SDF & {mrcnn_sdf_p} & {mrcnn_sdf_r} & {mrcnn_sdf_f1} & {mrcnn_sdf_map50} & {mrcnn_sdf_map5095} & {mrcnn_sdf_time}\\\\

        \\vspace{{0.2cm}}\\\\
          
        \\multirow{{4}}{{2cm}}[0em]{{CellPose}} & RDS & {cellpose_rds_p} & {cellpose_rds_r} & {cellpose_rds_f1} & {cellpose_rds_map50} & {cellpose_rds_map5095} & {cellpose_rds_time}\\\\
        & RDF & {cellpose_rdf_p} & {cellpose_rdf_r} & {cellpose_rdf_f1} & {cellpose_rdf_map50} & {cellpose_rdf_map5095} & {cellpose_rdf_time}\\\\
        & SDS & {cellpose_sds_p} & {cellpose_sds_r} & {cellpose_sds_f1} & {cellpose_sds_map50} & {cellpose_sds_map5095} & {cellpose_sds_time}\\\\
        & SDF & {cellpose_sdf_p} & {cellpose_sdf_r} & {cellpose_sdf_f1} & {cellpose_sdf_map50} & {cellpose_sdf_map5095} & {cellpose_sdf_time}\\\\

        \\vspace{{0.2cm}}\\\\
          
        \\multirow{{4}}{{2cm}}[0em]{{CCV}} & RDS & {ccv_rds_p} & {ccv_rds_r} & {ccv_rds_f1} & {ccv_rds_map50} & {ccv_rds_map5095} & {ccv_rds_time}\\\\
        & RDF & {ccv_rdf_p} & {ccv_rdf_r} & {ccv_rdf_f1} & {ccv_rdf_map50} & {ccv_rdf_map5095} & {ccv_rdf_time}\\\\
        & SDS & {ccv_sds_p} & {ccv_sds_r} & {ccv_sds_f1} & {ccv_sds_map50} & {ccv_sds_map5095} & {ccv_sds_time}\\\\
        & SDF & {ccv_sdf_p} & {ccv_sdf_r} & {ccv_sdf_f1} & {ccv_sdf_map50} & {ccv_sdf_map5095} & {ccv_sdf_time}\\\\

        \\hline
    \\end{{tabular}}
\\end{{table}}
"""

# Datasets to extract (real dataset square - RDS, real dataset full - RDF, synthetic dataset square - SDS, synthetic dataset full - SDF)
datasets = ['real_square', 'real_full', 'synthetic_square', 'synthetic_full']

# Extract values for each method and dataset
yolo_rds = get_data('yolo', 'real_square')
yolo_rdf = get_data('yolo', 'real_full')
yolo_sds = get_data('yolo', 'synthetic_square')
yolo_sdf = get_data('yolo', 'synthetic_full')

mrcnn_rds = get_data('mrcnn', 'real_square')
mrcnn_rdf = get_data('mrcnn', 'real_full')
mrcnn_sds = get_data('mrcnn', 'synthetic_square')
mrcnn_sdf = get_data('mrcnn', 'synthetic_full')

cellpose_rds = get_data('cellpose', 'real_square')
cellpose_rdf = get_data('cellpose', 'real_full')
cellpose_sds = get_data('cellpose', 'synthetic_square')
cellpose_sdf = get_data('cellpose', 'synthetic_full')

ccv_rds = get_data('ccv', 'real_square')
ccv_rdf = get_data('ccv', 'real_full')
ccv_sds = get_data('ccv', 'synthetic_square')
ccv_sdf = get_data('ccv', 'synthetic_full')

# Fill in the LaTeX template with the extracted data

filled_latex = latex_template.format(
    yolo_rds_p=round(yolo_rds[0], 4), yolo_rds_r=round(yolo_rds[1], 4), yolo_rds_f1=round(yolo_rds[2], 4), yolo_rds_map50=round(yolo_rds[3], 4), yolo_rds_map5095=round(yolo_rds[4], 4), yolo_rds_time=round(yolo_rds[5], 4),
    yolo_rdf_p=round(yolo_rdf[0], 4), yolo_rdf_r=round(yolo_rdf[1], 4), yolo_rdf_f1=round(yolo_rdf[2], 4), yolo_rdf_map50=round(yolo_rdf[3], 4), yolo_rdf_map5095=round(yolo_rdf[4], 4), yolo_rdf_time=round(yolo_rdf[5], 4),
    yolo_sds_p=round(yolo_sds[0], 4), yolo_sds_r=round(yolo_sds[1], 4), yolo_sds_f1=round(yolo_sds[2], 4), yolo_sds_map50=round(yolo_sds[3], 4), yolo_sds_map5095=round(yolo_sds[4], 4), yolo_sds_time=round(yolo_sds[5], 4),
    yolo_sdf_p=round(yolo_sdf[0], 4), yolo_sdf_r=round(yolo_sdf[1], 4), yolo_sdf_f1=round(yolo_sdf[2], 4), yolo_sdf_map50=round(yolo_sdf[3], 4), yolo_sdf_map5095=round(yolo_sdf[4], 4), yolo_sdf_time=round(yolo_sdf[5], 4),
    
    mrcnn_rds_p=round(mrcnn_rds[0], 4), mrcnn_rds_r=round(mrcnn_rds[1], 4), mrcnn_rds_f1=round(mrcnn_rds[2], 4), mrcnn_rds_map50=round(mrcnn_rds[3], 4), mrcnn_rds_map5095=round(mrcnn_rds[4], 4), mrcnn_rds_time=round(mrcnn_rds[5], 4),
    mrcnn_rdf_p=round(mrcnn_rdf[0], 4), mrcnn_rdf_r=round(mrcnn_rdf[1], 4), mrcnn_rdf_f1=round(mrcnn_rdf[2], 4), mrcnn_rdf_map50=round(mrcnn_rdf[3], 4), mrcnn_rdf_map5095=round(mrcnn_rdf[4], 4), mrcnn_rdf_time=round(mrcnn_rdf[5], 4),
    mrcnn_sds_p=round(mrcnn_sds[0], 4), mrcnn_sds_r=round(mrcnn_sds[1], 4), mrcnn_sds_f1=round(mrcnn_sds[2], 4), mrcnn_sds_map50=round(mrcnn_sds[3], 4), mrcnn_sds_map5095=round(mrcnn_sds[4], 4), mrcnn_sds_time=round(mrcnn_sds[5], 4),
    mrcnn_sdf_p=round(mrcnn_sdf[0], 4), mrcnn_sdf_r=round(mrcnn_sdf[1], 4), mrcnn_sdf_f1=round(mrcnn_sdf[2], 4), mrcnn_sdf_map50=round(mrcnn_sdf[3], 4), mrcnn_sdf_map5095=round(mrcnn_sdf[4], 4), mrcnn_sdf_time=round(mrcnn_sdf[5], 4),
    
    cellpose_rds_p=round(cellpose_rds[0], 4), cellpose_rds_r=round(cellpose_rds[1], 4), cellpose_rds_f1=round(cellpose_rds[2], 4), cellpose_rds_map50=round(cellpose_rds[3], 4), cellpose_rds_map5095=round(cellpose_rds[4], 4), cellpose_rds_time=round(cellpose_rds[5], 4),
    cellpose_rdf_p=round(cellpose_rdf[0], 4), cellpose_rdf_r=round(cellpose_rdf[1], 4), cellpose_rdf_f1=round(cellpose_rdf[2], 4), cellpose_rdf_map50=round(cellpose_rdf[3], 4), cellpose_rdf_map5095=round(cellpose_rdf[4], 4), cellpose_rdf_time=round(cellpose_rdf[5], 4),
    cellpose_sds_p=round(cellpose_sds[0], 4), cellpose_sds_r=round(cellpose_sds[1], 4), cellpose_sds_f1=round(cellpose_sds[2], 4), cellpose_sds_map50=round(cellpose_sds[3], 4), cellpose_sds_map5095=round(cellpose_sds[4], 4), cellpose_sds_time=round(cellpose_sds[5], 4),
    cellpose_sdf_p=round(cellpose_sdf[0], 4), cellpose_sdf_r=round(cellpose_sdf[1], 4), cellpose_sdf_f1=round(cellpose_sdf[2], 4), cellpose_sdf_map50=round(cellpose_sdf[3], 4), cellpose_sdf_map5095=round(cellpose_sdf[4], 4), cellpose_sdf_time=round(cellpose_sdf[5], 4),
    
    ccv_rds_p=round(ccv_rds[0], 4), ccv_rds_r=round(ccv_rds[1], 4), ccv_rds_f1=round(ccv_rds[2], 4), ccv_rds_map50=round(ccv_rds[3], 4), ccv_rds_map5095=round(ccv_rds[4], 4), ccv_rds_time=round(ccv_rds[5], 4),
    ccv_rdf_p=round(ccv_rdf[0], 4), ccv_rdf_r=round(ccv_rdf[1], 4), ccv_rdf_f1=round(ccv_rdf[2], 4), ccv_rdf_map50=round(ccv_rdf[3], 4), ccv_rdf_map5095=round(ccv_rdf[4], 4), ccv_rdf_time=round(ccv_rdf[5], 4),
    ccv_sds_p=round(ccv_sds[0], 4), ccv_sds_r=round(ccv_sds[1], 4), ccv_sds_f1=round(ccv_sds[2], 4), ccv_sds_map50=round(ccv_sds[3], 4), ccv_sds_map5095=round(ccv_sds[4], 4), ccv_sds_time=round(ccv_sds[5], 4),
    ccv_sdf_p=round(ccv_sdf[0], 4), ccv_sdf_r=round(ccv_sdf[1], 4), ccv_sdf_f1=round(ccv_sdf[2], 4), ccv_sdf_map50=round(ccv_sdf[3], 4), ccv_sdf_map5095=round(ccv_sdf[4], 4), ccv_sdf_time=round(ccv_sdf[5], 4))

with open(tex_file, 'w') as f:
    f.write(filled_latex)