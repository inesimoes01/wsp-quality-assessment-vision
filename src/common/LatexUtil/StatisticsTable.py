import pandas as pd

def fill_table(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Define a mapping from CSV method names to LaTeX method and dataset names
    method_mapping = {
        "droplet_synthetic_square_dataset_yolo": ("YOLOv8", "SDS"),
        "droplet_real_square_dataset_yolo": ("YOLOv8", "RDS"),
        "droplet_synthetic_full_dataset_yolo": ("YOLOv8", "SDF"),
        "droplet_real_full_dataset_yolo": ("YOLOv8", "RDF"),
        
        "droplet_synthetic_square_dataset_mrcnn": ("MRCNN", "SDS"),
        "droplet_real_square_dataset_mrcnn": ("MRCNN", "RDS"),
        "droplet_synthetic_full_dataset_mrcnn": ("MRCNN", "SDF"),
        "droplet_real_full_dataset_mrcnn": ("MRCNN", "RDF"),
        
        "droplet_synthetic_square_dataset_cellpose": ("Cellpose", "SDS"),
        "droplet_real_square_dataset_cellpose": ("Cellpose", "RDS"),
        "droplet_synthetic_full_dataset_cellpose": ("Cellpose", "SDF"),
        "droplet_real_full_dataset_cellpose": ("Cellpose", "RDF"),
        
        "droplet_synthetic_square_dataset_ccv": ("CCV", "SDS"),
        "droplet_real_square_dataset_ccv": ("CCV", "RDS"),
        "droplet_synthetic_full_dataset_ccv": ("CCV", "SDF"),
        "droplet_real_full_dataset_ccv": ("CCV", "RDF"),
        
        # "droplet_synthetic_square_dataset_dropleaf": ("DropLeaf", "SDS"),
        # "droplet_real_square_dataset_dropleaf": ("DropLeaf", "RDS"),
        # "droplet_synthetic_full_dataset_dropleaf": ("DropLeaf", "SDF"),
        # "droplet_real_full_dataset_dropleaf": ("DropLeaf", "RDF"),
    }

    # Start constructing the LaTeX table
    table_latex = """
\\begin{table}[h!]
    \\centering
    \\caption{Relative error of each one of the metrics used to evaluate the spray quality of a water-sensitive paper.}
    \\label{tab:my_label}
    \\begin{tabular}{c c c c c c}
        \\hline
        \\textbf{Method} & \\textbf{Dataset} & \\textbf{No Droplets} & \\textbf{VMD} & \\textbf{RSF} &  \\textbf{Coverage Percentage } \\\\
        \\hline
    """

    # Iterate over the method mappings and fill in the values from the CSV
    for key, (method, dataset) in method_mapping.items():
        # Find the corresponding row in the DataFrame
        row = df[df['method'] == key].iloc[0]
        
        # Extract values for the table
        no_droplets_error = row['NoDroplets_error']
        vmd_error = row['VMD_error']
        rsf_error = row['RSF_error']
        coverage_percentage_error = row['CoveragePercentage_error']

        # Add a row to the LaTeX table
        table_latex += f"        \\multirow{{2}}{{2cm}}[0em]{{\\textbf{{{method}}}}} & {dataset} & {no_droplets_error:.4f} & {vmd_error:.4f} & {rsf_error:.4f} & {coverage_percentage_error:.4f} \\\\\n"
    
    # Finish the table
    table_latex += """
        \\hline
    \\end{tabular}
\\end{table}
"""

    return table_latex

# Example usage
csv_file = 'results\\evaluation\\droplet\\general\\droplet_real_statistics_general.csv'  # Path to your CSV file
latex_code = fill_table(csv_file)
print(latex_code)
