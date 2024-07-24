import pandas as pd

# structure of the results in a DataFrame
columns = ['Algorithm', 'SSD_P', 'SSD_R', 'SSD_F1', 'SD_P', 'SD_R', 'SD_F1', 'Real_P', 'Real_R', 'Real_F1']
data = {
    'Algorithm': ['CV1', 'CV2', 'YOLO', 'UNET', 'Automatic Annotation', 'APP1', 'APP2'],
    'SSD_P': ['x']*7, 'SSD_R': ['x']*7, 'SSD_F1': ['x']*7,
    'SD_P': ['y']*7, 'SD_R': ['y']*7, 'SD_F1': ['y']*7,
    'Real_P': ['z']*7, 'Real_R': ['z']*7, 'Real_F1': ['z']*7
}

results_df = pd.DataFrame(data, columns=columns)

# Example function to update results (replace with actual algorithm result updates)
def update_results(algorithm, ssd_p, ssd_r, ssd_f1, sd_p, sd_r, sd_f1, real_p, real_r, real_f1):
    results_df.loc[results_df['Algorithm'] == algorithm, ['SSD_P', 'SSD_R', 'SSD_F1', 'SD_P', 'SD_R', 'SD_F1', 'Real_P', 'Real_R', 'Real_F1']] = [
        ssd_p, ssd_r, ssd_f1, sd_p, sd_r, sd_f1, real_p, real_r, real_f1
    ]

# # Update results for CV1 as an example (replace with actual values)
# update_results('CV1', 0.9, 0.8, 0.85, 0.75, 0.7, 0.72, 0.65, 0.6, 0.62)

# Generate the LaTeX table
def generate_latex_table(df):
    latex_str = '''
    \\begin{table}[h!]
        \\centering
        \\caption{Results}
        \\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}
        \\hline
            \\multirow{2}{*}{\\diagbox{Algorithm}{Dataset}} & \\multicolumn{3}{|c|}{SSD} & \\multicolumn{3}{|c|}{SD} & \\multicolumn{3}{|c|}{Real Dataset}\\\\
           
            & P & R & F1 & P & R & F1 & P & R & F1  \\\\
            \\hline \\hline
    '''
    for _, row in df.iterrows():
        latex_str += f"        {row['Algorithm']} & {row['SSD_P']} & {row['SSD_R']} & {row['SSD_F1']} & {row['SD_P']} & {row['SD_R']} & {row['SD_F1']} & {row['Real_P']} & {row['Real_R']} & {row['Real_F1']}\\\\\n"
        latex_str += '         \\hline\n'
    latex_str += '''
        \\end{tabular}
       
        \\label{tab:minDPI}
    \\end{table}
    '''
    return latex_str

def update_table(algorithm, ssd_p, ssd_r, ssd_f1, sd_p, sd_r, sd_f1, real_p, real_r, real_f1):
    
    update_results(algorithm, ssd_p, ssd_r, ssd_f1, sd_p, sd_r, sd_f1, real_p, real_r, real_f1)
    latex_table = generate_latex_table(results_df)
    with open('src\\Common\\results_table.tex', 'w') as f:
        f.write(latex_table)

    print("LaTeX table updated and written to results_table.tex")
