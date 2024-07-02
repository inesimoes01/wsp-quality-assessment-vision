import numpy as np
import sys
import config
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import csv

def generate_droplet_sizes_rosin_rammler(num_spots):
    x_o = config.CHARACTERISTIC_PARTICLE_SIZE
    n = config.UNIFORMITY_CONSTANT

    # list of size no_droplets with random numbers from 0 to 1
    random_numbers = np.random.rand(num_spots)

    # inverse transform sampling to generate droplet sizes
    droplet_sizes = x_o * (-np.log(1 - random_numbers))**(1/n)
    
    return droplet_sizes 

def rosin_rammler(d, d0, n):
    return 1 - np.exp(-(d / d0) ** n)

def fit_droplet_size_graph(droplet_sizes):
    sorted_sizes, cumulative_fractions = calculate_cumulative_fractions(droplet_sizes)

    popt, pcov = curve_fit(rosin_rammler, sorted_sizes, cumulative_fractions, p0=[5, 1])
    d0, n = popt

    # Generate fitted curve
    d_fit = np.linspace(min(droplet_sizes), max(droplet_sizes), 100)
    f_fit = rosin_rammler(d_fit, d0, n)

    with open(os.path.join(config.RESULTS_LATEX, 'droplet_data.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['DropletSize', 'CumulativeFraction'])
        writer.writerows(zip(sorted_sizes, cumulative_fractions))

    with open(os.path.join(config.RESULTS_LATEX, 'fitted_curve.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['DropletSize', 'FittedFraction'])
        writer.writerows(zip(d_fit, f_fit))

    # generate LaTeX code for the plot
    latex_template = """
    \\begin{{figure}}
        \\centering
        \\begin{{tikzpicture}}
            \\begin{{axis}}[
                width=10cm,
                height=6cm,
                xlabel={{Droplet Size}},
                ylabel={{Cumulative Fraction}},
                grid=major,
                legend pos=south east,
                title={{Rosin-Rammler Fit to Droplet Size Data}}
            ]

            % Plot the fitted curve
            \\addplot[
                thick,
                blue,
            ] table [x=DropletSize, y=FittedFraction, col sep=comma] {{csv_plots/fitted_curve.csv}};
            \\addlegendentry{{Fit: $d_0={:.2f}$, $n={:.2f}$}}

            % Plot the data points
            \\addplot[
                only marks,
                red,
                mark=*,
            ] table [x=DropletSize, y=CumulativeFraction, col sep=comma] {{csv_plots/droplet_data.csv}};
            \\addlegendentry{{Data}}

            \\end{{axis}}
        \\end{{tikzpicture}}
        \\caption{{Rosin-Rammler Fit to Droplet Size Data}}
    \\end{{figure}}
    """

    # Format the LaTeX template with actual values of d0 and n
    latex_code = latex_template.format(d0, n)

    # Save the LaTeX code to a .tex file
    with open(os.path.join(config.RESULTS_LATEX, 'rosin_rammler_graph.tex'), 'w') as f:
        f.write(latex_code)



def calculate_cumulative_fractions(droplet_sizes):
    sorted_sizes = np.sort(droplet_sizes)
    cumulative_fractions = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
    return sorted_sizes, cumulative_fractions


