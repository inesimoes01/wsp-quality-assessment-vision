import numpy as np
import matplotlib.pyplot as plt


def rosin_rammler_distribution(x, x_o, n):
    return 1 - np.exp(-(x / x_o)**n)

def generate_droplet_sizes_rosin_rammler(x_o, n, num_droplets):
    # Generate random numbers from 0 to 1
    random_numbers = np.random.rand(num_droplets)
    # Use inverse transform sampling to generate droplet sizes
    droplet_sizes = x_o * (-np.log(1 - random_numbers))**(1/n)
    
    return droplet_sizes

x_o = 7.0  # Characteristic particle size
n = 2.0    # Uniformity constant
num_droplets = 1000

droplet_sizes = generate_droplet_sizes(x_o, n, num_droplets)

print(droplet_sizes)

plt.hist(droplet_sizes, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Droplet Size')
plt.ylabel('Probability Density')
plt.title('Rosin-Rammler Distribution of Droplet Sizes')
plt.grid(True)
plt.show()
