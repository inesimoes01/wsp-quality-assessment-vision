import numpy as np
import matplotlib.pyplot as plt

# Parameters for the normal distribution
mean_droplets = 500   # Mean number of droplets per image
std_droplets = 200  # Standard deviation of droplets per image
num_images = 1000     # Number of images in the dataset

# Generate the number of droplets for each image using a normal distribution
num_droplets_per_image = np.random.normal(mean_droplets, std_droplets, num_images)
num_droplets_per_image = np.round(num_droplets_per_image).astype(int)  # Round to integer values

# Ensure non-negative values
num_droplets_per_image = np.maximum(num_droplets_per_image, 0)

# Plot the histogram of the number of droplets per image
plt.figure(figsize=(8, 6))
plt.hist(num_droplets_per_image, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Number of Droplets per Image')
plt.xlabel('Number of Droplets')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
