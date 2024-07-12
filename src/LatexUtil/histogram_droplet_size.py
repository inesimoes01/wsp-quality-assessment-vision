import pandas as pd
import matplotlib.pyplot as plt


# Load the CSV data into a pandas DataFrame
df = pd.read_csv('results\latex\droplet_data.csv')

# Plotting the histogram
plt.hist(df['DropletSize'], bins=10, edgecolor='black')
plt.xlabel('Droplet Size')
plt.ylabel('Frequency')
plt.title('Histogram of Droplet Size')
plt.show()