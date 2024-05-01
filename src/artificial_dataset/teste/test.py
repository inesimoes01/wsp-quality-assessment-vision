# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # # Example dataset (term occurrences across documents)
# # # dataset = {
# # #     'term1': [2, 3, 4, 5, 6],  # Example occurrences of term1 across 5 documents
# # #     'term2': [1, 1, 1, 2, 2],  # Example occurrences of term2 across 5 documents
# # #     # Add more terms and their occurrences here
# # # }

# # # # Calculate variance for each term
# # # term_variances = {term: np.var(occurrences) for term, occurrences in dataset.items()}

# # # # Plot term variances
# # # plt.bar(term_variances.keys(), term_variances.values())
# # # plt.xlabel('Term')
# # # plt.ylabel('Variance')
# # # plt.title('Term Variance Across Documents')
# # # plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
# # # plt.tight_layout()  # Adjust layout to prevent clipping of labels
# # # plt.show()

# # # import numpy as np
import matplotlib.pyplot as plt

# # # # Example dataset of radius values
# # # radius_values = [10, 15, 20, 25, 30]

# # # # Calculate the variance of the radius values
# # # radius_variance = np.var(radius_values)

# # # print("Variance of radius values:", radius_variance)

# # # # Plot the variance
# # # plt.figure(figsize=(8, 6))
# # # plt.bar("Variance", radius_variance, color='blue')
# # # plt.title("Variance of Radius Values")
# # # plt.ylabel("Variance")
# # # plt.show()


# # import pandas as pd

# # data = [[10, 18, 11], [13, 15, 8], [9, 20, 3]]

# # df = pd.DataFrame(data)
# # df = df.var()

# # print("Variance of radius values:", df)
# # df.plot()
# # # # Plot the variance
# # # plt.figure(figsize=(8, 6))
# # # plt.
# # # plt.bar("Variance", df, color='blue')
# # # plt.title("Variance of Radius Values")
# # # plt.ylabel("Variance")
# # plt.show()

# # print(df.var())

# # A code to illustrate the var() function in Pandas

# # Importing the pandas library
# import pandas as pd

# # Creating a DataFrame
# df = pd.DataFrame([[1,2,3,4,5],
#                    [1,7,5,9,0.5],
#                    [3,11,13,14,12]],
#                    columns=list('ABCDE'))
# # Printing the DataFrame
# # print(df)

# # # Obtaining the median value vertically across rows
# # print(df.var())

# # Obtaining the median value horizontally over columns
# print(df.var(axis="columns"))

import numpy as np

def nukiyama_tanasawa_distribution(d_min, d_max, n, k):
    """
    Generate droplet sizes following the Nukiyama-Tanasawa distribution.

    Parameters:
        d_min (float): Minimum droplet size.
        d_max (float): Maximum droplet size.
        n (int): Number of droplets.
        k (float): Shape parameter of the distribution.

    Returns:
        droplet_sizes (ndarray): Array of droplet sizes.
    """
    u = np.random.rand(n)
    droplet_sizes = d_min * ((-np.log(1 - u)) ** (1 / k))
    droplet_sizes = np.clip(droplet_sizes, d_min, d_max)
    return droplet_sizes

def rosin(n, D50, n_slope):


# Generate droplet sizes
    #sizes = np.random.exponential(100*exp(-(x/k)**n))
    sizes = np.random.exponential(scale=D50 / (np.log(2) ** (1 / n_slope)), size=n)
    return sizes


# Parameters for Rosin-Rammler distribution
n = 100 # Number of droplets
D50 = 10.0  # Median droplet size
n_slope = 2.0  # Distribution slope parameter
#sizes = nukiyama_tanasawa_distribution(1, 150, 1000, 1)
sizes = rosin(n, D50, n_slope)
print(sizes)

plt.hist(sizes, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Droplet Size')
plt.ylabel('Probability Density')
plt.title('Rosin-Rammler Distribution of Droplet Sizes')
plt.grid(True)
plt.show()

# # Example usage:
# d_min = 1.0  # Minimum droplet size
# d_max = 10.0  # Maximum droplet size
# n = 1  # Number of droplets
# k = 2.0  # Shape parameter

# droplet_sizes = nukiyama_tanasawa_distribution(d_min, d_max, n, k)

# # Print the generated droplet sizes
# print(droplet_sizes)
