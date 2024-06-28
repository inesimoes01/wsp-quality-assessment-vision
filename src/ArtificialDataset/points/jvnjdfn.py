import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import PathPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Define the polygon points
polygon_points = np.array([
    [1, 1],
    [2, 3],
    [3, 1],
    [4, 4],
    [5, 1],
    [4, 0],
    [2, 0]
])

# Create a path from the polygon points
path = mpath.Path(polygon_points)

# Create a grid of points covering the bounding box of the polygon
x_min, y_min = polygon_points.min(axis=0)
x_max, y_max = polygon_points.max(axis=0)
x, y = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

# Flatten the grid for path contains_points method
points = np.vstack((x.flatten(), y.flatten())).T

# Check which points are inside the polygon
inside = path.contains_points(points).reshape(x.shape)

# Create a distance gradient from the polygon edge
distance_gradient = np.zeros_like(x)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if inside[i, j]:
            # Calculate the distance to the nearest polygon edge
            distance = np.min(np.sqrt((polygon_points[:, 0] - x[i, j])**2 + (polygon_points[:, 1] - y[i, j])**2))
            distance_gradient[i, j] = distance

# Normalize the distance gradient
norm = Normalize(vmin=distance_gradient.min(), vmax=distance_gradient.max())

# Create a colormap
cmap = plt.get_cmap('viridis')
mappable = ScalarMappable(norm=norm, cmap=cmap)

# Plot the polygon and fill with gradient
fig, ax = plt.subplots()
patch = PathPatch(path, facecolor='none', edgecolor='black')
ax.add_patch(patch)
ax.imshow(mappable.to_rgba(distance_gradient), extent=[x_min, x_max, y_min, y_max], origin='lower', alpha=0.7)

# Plot the polygon edge on top
ax.plot(polygon_points[:, 0], polygon_points[:, 1], color='black')

# Set the limits and aspect ratio
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal')

plt.show()
