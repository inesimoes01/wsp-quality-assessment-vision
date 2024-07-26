from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# Define a function to convert a Shapely polygon to a Matplotlib path
def polygon_to_path(polygon):
    vertices = []
    codes = []
    exterior = polygon.exterior.coords
    vertices.extend(exterior)
    codes.extend([Path.MOVETO] + [Path.LINETO]*(len(exterior) - 2) + [Path.CLOSEPOLY])
    
    for interior in polygon.interiors:
        vertices.extend(interior.coords)
        codes.extend([Path.MOVETO] + [Path.LINETO]*(len(interior.coords) - 2) + [Path.CLOSEPOLY])
    
    return Path(vertices, codes)

# Example shapes
shape1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
shape2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
shape3 = Polygon([(2, 0.5), (4, 0.5), (4, 2.5), (2, 2.5)])

# Merge shapes
merged_shapes = unary_union([shape1, shape2, shape3])

# Ensure merged_shapes is a list of polygons
if merged_shapes.geom_type == 'MultiPolygon':
    separated_shapes = list(merged_shapes)
else:
    separated_shapes = [merged_shapes]

# Plotting
fig, ax = plt.subplots()

for shape in separated_shapes:
    path = polygon_to_path(shape)
    patch = PathPatch(path, edgecolor='black', facecolor='none', linewidth=1)
    ax.add_patch(patch)

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_aspect('equal')
plt.show()
