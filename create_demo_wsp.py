import numpy as np
import cv2

# Create a yellow background rectangle
width, height = 800, 600  # Set dimensions of the rectangle
yellow_color = (97, 225, 243)  # Yellow color in BGR format
rectangle = np.full((height, width, 3), yellow_color, dtype=np.uint8)

# Define a list of possible spot colors (in BGR format)
possible_colors = [
    (29, 33, 52), 
    (61, 42, 64),
    (89, 8, 37),
    (172, 4, 46)
]

# Generate random spots with colors from the list
num_spots = 100  # Number of spots
for _ in range(num_spots):
    spot_color = possible_colors[np.random.randint(0, len(possible_colors))]
    spot_radius = np.random.randint(1, 5)  # Random radius for spots
    center_x = np.random.randint(spot_radius, width - spot_radius)
    center_y = np.random.randint(spot_radius, height - spot_radius)
    cv2.circle(rectangle, (center_x, center_y), spot_radius, spot_color, -1)

# Display the generated image
cv2.imshow('Water Sensitive Paper Demo', rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
cv2.imwrite('water_sensitive_paper_demo.png', rectangle)
