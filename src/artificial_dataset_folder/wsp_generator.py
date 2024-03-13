import numpy as np
import cv2

### VALUES
max_num_spots = 400
min_num_spots = 300
max_radius = 15
width, height = 76*26, 26*26

def generate_one_wsp(droplet_color, background_color):
    # rectangle for background
    rectangle = np.full((height, width, 3), background_color, dtype=np.uint8)

    # generate number of spots
    num_spots = np.random.randint(min_num_spots, max_num_spots)

    # generate random spots with colors from the list
    droplets_data = []
    for _ in range(num_spots):
        spot_color = droplet_color[np.random.randint(0, len(droplet_color))]
        spot_radius = np.random.randint(1, max_radius) 
        center_x = np.random.randint(spot_radius, width - spot_radius)
        center_y = np.random.randint(spot_radius, height - spot_radius)
        cv2.circle(rectangle, (center_x, center_y), spot_radius, spot_color, -1)

        droplets_data.append({
            'color': spot_color,
            'radius': spot_radius,
            'center_x': center_x,
            'center_y': center_y
        })

    return width, height, rectangle, num_spots, droplets_data
    
