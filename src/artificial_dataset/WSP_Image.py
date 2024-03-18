import numpy as np
import cv2
from datetime import datetime

from paths import path_to_images_folder, max_num_spots, min_num_spots, max_radius, width, height

class WSP_Image:
    def __init__(self, index, droplet_color, background_color, today_date):
        self.index = index
        self.max_num_spots = max_num_spots
        self.min_num_spots = min_num_spots
        self.max_radius = max_radius
        self.width = width
        self.height = height
        self.droplet_color = droplet_color
        self.background_color = background_color
        self.today_date = today_date
        self.generate_one_wsp()
        self.save_wsp()

    def generate_one_wsp(self):
        # rectangle for background
        self.rectangle = np.full((self.height, self.width, 3), self.background_color, dtype=np.uint8)

        # generate number of spots
        self.num_spots = np.random.randint(min_num_spots, max_num_spots)

        # generate random spots with colors from the list
        self.droplets_data = []
        for _ in range(self.num_spots):
            spot_color = self.droplet_color[np.random.randint(0, len(self.droplet_color))]
            spot_radius = np.random.randint(1, max_radius) 
            center_x = np.random.randint(spot_radius, width - spot_radius)
            center_y = np.random.randint(spot_radius, height - spot_radius)
            cv2.circle(self.rectangle, (center_x, center_y), spot_radius, spot_color, -1)

            self.droplets_data.append({
                'color': spot_color,
                'radius': spot_radius,
                'center_x': center_x,
                'center_y': center_y
            })

        self.droplet_radii = [d['radius'] for d in self.droplets_data]
    
    def save_wsp(self):
        cv2.imwrite(path_to_images_folder + '\wsp_' + self.today_date + '_' + str(self.index) + '.png', self.rectangle)
        return width, height, self.rectangle, self.num_spots, self.droplets_data





    
