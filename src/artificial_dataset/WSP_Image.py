import numpy as np
import cv2
from datetime import datetime
import sys

sys.path.insert(0, 'src/common')
from Variables import *
from Util import *

sys.path.insert(0, 'src')
from Droplet import *

class WSP_Image:
    def __init__(self, index:int, colors, today_date:datetime):
        self.index = index
        self.max_num_spots:int = max_num_spots
        self.min_num_spots:int = min_num_spots
        self.max_radius:int = max_radius
        self.width:int = width * resolution
        self.height:int = height * resolution
        self.droplet_color = colors.droplet_color
        self.background_color = colors.background_color
        self.today_date:datetime = today_date

        self.generate_one_wsp()
        self.save_image(self.rectangle, isShadow = False)

        image_with_shadow = self.add_shadow()
        self.save_image(image_with_shadow, isShadow = True)

    def generate_one_wsp(self):
        # rectangle for background
        self.rectangle = np.full((self.height, self.width, 3), self.background_color, dtype=np.uint8)

        # generate number of spots
        self.num_spots = np.random.randint(min_num_spots, max_num_spots)

        # generate random spots with colors from the list
        self.droplets_data:list[Droplet] = []
        for i in range(self.num_spots):
            spot_color = self.droplet_color[np.random.randint(0, len(self.droplet_color))]
            spot_radius = np.random.randint(1, max_radius) 
            center_x = np.random.randint(spot_radius, self.width - spot_radius)
            center_y = np.random.randint(spot_radius, self.height - spot_radius)
            cv2.circle(self.rectangle, (center_x, center_y), spot_radius, spot_color, -1)

            # save each Droplet
            self.droplets_data.append(Droplet(center_x, center_y, spot_radius, i+1, [], spot_color))

        self.droplet_radii = [d.radius for d in self.droplets_data]


    def add_shadow(self):
        # create shadow shape
        shadow_mask = np.zeros_like(self.rectangle[:, :, 0], dtype=np.uint8)
        rows, cols = shadow_mask.shape
        triangle = np.array([[0, rows], [cols, rows], [0, 0]], np.int32)
        cv2.fillPoly(shadow_mask, [triangle], 255)
        shadow_mask_3channel = cv2.merge((shadow_mask, shadow_mask, shadow_mask))

        # apply shadow effect 
        return cv2.addWeighted(self.rectangle, 1, shadow_mask_3channel, -0.2, -0.5)


    def save_image(self, image, isShadow):
        self.blur_image = cv2.blur(image, (3,3))

        if isShadow: cv2.imwrite(path_to_images_folder + '\\' + self.today_date + '_' + str(self.index) + '_shadow.png', self.blur_image)
        else: cv2.imwrite(path_to_images_folder + '\\' + self.today_date + '_' + str(self.index) + '.png', self.blur_image)
        
        



    
