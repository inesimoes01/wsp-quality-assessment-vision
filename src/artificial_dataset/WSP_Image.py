import numpy as np
import cv2
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import copy
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
from PIL.PngImagePlugin import PngInfo


sys.path.insert(0, 'src/common')
from Variables import *
from Util import *

sys.path.insert(0, 'src')
from Droplet import *

class WSP_Image:
    def __init__(self, index:int, colors):
        self.filename = index
        self.max_num_spots:int = max_num_spots
        self.min_num_spots:int = min_num_spots
        self.max_radius:int = max_radius
        self.width:float = width_mm * resolution
        self.height:float = height_mm * resolution
        self.droplet_color = colors.droplet_color
        self.background_color_1 = colors.background_color_1
        self.background_color_2 = colors.background_color_2

        self.generate_one_wsp()
        self.save_image(self.rectangle)
        

    def generate_one_wsp(self):
        # rectangle for background
        # TODO make this better, maybe change types of the image
        #self.rectangle = np.full((self.height, self.width, 3), self.background_color_1, dtype=np.uint8)
        rectangle = self.create_background(self.background_color_1, self.background_color_2)
        rectangle.save("temp.png")
        self.rectangle = cv2.imread("temp.png")
    
        # generate number of spots
        self.num_spots = np.random.randint(min_num_spots, max_num_spots)
        
        # generate random spots with colors from the list
        self.droplets_data:list[Droplet] = []
        for i in range(self.num_spots):
            isElipse = False
            spot_color = self.droplet_color[np.random.randint(0, len(self.droplet_color))]
            spot_radius = np.random.randint(min_radius, max_radius) 
            center_x = np.random.randint(spot_radius, self.width - spot_radius)
            center_y = np.random.randint(spot_radius, self.height - spot_radius)
            if (i % 10 == 1): 
                isElipse = True
                # spot_radius_difference = spot_radius + np.random.randint(1, 5)
                # angle = np.random.randint(1, 5)
                cv2.ellipse(self.rectangle, (center_x, center_y), (spot_radius, spot_radius + 5), 5, 0, 360, spot_color, -1)
            else: cv2.circle(self.rectangle, (center_x, center_y), spot_radius, spot_color, -1)

            # save each Droplet
            self.droplets_data.append(Droplet(isElipse, center_x, center_y, spot_radius*2, i+1, [], spot_color))

  
        self.droplet_diameter = [d.diameter for d in self.droplets_data]

    def add_shadow(self):
        # create shadow shape
        shadow_mask = np.zeros_like(self.rectangle[:, :, 0], dtype=np.uint8)
        rows, cols = shadow_mask.shape
        triangle = np.array([[0, rows], [cols, rows], [0, 0]], np.int32)
        cv2.fillPoly(shadow_mask, [triangle], 255)
        shadow_mask_3channel = cv2.merge((shadow_mask, shadow_mask, shadow_mask))

        # apply shadow effect 
        return cv2.addWeighted(self.rectangle, 1, shadow_mask_3channel, -0.2, -0.5)
    
    def create_background(self, color1, color2):
        w, h = (width_mm*resolution, height_mm*resolution)
        rectangle = Image.new('RGBA', (w, h), (0, 0, 0, 0))

        # Draw first polygon with radial gradient
        polygon = [(0, 0), (width_mm*resolution, 0),(width_mm*resolution, height_mm*resolution), (0, height_mm*resolution), ]
        point = (height_mm*resolution*2/3, width_mm*resolution/4)
        rectangle = self.radial_gradient(rectangle, polygon, point, color1, color2)

        return rectangle
    
    # Draw polygon with linear gradient from point 1 to point 2 and ranging
    # from color 1 to color 2 on given image
    def linear_gradient(self, i, poly, p1, p2, c1, c2):

        # Draw initial polygon, alpha channel only, on an empty canvas of image size
        ii = Image.new('RGBA', i.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ii)
        draw.polygon(poly, fill=(0, 0, 0, 255), outline=None)

        # Calculate angle between point 1 and 2
        p1 = np.array(p1)
        p2 = np.array(p2)
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) / np.pi * 180

        # Rotate and crop shape
        temp = ii.rotate(angle, expand=True)
        temp = temp.crop(temp.getbbox())
        wt, ht = temp.size

        # Create gradient from color 1 to 2 of appropriate size
        gradient = np.linspace(c1, c2, wt, True).astype(np.uint8)
        gradient = np.tile(gradient, [2 * ht, 1, 1])
        gradient = Image.fromarray(gradient)

        # Paste gradient on blank canvas of sufficient size
        temp = Image.new('RGBA', (max(i.size[0], gradient.size[0]),
                                max(i.size[1], gradient.size[1])), (0, 0, 0, 0))
        temp.paste(gradient)
        gradient = temp

        # Rotate and translate gradient appropriately
        x = np.sin(angle * np.pi / 180) * ht
        y = np.cos(angle * np.pi / 180) * ht
        gradient = gradient.rotate(-angle, center=(0, 0),
                                translate=(p1[0] + x, p1[1] - y))

        # Paste gradient on temporary image
        ii.paste(gradient.crop((0, 0, ii.size[0], ii.size[1])), mask=ii)

        # Paste temporary image on actual image
        i.paste(ii, mask=ii)

        return i

    # Draw polygon with radial gradient from point to the polygon border
    # ranging from color 1 to color 2 on given image
    def radial_gradient(self, i, poly, p, c1, c2):

        # Draw initial polygon, alpha channel only, on an empty canvas of image size
        ii = Image.new('RGBA', i.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ii)
        draw.polygon(poly, fill=(0, 0, 0, 255), outline=None)

        # Use polygon vertex with highest distance to given point as end of gradient
        p = np.array(p)
        max_dist = max([np.linalg.norm(np.array(v) - p) for v in poly])

        # Calculate color values (gradient) for the whole canvas
        x, y = np.meshgrid(np.arange(i.size[0]), np.arange(i.size[1]))
        c = np.linalg.norm(np.stack((x, y), axis=2) - p, axis=2) / max_dist
        c = np.tile(np.expand_dims(c, axis=2), [1, 1, 3])
        c = (c1 * (1 - c) + c2 * c).astype(np.uint8)
        c = Image.fromarray(c)

        # Paste gradient on temporary image
        ii.paste(c, mask=ii)

        # Paste temporary image on actual image
        i.paste(ii, mask=ii)

        return i

    def save_image(self, image):
        path = os.path.join(path_to_images_folder, str(self.filename) + '.png')

        self.blur_image = cv2.blur(image, (3,3))
        cv2.imwrite(path, self.blur_image)  

# background_color_1 = (255, 244, 137)
# background_color_2 = (185, 148, 0)
# wsp = WSP_Image(1)
# WSP_Image.create_background(wsp, background_color_2, background_color_1)      