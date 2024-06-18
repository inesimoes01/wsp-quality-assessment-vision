import numpy as np
import cv2
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import copy
import math
from PIL.PngImagePlugin import PngInfo
import random


sys.path.insert(0, 'src/common')
import config as config
from Util import *

sys.path.insert(0, 'src')
from Droplet import *
from Colors import *

class CreateWSP:
    def __init__(self, index:int, colors:Colors, num_spots, type_of_dataset):
        self.filename = index
        self.max_num_spots:int = config.MAX_NUM_SPOTS
        self.min_num_spots:int = config.MIN_NUM_SPOTS
        self.width:float = config.WIDTH_MM * config.RESOLUTION
        self.height:float = config.HEIGHT_MM * config.RESOLUTION
        self.droplet_color_small = colors.droplet_color_small
        self.droplet_color_big = colors.droplet_color_big
        self.background_color_1 = colors.background_color_1
        self.background_color_2 = colors.background_color_2
        self.num_spots = num_spots

        self.type_of_dataset = type_of_dataset

        self.generate_one_wsp()
        self.save_image(self.rectangle)
    
    

    def generate_one_wsp(self):
        # rectangle for background
        rectangle = self.create_background(self.background_color_1, self.background_color_2)
        rectangle.save("temp.png")
        self.rectangle = cv2.imread("temp.png")

        # generate the droplet sizes given a rosin rammler distribution
        self.droplet_radius = self.generate_droplet_sizes_rosin_rammler(config.CHARACTERISTIC_PARTICLE_SIZE, config.UNIFORMITY_CONSTANT, self.num_spots)
        self.droplets_data:list[Droplet] = []
        areThereElipses = False

        # TODO maybe make the circles be outside the wsp
        match self.type_of_dataset:

            case 0:     # only singular circles inside the wsp
                areThereElipses = False
                for i in range(self.num_spots):
                    isElipse = False
                    
                    spot_radius = math.ceil(self.droplet_radius[i])

                    if spot_radius < config.DROPLET_COLOR_THRESHOLD:
                        spot_color = self.droplet_color_big[np.random.randint(0, len(self.droplet_color_big))]
                    else:
                        spot_color = self.droplet_color_small[np.random.randint(0, len(self.droplet_color_small))]
                    
                    count = 0
                    while (True):
                        center_x = np.random.randint(spot_radius, self.width - spot_radius)
                        center_y = np.random.randint(spot_radius, self.height - spot_radius)

                        overlap_bool = self.check_if_overlapping(center_x, center_y, spot_radius)
                
                        if not overlap_bool:
                            self.save_draw_droplet(areThereElipses, int(center_x), int(center_y), int(spot_radius), spot_color, i)
                            break
                        
                        count += 1
                        if count > 100:
                            break
            
            case 1:     # only singular droplets (circular and elipse)
                areThereElipses = True
                for i in range(self.num_spots):
                    isElipse = False
                    
                    spot_radius = math.ceil(self.droplet_radius[i])

                    if spot_radius < config.DROPLET_COLOR_THRESHOLD:
                        spot_color = self.droplet_color_big[np.random.randint(0, len(self.droplet_color_big))]
                    else:
                        spot_color = self.droplet_color_small[np.random.randint(0, len(self.droplet_color_small))]
                    
                    count = 0
                    while (True):
                        center_x = np.random.randint(spot_radius, self.width - spot_radius)
                        center_y = np.random.randint(spot_radius, self.height - spot_radius)

                        overlap_bool = self.check_if_overlapping(center_x, center_y, spot_radius)
                
                        if not overlap_bool:
                            self.save_draw_droplet(areThereElipses, int(center_x), int(center_y), int(spot_radius), spot_color, i)
                            break
                        
                        count += 1
                        if count > 100:
                            break

            case 2:     # overlapped and singular circles
                areThereElipses = False
                for i in range(self.num_spots):
                    isElipse = False
                    
                    spot_radius = math.ceil(self.droplet_radius[i])

                    if spot_radius < config.DROPLET_COLOR_THRESHOLD: spot_color = self.droplet_color_big[np.random.randint(0, len(self.droplet_color_big))]
                    else: spot_color = self.droplet_color_small[np.random.randint(0, len(self.droplet_color_small))]
                        
                    center_x = np.random.randint(spot_radius, self.width - spot_radius)
                    center_y = np.random.randint(spot_radius, self.height - spot_radius)

                    self.save_draw_droplet(areThereElipses, int(center_x), int(center_y), int(spot_radius), spot_color, i)
                    
            case 3:     # overlapped and singular droplets (circles and elipses)
                areThereElipses = True
                for i in range(self.num_spots):
                    isElipse = False
                    
                    spot_radius = math.ceil(self.droplet_radius[i])

                    if spot_radius < config.DROPLET_COLOR_THRESHOLD: spot_color = self.droplet_color_big[np.random.randint(0, len(self.droplet_color_big))]
                    else: spot_color = self.droplet_color_small[np.random.randint(0, len(self.droplet_color_small))]
                        
                    center_x = np.random.randint(spot_radius, self.width - spot_radius)
                    center_y = np.random.randint(spot_radius, self.height - spot_radius)
                    
                    self.save_draw_droplet(areThereElipses, int(center_x), int(center_y), int(spot_radius), spot_color, i)

            case 4:     # classes overlapped and singular are leveled
                areThereElipses = True
                count_overlapping_sets = 0
                count_single = 0
                num_overlapping_droplets = math.ceil(self.num_spots * 1 / 3)  
                num_overlapping_sets = math.ceil(self.num_spots * 1 / 3) 
                overlapped_sets = self.generate_overlapping_value(num_overlapping_sets, num_overlapping_droplets)
                sumy = sum(overlapped_sets)
                count_total_no_droplets = 0
                
                while(self.num_spots > count_total_no_droplets):
                
                    # create only overlapping circles
                    if count_overlapping_sets < num_overlapping_sets:
                        no_drops_in_set = overlapped_sets[count_overlapping_sets]

                        # initial circle position
                        center_x = np.random.randint(5, self.width - 5)
                        center_y = np.random.randint(5, self.height - 5)

                        for k in range(no_drops_in_set): 
                            spot_radius = math.ceil(self.droplet_radius[count_total_no_droplets])
                            if spot_radius < config.DROPLET_COLOR_THRESHOLD: spot_color = self.droplet_color_big[np.random.randint(0, len(self.droplet_color_big))]
                            else: spot_color = self.droplet_color_small[np.random.randint(0, len(self.droplet_color_small))]

                            # if n_spot is even it will move to be on the right of the original circle
                            # if n_spot is odd it will move to be under the original circle
                            if k % 2 == 0: center_x += self.droplet_radius[count_total_no_droplets - 1 ]
                            else: center_y += self.droplet_radius[count_total_no_droplets - 1] 
                                
                            count_total_no_droplets += 1

                            self.save_draw_droplet(areThereElipses, int(center_x), int(center_y), int(spot_radius), spot_color, count_total_no_droplets)

                        count_overlapping_sets += 1
                    
                    # create only single droplets
                    else:
                        spot_radius = math.ceil(self.droplet_radius[count_total_no_droplets])
                        if spot_radius < config.DROPLET_COLOR_THRESHOLD: spot_color = self.droplet_color_big[np.random.randint(0, len(self.droplet_color_big))]
                        else: spot_color = self.droplet_color_small[np.random.randint(0, len(self.droplet_color_small))]

                        while (True):
                            center_x = np.random.randint(spot_radius, self.width - spot_radius)
                            center_y = np.random.randint(spot_radius, self.height - spot_radius)

                            overlapping = self.check_if_overlapping(center_x, center_y, spot_radius)

                            if not overlapping:
                                break
                        
                        count_total_no_droplets += 1
                        self.save_draw_droplet(areThereElipses, int(center_x), int(center_y), int(spot_radius), spot_color, count_total_no_droplets)

                    
        self.droplet_radius = [r.radius for r in self.droplets_data]

    def save_draw_droplet(self, areThereElipses, center_x, center_y, spot_radius, spot_color, i):
        isElipse = False
        if areThereElipses:
            if (i % 10 == 1): 
                isElipse = True
                cv2.ellipse(self.rectangle, (center_x, center_y), (spot_radius, spot_radius + 5), 5, 0, 360, spot_color, -1)
            else: cv2.circle(self.rectangle, (center_x, center_y), spot_radius, spot_color, -1)
        else:
            cv2.circle(self.rectangle, (center_x, center_y), spot_radius, spot_color, -1)

        self.droplets_data.append(Droplet(isElipse, center_x, center_y, spot_radius, i+1, [], spot_color))

    def generate_overlapping_value(self, no_overlap_sets, no_droplets):
        numbers = [2] * no_overlap_sets
        remaining_sum = no_droplets - no_overlap_sets* 2

        while remaining_sum > 0:
            index = random.randint(0, no_overlap_sets - 1)
            numbers[index] += 1
            remaining_sum -= 1

        return numbers
    
    def check_if_overlapping(self, center_x, center_y, spot_radius):
        overlapping = False
        for droplet in self.droplets_data:
            distance = math.sqrt((center_x - droplet.center_x)**2 + (center_y - droplet.center_y)**2)
            if distance - 5 < spot_radius + droplet.radius:
                overlapping = True
                break

        return overlapping
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
        rectangle = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))

        # Draw first polygon with radial gradient
        polygon = [(0, 0), (self.width, 0),(self.width, self.height), (0, self.height), ]
        point = (self.height*2/3, self.width/4)
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
        path = os.path.join(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR, str(self.filename) + '.png')

        self.blur_image = cv2.GaussianBlur(image, (3, 3), 0)
        cv2.imwrite(path, self.blur_image)  

    def generate_droplet_sizes_rosin_rammler(self, x_o, n, no_droplets):
        # list of size no_droplets with random numbers from 0 to 1
        random_numbers = np.random.rand(no_droplets)

        # inverse transform sampling to generate droplet sizes
        droplet_sizes = x_o * (-np.log(1 - random_numbers))**(1/n)
        
        return droplet_sizes

