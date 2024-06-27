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
import CreateDroplet
import CreateBackground
from CreateShapes import Shapes


sys.path.insert(0, 'src/common')
import config as config
from Util import *

sys.path.insert(0, 'src')
from Droplet import *
from CreateColors import *

class CreateWSP:
    def __init__(self, index:int, colors:Colors, shapes:Shapes, num_spots, type_of_dataset):
        self.filename = index
        self.max_num_spots:int = config.MAX_NUM_SPOTS
        self.min_num_spots:int = config.MIN_NUM_SPOTS
        self.width:float = config.WIDTH_MM * config.RESOLUTION
        self.height:float = config.HEIGHT_MM * config.RESOLUTION
        self.num_spots = num_spots
        self.colors = colors
        self.shapes = shapes

        self.type_of_dataset = type_of_dataset
       
        # generate the droplet sizes given a rosin rammler distribution
        self.droplet_radius = self.generate_droplet_sizes_rosin_rammler(config.CHARACTERISTIC_PARTICLE_SIZE, config.UNIFORMITY_CONSTANT, self.num_spots)
        
        CreateBackground.create_background(self.colors.background_color_1, self.colors.background_color_2, self.width, self.height)
        self.rectangle = cv2.imread(config.DATA_ARTIFICIAL_RAW_BACKGROUND_IMG, cv2.IMREAD_COLOR)
        self.rectangle = cv2.cvtColor(self.rectangle, cv2.COLOR_BGR2RGB)
        
        self.generate_one_wsp()

        self.save_image(self.rectangle)
    
    
    def generate_one_wsp(self):
        # rectangle for background
        self.droplets_data:list[Droplet] = []
        areThereElipses = False

        match self.type_of_dataset:

            case 0:     # only singular circles inside the wsp
                areThereElipses = False
                for i in range(self.num_spots):
                    spot_radius, spot_color = self.get_droplet(i)
                    
                    count = 0
                    while (True):
                        center_x = np.random.randint(spot_radius, self.width - spot_radius)
                        center_y = np.random.randint(spot_radius, self.height - spot_radius)

                        overlap_bool = self.check_if_overlapping(center_x, center_y, spot_radius, i, areThereElipses)
                
                        if not overlap_bool:
                            self.save_draw_droplet(areThereElipses, int(center_x), int(center_y), int(spot_radius), spot_color, i)
                            break
                        
                        count += 1
                        if count > 100:
                            break
            
            case 1:     # only singular droplets (circular and elipse)
                areThereElipses = True
                for i in range(self.num_spots):
                    spot_radius, spot_color = self.get_droplet(i)
                    
                    count = 0
                    while (True):
                        center_x = np.random.randint(spot_radius, self.width - spot_radius)
                        center_y = np.random.randint(spot_radius, self.height - spot_radius)

                        overlap_bool = self.check_if_overlapping(center_x, center_y, spot_radius, i, areThereElipses)
                
                        if not overlap_bool:
                            self.save_draw_droplet(areThereElipses, int(center_x), int(center_y), int(spot_radius), spot_color, i)
                            break
                        
                        count += 1
                        if count > 100:
                            break

            case 2:     # overlapped and singular circles
                areThereElipses = False
                for i in range(self.num_spots):
                    spot_radius, spot_color = self.get_droplet(i)
                        
                    center_x = np.random.randint(spot_radius, self.width - spot_radius)
                    center_y = np.random.randint(spot_radius, self.height - spot_radius)

                    self.save_draw_droplet(areThereElipses, int(center_x), int(center_y), int(spot_radius), spot_color, i)
                    
            case 3:     # overlapped and singular droplets (circles and elipses)
                areThereElipses = True
                for i in range(self.num_spots):
                    spot_radius, spot_color = self.get_droplet(i)

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
                            spot_radius, spot_color = self.get_droplet(count_total_no_droplets)
                            no_rand = random.randint(0, 1)
                            # if n_spot is even it will move to be on the right of the original circle
                            # if n_spot is odd it will move to be under the original circle
                            if no_rand == 0:
                                center_x += self.droplet_radius[count_total_no_droplets - 1] + self.droplet_radius[count_total_no_droplets]
                            else: 
                                center_y += self.droplet_radius[count_total_no_droplets - 1] + self.droplet_radius[count_total_no_droplets]
                            # if k % 2 == 0: center_x += self.droplet_radius[count_total_no_droplets - 1 ]
                            # else: center_y += self.droplet_radius[count_total_no_droplets - 1] 
                                
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

                            overlapping = self.check_if_overlapping(center_x, center_y, spot_radius, count_total_no_droplets, areThereElipses)

                            if not overlapping:
                                break
                        
                        count_total_no_droplets += 1
                        self.save_draw_droplet(areThereElipses, int(center_x), int(center_y), int(spot_radius), spot_color, count_total_no_droplets)

        self.droplet_radius = [r.radius for r in self.droplets_data]

    def get_droplet(self, i):
        spot_radius = math.ceil(self.droplet_radius[i])
        spot_color = (255, 255, 255)

        # if spot_radius < config.DROPLET_COLOR_THRESHOLD_1:
        #     spot_color = self.droplet_color_big[np.random.randint(0, len(self.droplet_color_big))]
        # else:
        #     spot_color = self.droplet_color_small[np.random.randint(0, len(self.droplet_color_small))]

        return spot_radius, spot_color

    def save_draw_droplet(self, areThereElipses, center_x, center_y, spot_radius, spot_color, i):

        isElipse = False
        if areThereElipses:
            if (i % 10 == 0): 
                isElipse = True
                CreateDroplet.draw_perfect_circle(self.rectangle, (center_x, center_y), spot_radius, self.colors.light_blue_rgb, self.colors.dark_blue_rgb, self.colors.brown_rgb, self.colors.background_color_1)

                #cv2.ellipse(self.rectangle, (center_x, center_y), (spot_radius, spot_radius + config.ELIPSE_MAJOR_AXE_VALUE), 5, 0, 360, spot_color, -1)
            else: CreateDroplet.draw_perfect_circle(self.rectangle, (center_x, center_y), spot_radius, self.colors.light_blue_rgb, self.colors.dark_blue_rgb, self.colors.brown_rgb,  self.colors.background_color_1)

                #cv2.circle(self.rectangle, (center_x, center_y), spot_radius, spot_color, -1)
        else:
            CreateDroplet.draw_perfect_circle(self.rectangle, (center_x, center_y), spot_radius, self.colors.light_blue_rgb, self.colors.dark_blue_rgb, self.colors.brown_rgb, self.colors.background_color_1)

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
    
    def check_if_overlapping(self, center_x, center_y, spot_radius, i, areThereElipses):
        overlapping = False

        for droplet in self.droplets_data:
            distance = math.sqrt((center_x - droplet.center_x)**2 + (center_y - droplet.center_y)**2)

            # circle and circle
            if ((i % 10 == 1) and not droplet.isElipse):
                # add a small value to make sure the droplets are actually overlapped and not just touching
                if distance < spot_radius + droplet.radius + config.OVERLAPPING_THRESHOLD:
                    overlapping = True
                    break

            # circle and elipse or elipse and circle
            elif (((i % 10 == 1) and droplet.isElipse) or ((i % 10 == 0) and not droplet.isElipse)) and areThereElipses:
                if distance < spot_radius + droplet.radius + config.ELIPSE_MAJOR_AXE_VALUE + config.OVERLAPPING_THRESHOLD:
                    overlapping = True
                    break

            # elipse and elipse
            elif ((i % 10 == 0) and droplet.isElipse) and areThereElipses:
                if distance < spot_radius + droplet.radius + config.ELIPSE_MAJOR_AXE_VALUE * 2 + config.OVERLAPPING_THRESHOLD:
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

    def save_image(self, image):
        path = os.path.join(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR, str(self.filename) + '.png')

        self.blur_image = cv2.GaussianBlur(image, (3, 3), 0)
        rgb_image = cv2.cvtColor(self.blur_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, rgb_image)

    def generate_droplet_sizes_rosin_rammler(self, x_o, n, no_droplets):
        # list of size no_droplets with random numbers from 0 to 1
        random_numbers = np.random.rand(no_droplets)

        # inverse transform sampling to generate droplet sizes
        droplet_sizes = x_o * (-np.log(1 - random_numbers))**(1/n)
        
        return droplet_sizes

