import numpy as np
import sys

sys.path.insert(0, 'src/common')
import config as config
from Util import *
from Droplet import *

#TODO adjust pixels to real word dimensions

class Statistics:
    def __init__(self, vmd_value:float, rsf_value:float, coverage_percentage:float, no_droplets:int, droplet_info:list[Droplet]):
        self.vmd_value = vmd_value
        self.rsf_value = rsf_value
        self.coverage_percentage = coverage_percentage
        self.no_droplets = no_droplets
        self.droplet_info = droplet_info
        
        self.no_droplets_overlapped = 0
        for drop in self.droplet_info:
            if len(drop.overlappedIDs) > 0:
                self.no_droplets_overlapped += 1
        
        self.overlaped_percentage = self.no_droplets_overlapped / self.no_droplets * 100
    
        
    def calculate_cumulative_fraction(volumes_sorted):
        total_volume = sum(volumes_sorted)
        return np.cumsum(volumes_sorted) / total_volume

    def calculate_vmd(cumulative_fraction, volumes_sorted):
        vmd_index = np.argmax(cumulative_fraction >= 0.5)
        return volumes_sorted[vmd_index]

    def calculate_coverage_percentage_gt(image, image_height, image_width, background_color1, background_color2):
        background_upper = np.array(background_color1, dtype=np.uint8)  # lower bound for yellow
        background_lower = np.array(background_color2, dtype=np.uint8)  # upper bound for yellow

        # sum number of pixels that are part of the background
        not_covered_area = 0
        for y in range(image_height):
            for x in range(image_width):
                #pixel = image.getpixel((x, y))
                droplet_bgr = tuple(image[y, x])
                droplet_rgb = droplet_bgr[::-1]

                # check if the pixel falls within the yellow range
                isYellow = np.all(droplet_rgb >= background_lower) and np.all(droplet_rgb <= background_upper)
                
                if isYellow:
                    not_covered_area += 1
        
        # calculate percentage of paper that is coverered
        total_area = image_width * image_height
        return ((total_area - not_covered_area) / total_area) * 100
    
    def calculate_coverage_percentage_c(image, image_height, image_width, contour_area):
        total_area = image_height * image_width

        return contour_area / total_area * 100

    def calculate_rsf(cumulative_fraction, volumes_sorted, vmd_value):
        dv_one = np.argmax(cumulative_fraction >= 0.1)
        dv_nine = np.argmax(cumulative_fraction >= 0.9)
        rsf = (volumes_sorted[dv_nine] - volumes_sorted[dv_one]) / vmd_value
        return rsf

    # def save_stats_file(self, filename):
    #     stats 
    #     statistics_file_path = path_to_real_dataset_inesc + '\\stats\\' + filename + '.txt'
    #     with open(statistics_file_path, 'w') as f:
    #         f.write(f"Number of droplets: {self.no_droplets:d}\n")
    #         f.write(f"Coverage percentage: {self.coverage_percentage:.2f}\n")
    #         f.write(f"VMD value: {self.vmd_value:.2f}\n")
    #         f.write(f"RSF value: {self.rsf_value:.2f}\n")
    #         f.write(f"RSF value: {self.cover:.2f}\n")

    def radius_to_volume(droplet_radius, width_px):
        ratio_pxTOcm = config.WIDTH_MM / width_px
        
        volume_list = []
        for radius_px in droplet_radius:
            radius_mm = radius_px * ratio_pxTOcm        
            volume_list.append((radius_mm * 2 * np.pi) / 6)
        return volume_list
    

    