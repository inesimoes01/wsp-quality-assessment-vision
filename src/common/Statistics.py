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
    
    def calculate_statistics(volumes_sorted, image_area, contour_area):
        cumulative_fraction = Statistics.calculate_cumulative_fraction(volumes_sorted)
        vmd_value = Statistics.calculate_vmd(cumulative_fraction, volumes_sorted)
        coverage_percentage = Statistics.calculate_coverage_percentage(image_area, contour_area)
        rsf_value = Statistics.calculate_rsf(cumulative_fraction, volumes_sorted, vmd_value)
        return vmd_value, coverage_percentage, rsf_value, cumulative_fraction


    def calculate_cumulative_fraction(volumes_sorted):
        total_volume = sum(volumes_sorted)
        return np.cumsum(volumes_sorted) / total_volume

    def calculate_vmd(cumulative_fraction, volumes_sorted):
        vmd_index = np.argmax(cumulative_fraction >= 0.5)
        return volumes_sorted[vmd_index]

    def calculate_coverage_percentage(image_area, contour_area):
        return contour_area / image_area * 100

    def calculate_rsf(cumulative_fraction, volumes_sorted, vmd_value):
        dv_one = np.argmax(cumulative_fraction >= 0.1)
        dv_nine = np.argmax(cumulative_fraction >= 0.9)
        rsf = (volumes_sorted[dv_nine] - volumes_sorted[dv_one]) / vmd_value
        return rsf

    def area_to_volume(droplet_area, width_px, width_mm):
        ratio_pxTOcm = width_mm / width_px
        
        volume_list = []
        for area_px in droplet_area:
            area_mm = area_px * ratio_pxTOcm        
            volume_list.append((area_mm * 4 * np.pi) / 6)
        return volume_list
    
