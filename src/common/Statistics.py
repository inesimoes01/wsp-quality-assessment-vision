import numpy as np
import sys

sys.path.insert(0, 'src/common')
import config as config
from Util import *
from Droplet import *

#TODO adjust pixels to real word dimensions

class Statistics:
    def __init__(self, vmd_value:float, rsf_value:float, coverage_percentage:float, no_droplets:int, no_droplets_overlapped, overlaped_percentage, droplet_info = None):
        self.vmd_value = vmd_value
        self.rsf_value = rsf_value
        self.coverage_percentage = coverage_percentage
        self.no_droplets = no_droplets
        self.droplet_info = droplet_info
        self.no_droplets_overlapped = no_droplets_overlapped
        self.overlaped_percentage = overlaped_percentage
    
    
    def calculate_statistics(diameters_sorted, image_area, contour_area):
        if len(diameters_sorted) > 0:
            cumulative_fraction = Statistics.calculate_cumulative_fraction(diameters_sorted)
            vmd_value = Statistics.calculate_vmd(cumulative_fraction, diameters_sorted)
            coverage_percentage = Statistics.calculate_coverage_percentage(image_area, contour_area)
            rsf_value = Statistics.calculate_rsf(cumulative_fraction, diameters_sorted, vmd_value)
            return vmd_value, coverage_percentage, rsf_value, cumulative_fraction
        else: return 0, 0, 0, 0


    def calculate_cumulative_fraction(diameters_sorted):
        total_diameter = sum(diameters_sorted)
        return np.cumsum(diameters_sorted) / total_diameter

    def calculate_vmd(cumulative_fraction, diameters_sorted):
        vmd_index = np.argmax(cumulative_fraction >= 0.5)
        return diameters_sorted[vmd_index]

    def calculate_coverage_percentage(image_area, contour_area):
        return contour_area / image_area * 100

    def calculate_rsf(cumulative_fraction, diameters_sorted, vmd_value):
        dv_one = np.argmax(cumulative_fraction >= 0.1)
        dv_nine = np.argmax(cumulative_fraction >= 0.9)
        rsf = (diameters_sorted[dv_nine] - diameters_sorted[dv_one]) / vmd_value
        return rsf
    
    def area_to_diameter_micro(droplet_area, width_px, width_mm):
        ratio_pxTOcm = width_mm * 1000 / width_px
    
        diameter_list = []
        for area_px in droplet_area:
            diameter = 2 * np.sqrt(area_px / np.pi) * ratio_pxTOcm        
            diameter_list.append(diameter)
        return diameter_list
    
