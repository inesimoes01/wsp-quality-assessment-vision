import numpy as np
import sys

class Statistics:
    def __init__(self, vmd_value, rsf_value, coverage_percentage, no_droplets):
        self.vmd_value = vmd_value
        self.rsf_value = rsf_value
        self.coverage_percentage = coverage_percentage
        self.no_droplets = no_droplets

    # def __init__(self, image, image_height, image_width, background_color, volume_array, isGroundTruth):
    #     self.image = image
    #     self.image_height = image_height
    #     self.image_width = image_width
    #     self.background_color = background_color
    #     self.volumes_sorted = sorted(volume_array)

    #     self.calculate_cumulative_fraction()
    #     self.calculate_vmd()
    #     self.calculate_rsf()
    #     if isGroundTruth: self.calculate_coverage_percentage_gt()
    #     else: self.calculate_coverage_percentage_c()
        
    def calculate_cumulative_fraction(volumes_sorted):
        total_volume = sum(volumes_sorted)
        return np.cumsum(volumes_sorted) / total_volume

    def calculate_vmd(cumulative_fraction, volumes_sorted):
        vmd_index = np.argmax(cumulative_fraction >= 0.5)
        return volumes_sorted[vmd_index]
  
    def calculate_coverage_percentage_gt(image, image_height, image_width, background_color):
        # sum number of pixels that are part of the background
        not_covered_area = 0
        for y in range(image_height):
            for x in range(image_width):
                droplet_bgr = tuple(image[y, x])
                
                # check if pixel is yellow
                if tuple(map(lambda i, j: i - j, droplet_bgr, background_color)) == (0, 0, 0):
                    not_covered_area += 1
        
        # calculate percentage of paper that is coverered
        total_area = image_width * image_height
        return ((total_area - not_covered_area) / total_area) * 100
    
    def calculate_coverage_percentage_c(image, image_height, image_width, background_color):
        # sum number of pixels that are part of the background
        not_covered_area = 0
        for y in range(image_height):
            for x in range(image_width):
                droplet_bgr = tuple(image[y, x])
                
                # check if pixel is yellow
                if tuple(map(lambda i, j: i - j, droplet_bgr, background_color)) == (0, 0, 0):
                    not_covered_area += 1
        
        # calculate percentage of paper that is coverered
        total_area = image_width * image_height
        return ((total_area - not_covered_area) / total_area) * 100
    
    def calculate_rsf(cumulative_fraction, vmd_value):
        dv_one = np.argmax(cumulative_fraction >= 0.1)
        dv_nine = np.argmax(cumulative_fraction >= 0.9)

        return (dv_nine - dv_one) / vmd_value

    