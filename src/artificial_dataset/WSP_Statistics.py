import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt 
import copy

from Colors import *

sys.path.insert(0, 'src/common')
from Util import *
from Variables import * 
from Statistics import *

from WSP_Image import WSP_Image

class WSP_Statistics:
    def __init__(self, wsp_image:WSP_Image, colors):
        self.colors = colors
        self.wsp_image = wsp_image
     
        self.find_overlapping_circles()
     
        self.volume_list = sorted(Statistics.diameter_to_volume(self.wsp_image.droplet_diameter, wsp_image.width))
        
        cumulative_fraction = Statistics.calculate_cumulative_fraction(self.volume_list)
        vmd_value = Statistics.calculate_vmd(cumulative_fraction, self.volume_list)
        coverage_percentage = Statistics.calculate_coverage_percentage_gt(self.wsp_image.rectangle, self.wsp_image.height, self.wsp_image.width, self.wsp_image.background_color_1, self.wsp_image.background_color_2)
        rsf_value = Statistics.calculate_rsf(cumulative_fraction, self.volume_list, vmd_value)
       
        self.stats:Statistics = Statistics(vmd_value, rsf_value, coverage_percentage, wsp_image.num_spots, wsp_image.droplets_data)
        self.save_statistics_to_folder()
        self.create_masks()

  
    def find_overlapping_circles(self):
        self.no_overlapped_droplets = 0
        self.enumerate_image = copy.copy(self.wsp_image.blur_image)

        # iterate over each droplet and compare with all other droplets
        for droplet in self.wsp_image.droplets_data:
            cv2.putText(self.enumerate_image, f'{droplet.id}', (int(droplet.center_x-droplet.diameter), int(droplet.center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            center_y1 = droplet.center_y
            center_x1 = droplet.center_x
            r1 = int(droplet.diameter/2)
            id1 = droplet.id

            for droplet2 in self.wsp_image.droplets_data:
                id2 = droplet2.id
                center_y2 = droplet2.center_y
                center_x2 = droplet2.center_x
                r2 = int(droplet2.diameter/2)

                center_distance = np.sqrt((center_x2 - center_x1)**2 + (center_y2 - center_y1)**2)

                # if they overlap, mark it as overlapped
                if (center_distance < (r1 + r2) and id2 != id1):
                    droplet.overlappedIDs += [id2]
                    self.no_overlapped_droplets += 1
        #cv2.imwrite('images\\artificial_dataset\\numbered\\' ++ str(self.wsp_image.index) + '_groundtruth.png', self.enumerate_image)
       

    def verify_VDM(droplet_radii, vmd_value):
        check_vmd_s = 0
        check_vmd_h = 0
        equal = 0
        for i in range(len(droplet_radii)):
            if (vmd_value > droplet_radii[i]):
                check_vmd_h += 1
            if (vmd_value < droplet_radii[i]):
                check_vmd_s += 1
            if (droplet_radii[i] == 10):
                equal += 1

        print("len ", len(droplet_radii))
        print("number of droplets ", check_vmd_s, " ", check_vmd_h, " ", equal)
    
    def save_statistics_to_folder(self):
        statistics_file_path = os.path.join(path_to_statistics_gt_folder, str(self.wsp_image.filename) + '.txt')
        with open(statistics_file_path, 'w') as f:
            f.write(f"Number of droplets: {self.stats.no_droplets:d}\n")
            f.write(f"Coverage percentage: {self.stats.coverage_percentage:.2f}\n")
            f.write(f"VMD value: {self.stats.vmd_value:2f}\n")
            f.write(f"RSF value: {self.stats.rsf_value:.2f}\n")
            f.write(f"Number of overlapped droplets: {self.no_overlapped_droplets:d}\n")
            f.write(f"\nDROPLETS: no [id] ([center_x], [center_y], [diameter])\n")
            for drop in self.wsp_image.droplets_data:
                if(drop.overlappedIDs != []): f.write(f"Droplet no {drop.id} {drop.isElispe} ({drop.center_x}, {drop.center_y}, {drop.diameter}): {drop.overlappedIDs}\n")
                else: f.write(f"Droplet no {drop.id} {drop.isElispe} ({drop.center_x}, {drop.center_y}, {drop.diameter}): {drop.overlappedIDs}\n")



    def create_masks(self):
        self.mask = np.zeros_like(self.wsp_image.rectangle)
        mask_overlapped = copy.copy(self.mask)
        mask_single = copy.copy(self.mask)
        

        for drop in self.wsp_image.droplets_data:
            radius = int(drop.diameter/2)
            # single droplets
            if (drop.overlappedIDs == []):
                if drop.isElispe:
                    cv2.ellipse(mask_single, (drop.center_x, drop.center_y), (radius, radius + 5), 5, 0, 360, 255, -1)
                else:
                    cv2.circle(mask_single, (drop.center_x, drop.center_y), radius, 255, -1)
            # overlapped droplets
            else:
                if drop.isElispe:
                    cv2.ellipse(mask_overlapped, (drop.center_x, drop.center_y), (radius, radius + 5), 5, 0, 360, 255, -1)
                else:
                    cv2.circle(mask_overlapped, (drop.center_x, drop.center_y), radius, 255, -1)
        
        cv2.imwrite(os.path.join(path_to_masks_overlapped_gt_folder, str(self.wsp_image.filename) + '.png'), mask_overlapped)
        cv2.imwrite(os.path.join(path_to_masks_single_gt_folder, str(self.wsp_image.filename) + '.png'), mask_single)
            


  # def calculate_vmd(self):
    #     volumes_sorted = sorted(self.wsp_image.droplet_radii)
    #     total_volume = sum(volumes_sorted)
    #     self.cumulative_fraction = np.cumsum(volumes_sorted) / total_volume

    #     vmd_index = np.argmax(self.cumulative_fraction >= 0.5)
    #     self.vmd_value = volumes_sorted[vmd_index]

    # def calculate_coverage_percentage(self):
    #                                     # Define the acceptable range for yellow color in RGB
    #     background_lower = np.array(self.wsp_image.background_color_1, dtype=np.uint8)  # Lower bound for yellow
    #     background_upper = np.array(self.wsp_image.background_color_2, dtype=np.uint8)  # Upper bound for yellow

    #     # sum number of pixels that are part of the background
    #     not_covered_area = 0
    #     for y in range(self.wsp_image.height):
    #         for x in range(self.wsp_image.width):
    #             droplet_bgr = tuple(self.wsp_image.rectangle[y, x])

    #             # Check if the pixel falls within the yellow range
    #             isYellow = np.all([y, x] >= background_lower) and np.all([y, x] <= background_upper)
    #             # check if pixel is yellow
    #             if isYellow:
    #                 print("ahhh")
    #                 not_covered_area += 1
        
    #     # calculate percentage of paper that is coverered
    #     self.total_area = self.wsp_image.width * self.wsp_image.height
    #     self.coverage_percentage = ((self.total_area - not_covered_area) / self.total_area) * 100
    
    # def calculate_rsf(self):
    #     self.dv_one = np.argmax(self.cumulative_fraction >= 0.1)
    #     self.dv_nine = np.argmax(self.cumulative_fraction >= 0.9)

    #     self.rsf_value = (self.dv_nine - self.dv_one) / self.vmd_value



# class WSP_Statistics_Generator:
#     def __init__(self):  
#         self.total_num_droplets = 400
#         self.vmd = 14
#     def generate_radius(self):
#         droplet_vol = []
#         for i in range(self.total_num_droplets//2):
#             droplet_vol.append(np.random.randint(0, 14))
#         for i in range(self.total_num_droplets//2):
#             droplet_vol.append(np.random.randint(14, 40))
        
#         return droplet_vol
        
# stats = WSP_Statistics_Generator()
# droplet_vol = WSP_Statistics_Generator.generate_radius(stats)


# volumes_sorted = sorted(droplet_vol)
# total_volume = sum(volumes_sorted)
# cumulative_fraction = np.cumsum(volumes_sorted) / total_volume

# vmd_index = np.argmax(cumulative_fraction >= 0.5)
# vmd_value = volumes_sorted[vmd_index]

# print(vmd_value)




