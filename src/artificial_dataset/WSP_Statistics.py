import numpy as np
import cv2

from paths import path_to_statistics_folder

class WSP_Statistics:
    def __init__(self, wsp_image):
        self.wsp_image = wsp_image
        self.calculate_vmd()
        self.calculate_coverage_percentage()
        self.calculate_number_of_droplets_in_total_area()
        self.calculate_rsf()
        self.save_statistics_to_folder()
        

    def calculate_vmd(self):
        volumes_sorted = sorted(self.wsp_image.droplet_radii)
        total_volume = sum(volumes_sorted)
        self.cumulative_fraction = np.cumsum(volumes_sorted) / total_volume

        vmd_index = np.argmax(self.cumulative_fraction >= 0.5)
        self.vmd_value = volumes_sorted[vmd_index]

    def calculate_coverage_percentage(self):
        # sum number of pixels that are part of the background
        not_covered_area = 0
        for y in range(self.wsp_image.height):
            for x in range(self.wsp_image.width):
                droplet_bgr = tuple(self.wsp_image.rectangle[y, x])
                
                # check if pixel is yellow
                if tuple(map(lambda i, j: i - j, droplet_bgr, self.wsp_image.background_color)) == (0, 0, 0):
                    not_covered_area += 1
        
        # calculate percentage of paper that is coverered
        self.total_area = self.wsp_image.width * self.wsp_image.height
        self.coverage_percentage = ((self.total_area - not_covered_area) / self.total_area) * 100
    
    def calculate_number_of_droplets_in_total_area(self):
        self.droplets_per_area = self.wsp_image.num_spots / self.total_area
       
    def calculate_rsf(self):
        self.dv_one = np.argmax(self.cumulative_fraction >= 0.1)
        self.dv_nine = np.argmax(self.cumulative_fraction >= 0.9)

        self.rsf_value = (self.dv_nine - self.dv_one) / self.vmd_value

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
        statistics_file_path = path_to_statistics_folder + '\statistics_' + self.wsp_image.today_date + '_' + str(self.wsp_image.index) + '.txt'
        with open(statistics_file_path, 'w') as f:
            f.write(f"Number of droplets: {self.wsp_image.num_spots:d}\n")
            f.write(f"Coverage percentage: {self.coverage_percentage:.2f}%\n")
            f.write(f"Number of droplets per area: {self.droplets_per_area:.10f}\n")
            f.write(f"VMD value: {self.vmd_value:d}\n")
            f.write(f"RSF value: {self.rsf_value:.2f}\n")
            

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




