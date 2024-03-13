import numpy as np
import cv2
from datetime import datetime

from paths import path_to_images_folder, path_to_statistics_folder

def verify_VDM(droplet_radii, vmd_value):
    check_vmd_s = 0
    check_vmd_h = 0
    ahhh = 0
    for i in range(len(droplet_radii)):
        if (vmd_value > droplet_radii[i]):
            check_vmd_h += 1
        if (vmd_value < droplet_radii[i]):
            check_vmd_s += 1
        if (droplet_radii[i] == 10):
            ahhh += 1

    print("len ", len(droplet_radii))
    print("number of droplets ", check_vmd_s, " ", check_vmd_h, " ", ahhh)


def calculate_statistics(i, width, height, image, num_spots, droplets_data, background_color):
    # sum number of pixeis that are part of the background
    not_covered_area = 0
    for y in range(height):
        for x in range(width):
            droplet_bgr = tuple(image[y, x])
            
            # check if pixel is yellow
            if tuple(map(lambda i, j: i - j, droplet_bgr, background_color)) == (0, 0, 0):
                not_covered_area += 1
    
    # calculate percentage of paper that is coverered
    total_area = width * height
    coverage_percentage = ((total_area - not_covered_area) / total_area) * 100

    # calculate the number of droplets in the total area
    droplets_per_area = num_spots / total_area

    # list of the radius of all the droplets
    droplet_radii = [d['radius'] for d in droplets_data]

    vmd_value = calculate_vmd(droplet_radii)

    save_statistics_to_folder(i, coverage_percentage, droplets_per_area, droplet_radii, image)
    #return coverage_percentage, droplets_per_area, droplet_radii, vmd_value



def save_statistics_to_folder(i, coverage_percentage, droplets_per_area, droplet_radii, image):
    # calculate statistics
    #coverage_percentage, droplets_per_area, droplet_radii = calculate_statistics(width, height, image, num_spots, droplets_data)

    # save image
    today_date = str(datetime.now().date())
    cv2.imwrite(path_to_images_folder + '\wsp_' + today_date + '_' + str(i) + '.png', image)


    # save statistics
    statistics_file_path = path_to_statistics_folder + '\statistics_' + today_date + '_' + str(i) + '.txt'
    with open(statistics_file_path, 'w') as f:
        f.write(f"Coverage percentage: {coverage_percentage:.2f}%\n")
        f.write(f"Number of droplets per area: {droplets_per_area:.10f}\n")
        f.write("Droplet diameter distribution: ")
        for radius in droplet_radii:
            f.write(f"{radius}, ")

def calculate_vmd(droplet_radii):
    volumes_sorted = sorted(droplet_radii)
    total_volume = sum(volumes_sorted)
    cumulative_fraction = np.cumsum(volumes_sorted) / total_volume
   
    vmd_index = np.argmax(cumulative_fraction >= 0.5)
    vmd_value = volumes_sorted[vmd_index]

    print("Volume Median Diameter (VMD):", vmd_value)
