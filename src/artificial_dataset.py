import numpy as np
import cv2
import os
from datetime import datetime

### PATHS
path_main = 'images\\artificial_images'
path_to_images_folder = 'images\\artificial_images\image'
path_to_statistics_folder = 'images\\artificial_images\statistic'

### VALUES
num_wsp = 5
num_spots = 400
max_radius = 14

def generate_one_wsp(i):
    # yellow rectangle for background
    width, height = 800, 400  
    yellow_color = (97, 225, 243)
    rectangle = np.full((height, width, 3), yellow_color, dtype=np.uint8)

    # possible colors for the droplets
    possible_colors = []
    possible_colors.extend(interpolate_color((29, 33, 52), (61, 42, 64), 20))
    possible_colors.extend(interpolate_color((89, 8, 37), (172, 4, 46), 20))

    # generate random spots with colors from the list
    droplets_data = []
    for _ in range(num_spots):
        spot_color = possible_colors[np.random.randint(0, len(possible_colors))]
        spot_radius = np.random.randint(1, max_radius) 
        center_x = np.random.randint(spot_radius, width - spot_radius)
        center_y = np.random.randint(spot_radius, height - spot_radius)
        cv2.circle(rectangle, (center_x, center_y), spot_radius, spot_color, -1)

        droplets_data.append({
            'color': spot_color,
            'radius': spot_radius,
            'center_x': center_x,
            'center_y': center_y
        })


    # calculating statistics
    total_area = width * height
    covered_area = sum(np.pi * d['radius']**2 for d in droplets_data)
    coverage_percentage = (covered_area / total_area) * 100
    droplets_per_area = num_spots / total_area
    droplet_radii = [d['radius'] for d in droplets_data]

    # save image
    today_date = str(datetime.now().date())
    cv2.imwrite(path_to_images_folder + '\wsp_' + today_date + '_' + str(i) + '.png', rectangle)

    # save statistics
    statistics_file_path = path_to_statistics_folder + '\statistics_' + today_date + '_' + str(i) + '.txt'
    with open(statistics_file_path, 'w') as f:
        f.write(f"Coverage percentage: {coverage_percentage:.2f}%\n")
        f.write(f"Number of droplets per area: {droplets_per_area:.10f}\n")
        f.write("Droplet diameter distribution: ")
        for radius in droplet_radii:
            f.write(f"{radius}, ")

    # # display image
    # cv2.imshow('Water Sensitive Paper Demo', rectangle)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def interpolate_color(color1, color2, steps):
    # extract individual BGR components
    b1, g1, r1 = color1
    b2, g2, r2 = color2
    
    # calculate step size for each component
    b_step = (b2 - b1) / steps
    g_step = (g2 - g1) / steps
    r_step = (r2 - r1) / steps
    
    # generate interpolated colors
    interpolated_colors = []
    for i in range(steps):
        b = round(b1 + i * b_step)
        g = round(g1 + i * g_step)
        r = round(r1 + i * r_step)
        interpolated_colors.append((b, g, r))
    
    return interpolated_colors

def delete_old_dataset():
    for filename in os.listdir(path_to_images_folder):
        file_path_image = os.path.join(path_to_images_folder, filename)
        if os.path.isfile(file_path_image):
            os.remove(file_path_image)

    for filename in os.listdir(path_to_statistics_folder):
        file_path_statistic = os.path.join(path_to_statistics_folder, filename)
        if os.path.isfile(file_path_statistic):
            os.remove(file_path_statistic)

def create_folders():
    if not os.path.exists(path_main):
        os.makedirs(path_main)
    if not os.path.exists(path_to_images_folder):
        os.makedirs(path_to_images_folder)
    if not os.path.exists(path_to_statistics_folder):
        os.makedirs(path_to_statistics_folder)
    

# manage folders
create_folders()
delete_old_dataset()

# generate images
for i in range(num_wsp):
    generate_one_wsp(i)