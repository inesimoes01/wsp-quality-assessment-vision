import cv2

from segmentation.Distortion import *
from segmentation.Calculated_Statistics import *
from common.Droplet import *
from common.Statistics import *
from common.Util import *

def main(file_path, width, height):
    file_path = os.path.normpath(file_path)

    # get name of the file
    file_path_parts = file_path.split("/")
    parts = file_path[len(file_path_parts)].split(".")
    filename = parts[0]

    try:    
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        # generate undistorted image
        dist = Distortion(image, filename, save_photo=False)
        if dist.noPaper: return        

        calc_stats = Calculated_Statistics(dist.undistorted_image, filename, save_image=False)

        return calc_stats.stats.no_droplets, calc_stats.stats.coverage_percentage, calc_stats.stats.rsf_value, calc_stats.stats.vmd_value

    except Exception as e:
        print("An error occurred:", e)