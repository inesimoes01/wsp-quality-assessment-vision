import sys
import cv2
import copy

sys.path.insert(0, 'src')
from common.Util import *
from common.Variables import *
from common.Droplet import *
from common.Statistics  import * 
from segmentation.Distortion import * 
from segmentation.Calculated_Statistics import *



if __name__ == "__main__":
    # Read the command-line arguments
    file_path = str(sys.argv[1])

    file_path_parts = file_path.split("/")
    parts = file_path[len(file_path_parts)].split(".")
    filename = parts[0]
    
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    calculated:Calculated_Statistics = Calculated_Statistics(image, filename, "t", "t")
    droplets_calculated_dict = {droplet.id: droplet for droplet in calculated.droplets_data}
    stats_calculated:Statistics = calculated.stats
 
    print(stats_calculated.no_droplets)
