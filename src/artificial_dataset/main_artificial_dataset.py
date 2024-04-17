import os
from datetime import datetime
import time

from Colors import Colors
from WSP_Statistics import WSP_Statistics
from WSP_Image import WSP_Image
import sys


sys.path.insert(0, 'src/common')
from Variables import *
from Util import *

# manage folders
create_folders(path_main_dataset)
create_folders(path_to_images_folder)
create_folders(path_to_statistics_gt_folder)
create_folders(path_to_statistics_c_folder)
create_folders(path_to_outputs_folder)
create_folders(path_to_separation_folder)
create_folders(path_to_numbered_folder)
create_folders(path_to_masks_overlapped_folder)
create_folders(path_to_masks_single_circle_folder)
create_folders(path_to_masks_single_ellipse_folder)

delete_folder_contents(path_to_images_folder)
delete_folder_contents(path_to_statistics_gt_folder)
delete_folder_contents(path_to_statistics_c_folder)
delete_folder_contents(path_to_outputs_folder)
delete_folder_contents(path_to_separation_folder)
delete_folder_contents(path_to_inputs_folder)
delete_folder_contents(path_to_numbered_folder)
delete_folder_contents(path_to_masks_overlapped_folder)
delete_folder_contents(path_to_masks_single_circle_folder)
delete_folder_contents(path_to_masks_single_ellipse_folder)

colors = Colors()
today_date = str(datetime.now().date())

# generate images
for i in range(num_wsp):
    start_time = time.time()

    print("Creating image number ", i)
    wsp_image = WSP_Image(i, colors, today_date)
    print("Image created. Calculating statistics...")
    WSP_Statistics(wsp_image, colors)

    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken:", elapsed_time, "seconds")

    
## Background:  188.33333333333334