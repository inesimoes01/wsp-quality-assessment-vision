import os
from datetime import datetime

from Colors import Colors
from WSP_Statistics import WSP_Statistics
from WSP_Image import WSP_Image
import sys

sys.path.insert(0, 'src/others')
from paths import *
from util import *

### VALUES
num_wsp = 5

# manage folders
create_folders(path_main_dataset)
create_folders(path_to_images_folder)
create_folders(path_to_statistics_folder)
create_folders(path_to_outputs_folder)
create_folders(path_to_separation_folder)

delete_old_files(path_to_images_folder)
delete_old_files(path_to_statistics_folder)
delete_folder_contents(path_to_outputs_folder)
delete_old_files(path_to_separation_folder)


colors = Colors()
today_date = str(datetime.now().date())

# generate images
for i in range(num_wsp):
    print(i)
    WSP_Statistics(WSP_Image(i, colors.droplet_color, colors.background_color, today_date))
