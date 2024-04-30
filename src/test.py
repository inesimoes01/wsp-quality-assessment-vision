from exif import Image
import numpy as np
import pandas as pd
from datetime import datetime as dt

img_path = "images\\artificial_dataset\\image\\0.png"
import os
import time

# GObject-based wrapper around the Exiv2 library.
# sudo apt-get install gir1.2-gexiv2-0.4
from gi.repository import GExiv2


def fix_image_dates(img_path):
    t = os.path.getctime(img_path)
    ctime = time.strftime('%Y:%m:%d %H:%M:%S', time.localtime(t))
    exif = GExiv2.Metadata(img_path)
    exif['Exif.Image.DateTime'] = ctime
    exif['Exif.Photo.DateTimeDigitized'] = ctime
    exif['Exif.Photo.DateTimeOriginal'] = ctime
    exif.save_file()
    os.utime(img_path, (t, t))


if __name__ == '__main__':
    dir = '.'
    for root, dirs, file_names in os.walk(dir):
        for file_name in file_names:
            if file_name.lower().endswith(('jpg', 'png')):
                img_path = os.path.join(root, file_name)
                fix_image_dates(img_path)