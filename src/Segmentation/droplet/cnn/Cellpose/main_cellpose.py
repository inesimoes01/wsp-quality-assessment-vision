import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.color import rgb2gray

mpl.rcParams['figure.dpi'] = 300

def load_and_convert_image(file_path):
    img = skimage.io.imread(file_path)
    if img.shape[-1] == 4:  # If the image has an alpha channel
        img = skimage.color.rgba2rgb(img)  # Convert RGBA to RGB
    return skimage.color.rgb2gray(img) 

from urllib.parse import urlparse
from cellpose import models, core, io

use_GPU = core.use_gpu()

files = ['results\\cellpose\\real_dataset_to_annotate\\Referencia_350_05m_cut.png', 
         'results\\cellpose\\real_dataset_to_annotate\\2_Z2_cut.jpg']

imgs = [load_and_convert_image(f) for f in files]
nimg = len(imgs)
imgs_2D = imgs[:-1]

model = models.Cellpose(gpu = use_GPU, model_type = 'cyto3')

channels = [[0, 0], [0, 0]]
diameter = 12
masks, flows, styles, diams = model.eval(imgs_2D, diameter=diameter, flow_threshold=None, channels=channels)
io.save_masks(imgs_2D, masks, flows, files, png=True, savedir = "results\\cellpose\\pre_annotations", save_outlines = True, save_txt = True)


