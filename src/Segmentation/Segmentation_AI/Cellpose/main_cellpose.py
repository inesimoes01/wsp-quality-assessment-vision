import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.color import rgb2gray

mpl.rcParams['figure.dpi'] = 300

from urllib.parse import urlparse
from cellpose import models, core, io

use_GPU = core.use_gpu()

files = ['C:\\Users\\mines\\Desktop\\github\\vision-quality-assessment-opencv\\data\\artificial_dataset_2\\wsp\\image\\1.png', 'C:\\Users\\mines\\Desktop\\github\\vision-quality-assessment-opencv\\data\\artificial_dataset\\wsp\\image\\458.png']

imgs = [rgb2gray(skimage.io.imread(f)) for f in files]
nimg = len(imgs)
imgs_2D = imgs[:-1]

model = models.Cellpose(gpu = use_GPU, model_type = 'cyto3')

channels = [[0, 0], [0, 0]]
diameter = 10

masks, flows, styles, diams = model.eval(imgs_2D, diameter=diameter, flow_threshold=None, channels=channels)
io.save_masks(imgs_2D, masks, flows, files, png=True, savedir = "C:\\Users\\mines\\Desktop\\github\\vision-quality-assessment-opencv\\src\\Segmentation_AI\\Cellpose", save_outlines = True, save_txt = True)


