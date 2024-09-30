import warnings
warnings.filterwarnings('ignore')
import os
import sys
import numpy as np
import skimage.draw
import cv2
import random
import math
import re
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import pandas as pd
import skimage

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils

import custom_mrcnn_classes

ROOT_DIR = os.getcwd()

sys.path.append(ROOT_DIR)

# trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DATASET_PATH = "data\\droplets\\synthetic_dataset_droplets\\mrcnn"
MODEL_PATH = 'models\\droplets\\mrcnn\\logs_2\\droplet_dataset20240906T1631\\mask_rcnn_droplet_dataset_0100.h5'

DATASET_TEST_PATH = os.path.join(DATASET_PATH, "test")

def draw_precision_recall_curve( gt_bbox, gt_class_id, gt_mask, r):
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                            r['rois'], r['class_ids'], r['scores'], r['masks'])
    visualize.plot_precision_recall(AP, precisions, recalls)
    return precisions, recalls


def draw_confusion_matrix(config, dataset):

    gt_tot = np.array([])
    pred_tot = np.array([])

    #mAP list
    mAP_ = []

    #compute gt_tot, pred_tot and mAP for each image in the test dataset
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id)#, #use_mini_mask=False)
        info = dataset.image_info[image_id]

        # Run the model
        results = model.detect([image])
        r = results[0]
        
        #compute gt_tot and pred_tot
        gt, pred = utils.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
        gt_tot = np.append(gt_tot, gt)
        pred_tot = np.append(pred_tot, pred)
        
        #precision_, recall_, AP_ 
        AP_, precision_, recall_, overlap_ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                            r['rois'], r['class_ids'], r['scores'], r['masks'])
        #check if the vectors len are equal
        print("the actual len of the gt vect is : ", len(gt_tot))
        print("the actual len of the pred vect is : ", len(pred_tot))
        
        mAP_.append(AP_)
        print("Average precision of this image : ", AP_)
        print("The actual mean average precision for the whole images", sum(mAP_)/len(mAP_))
        #print("Ground truth object : "+dataset.class_names[gt])


    gt_tot=gt_tot.astype(int)
    pred_tot=pred_tot.astype(int)
    #save the vectors of gt and pred
    save_dir = "output"
    gt_pred_tot_json = {"gt_tot" : gt_tot, "pred_tot" : pred_tot}
    df = pd.DataFrame(gt_pred_tot_json)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_json(os.path.join(save_dir,"gt_pred_test.json"))

    tp,fp,fn=utils.plot_confusion_matrix_from_data(gt_tot,pred_tot,columns=["bg","droplet"] ,fz=18, figsize=(20,20), lw=0.5)

    return tp, fp, fn, gt_bbox, gt_class_id, gt_mask, r

inference_config = custom_mrcnn_classes.InferenceConfig()

# recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=DEFAULT_LOGS_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#model_path = model.find_last()


# load trained weights
print("Loading weights from ", MODEL_PATH)
model.load_weights(MODEL_PATH, by_name=True)


# load datasets
dataset_train = custom_mrcnn_classes.CustomDataset()
dataset_train.load_custom(DATASET_PATH, "train")
dataset_train.prepare()
dataset_val = custom_mrcnn_classes.CustomDataset()
dataset_val.load_custom(DATASET_PATH, "val")
dataset_val.prepare()

# image_paths = []
# for filename in os.listdir(os.path.join(DATASET_PATH, "test")):
#     if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
#         image_paths.append(os.path.join(DATASET_TEST_PATH, filename))

# for image_path in image_paths:
#     img = skimage.io.imread(image_path)
#     img_arr = np.array(img)
#     results = model.detect([img_arr])
#     r = results[0]
    #visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                #dataset_val.class_names, r['scores'], figsize=(5,5))
    

tp, fp, fn, gt_bbox, gt_class_id, gt_mask, r = draw_confusion_matrix(inference_config, dataset_val)


precisions, recalls = draw_precision_recall_curve(gt_bbox, gt_class_id, gt_mask, r)

import pandas as pd


# Create a DataFrame
df = pd.DataFrame({'Precision': precisions, 'Recall': recalls})

# Save the DataFrame to a CSV file
df.to_csv('precision_recall.csv', index=False)

print("CSV file saved successfully!")


