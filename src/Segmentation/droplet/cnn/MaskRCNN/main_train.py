import os
import sys

ROOT_DIR = os.path.abspath("")

sys.path.insert(0, 'src')
import Common.config as path_configurations
import Segmentation.droplet.cnn.MaskRCNN.custom_mrcnn_classes as custom_mrcnn_classes


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config as Config
from mrcnn import model as modellib, utils


# path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(path_configurations.MODELS_MRCNN_DIR, "mask_rcnn_coco.h5")

# directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(path_configurations.MODELS_MRCNN_DIR, "logs_2")

DATASET_PATH = os.path.join("data\\droplets\\synthetic_dataset_normal_droplets\\mrcnn")


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = custom_mrcnn_classes.CustomDataset()
    dataset_train.load_custom(DATASET_PATH, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = custom_mrcnn_classes.CustomDataset()
    dataset_val.load_custom(DATASET_PATH, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
                # learning_rate=config.LEARNING_RATE,
                # epochs=250,
                # layers='heads')
                
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads', #layers='all', 
                # augmentation = imgaug.augmenters.Sequential([ 
                # imgaug.augmenters.Fliplr(1), 
                # imgaug.augmenters.Flipud(1), 
                # imgaug.augmenters.Affine(rotate=(-45, 45)), 
                # imgaug.augmenters.Affine(rotate=(-90, 90)), 
                # imgaug.augmenters.Affine(scale=(0.5, 1.5)),
                # imgaug.augmenters.Crop(px=(0, 10)),
                # imgaug.augmenters.Grayscale(alpha=(0.0, 1.0)),
                # imgaug.augmenters.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # imgaug.augmenters.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                # imgaug.augmenters.Invert(0.05, per_channel=True), # invert color channels
                # imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                
                # ])
                )
				

'''
 this augmentation is applied consecutively to each image. In other words, for each image, the augmentation apply flip LR,
 and then followed by flip UD, then followed by rotation of -45 and 45, then followed by another rotation of -90 and 90,
 and lastly followed by scaling with factor 0.5 and 1.5. '''
	
    
# Another way of using imgaug    
# augmentation = imgaug.Sometimes(5/6,aug.OneOf(
                                            # [
                                            # imgaug.augmenters.Fliplr(1), 
                                            # imgaug.augmenters.Flipud(1), 
                                            # imgaug.augmenters.Affine(rotate=(-45, 45)), 
                                            # imgaug.augmenters.Affine(rotate=(-90, 90)), 
                                            # imgaug.augmenters.Affine(scale=(0.5, 1.5))
                                             # ]
                                        # ) 
                                   # )
                                    
				
config = custom_mrcnn_classes.CustomConfigDroplet()
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH

# download weights file
if not os.path.exists(weights_path):
  utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

train(model)			