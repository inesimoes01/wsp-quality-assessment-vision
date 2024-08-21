import os
import shutil
import sys
import cv2
import numpy as np
import random
from shutil import copyfile
from PIL import Image
from sklearn.model_selection import KFold

random.seed(42)

sys.path.insert(0, 'src')
import common.config as config


def prepare_dataset_for_yolo():
    original_dataset_folder = "data\\droplets\\synthetic_dataset_normal_droplets\\raw"
    output_squares_folder = "data\\droplets\\synthetic_dataset_normal_droplets\\cropped"
    separated_squares_folder = "data\\droplets\\synthetic_dataset_normal_droplets\\divided"

    print("Dividing images into squares...")
    #divide_images_into_squares_with_yolo_annotations(original_dataset_folder, output_squares_folder)

    print("Applying Kfold...")
    split_dataset_normal(output_squares_folder, separated_squares_folder)

def split_dataset_normal(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.1):

    train_dir_im = os.path.join(dest_dir, 'train', 'image')
    val_dir_im = os.path.join(dest_dir, 'val', 'image')
    test_dir_im = os.path.join(dest_dir, 'test', 'image')
    train_dir_l = os.path.join(dest_dir, 'train', 'label')
    val_dir_l = os.path.join(dest_dir, 'val', 'label')
    test_dir_l = os.path.join(dest_dir, 'test', 'label')
    

    for directory in [train_dir_im, val_dir_im, test_dir_im]:
        os.makedirs(directory, exist_ok=True)
    for directory in [train_dir_l, val_dir_l, test_dir_l]:
        os.makedirs(directory, exist_ok=True)
    
    # List all image in the source directory
    image = os.listdir(os.path.join(source_dir, config.DATA_GENERAL_IMAGE_FOLDER_NAME))
    
    # Shuffle the list of image
    random.shuffle(image)
    
    # Calculate split sizes
    num_image = len(image)
    num_train = int(train_ratio * num_image)
    num_val = int(val_ratio * num_image)
    num_test = num_image - num_train - num_val
    
    # Assign image to each split
    train_image = image[:num_train]
    val_image = image[num_train:num_train + num_val]
    test_image = image[num_train + num_val:]
    
    # Function to copy image and label
    def copy_image_and_label(image_list, split_dir_im, split_dir_l):
        for image_name in image_list:
            filename = os.path.splitext(image_name)[0]
        
            image_path = os.path.join(source_dir, config.DATA_GENERAL_IMAGE_FOLDER_NAME, filename + ".png")
            label_name = image_name.replace('.png', '.txt')  # Assuming label are txt files
            label_path = os.path.join(source_dir, config.DATA_GENERAL_LABEL_FOLDER_NAME, filename + ".txt")
            
            # Copy image
            shutil.copy(image_path, os.path.join(split_dir_im, filename+ ".png"))
            
            # Copy label
            shutil.copy(label_path, os.path.join(split_dir_l, filename+ ".txt"))
    
    # Copy image and label to train directory
    copy_image_and_label(train_image, train_dir_im, train_dir_l)
    
    # Copy image and label to validation directory
    copy_image_and_label(val_image, val_dir_im, val_dir_l)
    
    # Copy image and label to test directory
    copy_image_and_label(test_image, test_dir_im, test_dir_l)


def split_dataset_kfold(dataset_folder, output_dir, n_splits=5):

    image_dir = os.path.join(dataset_folder, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
    label_dir = os.path.join(dataset_folder, config.DATA_GENERAL_LABEL_FOLDER_NAME)

    images = []
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.png', '.txt'))
            if os.path.exists(label_path):
                images.append(image_path)
                labels.append(label_path)

    # create output directories if they don't exist
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(split_dir, config.DATA_GENERAL_IMAGE_FOLDER_NAME), exist_ok=True)
        os.makedirs(os.path.join(split_dir, config.DATA_GENERAL_LABEL_FOLDER_NAME), exist_ok=True)

    fold_size = len(images) // n_splits
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    folds = []
    for i in range(n_splits):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))



    fold = 1
    for train_indices, test_indices in folds:
 
        print(f"Processing Fold {fold}")
        
        # Test set for this fold
        
        test_images = [images[j] for j in test_indices]
        test_labels = [labels[j] for j in test_indices]
        
        # Training set (all data not in the test set)
        #train_indices = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        train_images = [images[j] for j in train_indices]
        train_labels = [labels[j] for j in train_indices]
        
        # Split the training set further into training and validation sets
        val_split = int(0.2 * len(train_images))
        val_images = train_images[:val_split]
        val_labels = train_labels[:val_split]
        train_images = train_images[val_split:]
        train_labels = train_labels[val_split:]
        
        # Save the data for this fold
        _save_split(train_images, train_labels, output_dir, f'fold_{fold}/train')
        _save_split(val_images, val_labels, output_dir, f'fold_{fold}/val')
        _save_split(test_images, test_labels, output_dir, f'fold_{fold}/test')
        
        fold += 1
        



def divide_images_into_squares_with_yolo_annotations(original_dataset_folder, output_folder, square_size=320):
    """
    divide image into squares of 320 x 320. if there is not enough pixels it creates a rectangle with the remaining
    it expects labels with yolo format
    """
    images_folder = os.path.join(original_dataset_folder, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
    labels_folder = os.path.join(original_dataset_folder, config.DATA_GENERAL_LABEL_FOLDER_NAME)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(os.path.join(output_folder, config.DATA_GENERAL_IMAGE_FOLDER_NAME)):
        os.makedirs(os.path.join(output_folder, config.DATA_GENERAL_IMAGE_FOLDER_NAME))
    if not os.path.exists(os.path.join(output_folder, config.DATA_GENERAL_LABEL_FOLDER_NAME)):
        os.makedirs(os.path.join(output_folder, config.DATA_GENERAL_LABEL_FOLDER_NAME))
    
    # list all images in images_folder
    image_files = os.listdir(images_folder)
    
    for image_file in image_files:
        parts = image_file.split(".")
        filename = parts[0]
    
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, image_file.replace(".png", ".txt")) 
        
        if not os.path.exists(label_path):
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            continue
    
        labels = _read_labels(label_path)
        square_count = 0
     
        height, width, _ = image.shape

        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                # extract the square
                square = image[y:y+square_size, x:x+square_size]

                original_square_shape = square.shape

                # ensure the square is the size expected

                # if square.shape[0] != square_size or square.shape[1] != square_size:
                #     original_square_shape = square.shape
                #     square = cv2.resize(square, (square_size, square_size))

                adjusted_labels = []

                for line in labels:
                    data = line.strip().split()
                    if len(data) < 2:
                        continue
                    
                    coordinates = []
                    for i in range(1, len(data), 2):
                        x_label = (float(data[i])) * width
                        y_label = (float(data[i + 1])) * height

                        if (x <= x_label <= (x + square_size)) and (y <= y_label <= (y + square_size)):
                            x_rel = (x_label - x) / original_square_shape[1] if original_square_shape[1] != square_size else (x_label - x) / square_size
                            y_rel = (y_label - y) / original_square_shape[0] if original_square_shape[0] != square_size else (y_label - y) / square_size
                            coordinates.append(x_rel)
                            coordinates.append(y_rel)
                    
                    if len(coordinates) > 0:
                        adjusted_labels.append(coordinates)
                
                if len(adjusted_labels) > 0:
                    # Save the square
                    square_filename = f"{filename}_{square_count}.png"
                    square_path = os.path.join(output_folder, config.DATA_GENERAL_IMAGE_FOLDER_NAME, square_filename)
                    cv2.imwrite(square_path, square)

                    label_filename = f"{filename}_{square_count}.txt"
                    cropped_label_path = os.path.join(output_folder, config.DATA_GENERAL_LABEL_FOLDER_NAME, label_filename)    
                    with open(cropped_label_path, 'w') as f:
                        for coords in adjusted_labels:
                            coord_str = " ".join([f"{p}" for p in coords])
                            f.write(f"{0} {coord_str}\n")
                    
                    square_count += 1

def _read_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    return lines


def _save_split(images, labels, output_dir, split_name):
    image_output_dir = os.path.join(output_dir, split_name, config.DATA_GENERAL_IMAGE_FOLDER_NAME)
    label_output_dir = os.path.join(output_dir, split_name, config.DATA_GENERAL_LABEL_FOLDER_NAME)
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    
    for image, label in zip(images, labels):
        copyfile(image, os.path.join(image_output_dir, os.path.basename(image)))
        copyfile(label, os.path.join(label_output_dir, os.path.basename(label)))




prepare_dataset_for_yolo()