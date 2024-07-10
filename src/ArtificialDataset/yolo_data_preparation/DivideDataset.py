import os
import random
import shutil

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.1):
    # Create destination directories if they don't exist
    train_dir_im = os.path.join(dest_dir, 'images', 'train')
    val_dir_im = os.path.join(dest_dir, 'images', 'val')
    test_dir_im = os.path.join(dest_dir, 'images', 'test')
    train_dir_l = os.path.join(dest_dir, 'labels', 'train')
    val_dir_l = os.path.join(dest_dir, 'labels', 'val')
    test_dir_l = os.path.join(dest_dir, 'labels', 'test')
    
    
    for directory in [train_dir_im, val_dir_im, test_dir_im]:
        os.makedirs(directory, exist_ok=True)
    for directory in [train_dir_l, val_dir_l, test_dir_l]:
        os.makedirs(directory, exist_ok=True)
    
    # List all images in the source directory
    images = os.listdir(os.path.join(source_dir, 'images'))
    
    # Shuffle the list of images
    random.shuffle(images)
    
    # Calculate split sizes
    num_images = len(images)
    num_train = int(train_ratio * num_images)
    num_val = int(val_ratio * num_images)
    num_test = num_images - num_train - num_val
    
    # Assign images to each split
    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]
    
    # Function to copy images and labels
    def copy_images_and_labels(image_list, split_dir_im, split_dir_l):
        for image_name in image_list:
            filename = os.path.splitext(image_name)[0]
            # parts = image_name.split(".")
            # filename = parts[0]
        
            image_path = os.path.join(source_dir, 'images', filename + ".jpg")
            label_name = image_name.replace('.png', '.txt')  # Assuming labels are txt files
            label_path = os.path.join(source_dir, 'labels', filename + ".txt")
            
            # Copy image
            shutil.copy(image_path, os.path.join(split_dir_im, filename+ ".jpg"))
            
            # Copy label
            shutil.copy(label_path, os.path.join(split_dir_l, filename+ ".txt"))
    
    # Copy images and labels to train directory
    copy_images_and_labels(train_images, train_dir_im, train_dir_l)
    
    # Copy images and labels to validation directory
    copy_images_and_labels(val_images, val_dir_im, val_dir_l)
    
    # Copy images and labels to test directory
    copy_images_and_labels(test_images, test_dir_im, test_dir_l)

# Example usage:
source_directory = 'C:\\Users\\mines\\Desktop\\github\\vision-quality-assessment-opencv\\data\\rectangle_dataset\\all'
destination_directory = "C:\\Users\\mines\\Desktop\\github\\vision-quality-assessment-opencv\\data\\rectangle_dataset\\yolo"

split_dataset(source_directory, destination_directory)

