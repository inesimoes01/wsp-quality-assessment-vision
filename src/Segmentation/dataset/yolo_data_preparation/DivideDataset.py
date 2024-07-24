import os
import random
import shutil

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.1):

    # Create destination directories if they don't exist
    train_dir_im = os.path.join(dest_dir, 'image', 'train')
    val_dir_im = os.path.join(dest_dir, 'image', 'val')
    test_dir_im = os.path.join(dest_dir, 'image', 'test')
    train_dir_l = os.path.join(dest_dir, 'label', 'train')
    val_dir_l = os.path.join(dest_dir, 'label', 'val')
    test_dir_l = os.path.join(dest_dir, 'label', 'test')
    

    for directory in [train_dir_im, val_dir_im, test_dir_im]:
        os.makedirs(directory, exist_ok=True)
    for directory in [train_dir_l, val_dir_l, test_dir_l]:
        os.makedirs(directory, exist_ok=True)
    
    # List all image in the source directory
    image = os.listdir(os.path.join(source_dir, 'image'))
    
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
        
            image_path = os.path.join(source_dir, 'image', filename + ".png")
            label_name = image_name.replace('.png', '.txt')  # Assuming label are txt files
            label_path = os.path.join(source_dir, 'label', filename + ".txt")
            
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

# Example usage:
source_directory = 'C:\\Users\\mines\\Desktop\\github\\vision-quality-assessment-opencv\\data\\artificial_dataset_3\\processed\\all'
destination_directory = 'C:\\Users\\mines\\Desktop\\github\\vision-quality-assessment-opencv\\data\\artificial_dataset_3\\processed\\yolo_data'

split_dataset(source_directory, destination_directory)

