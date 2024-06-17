import os 
import shutil
from matplotlib import pyplot as plt 
import csv
import cv2
from Droplet import *

def delete_old_files(file_path):
    for filename in os.listdir(file_path):
        path = os.path.join(file_path, filename)
        if os.path.isfile(path):
            os.remove(path)

def delete_folder_contents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def create_folders(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def save_dropletinfo_csv(file_path, droplet_info:list[Droplet]):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["DropletID", "isElipse", "CenterX", "CenterY", "Diameter", "OverlappedDropletsID"])
        for drop in droplet_info:
            row = [drop.id, drop.isElipse, drop.center_x, drop.center_y, drop.diameter, str(drop.overlappedIDs)]
            writer.writerow(row)

def mask_to_label(path_mask, path_labels, classid):
    # load the binary mask and get its contours
    mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # convert the contours to polygons
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 5:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append(polygon)
    
    # print the polygons
    with open('{}.txt'.format(path_labels[:-4]), 'a') as f:
        for polygon in polygons:
            f.write(f"{classid} {' '.join(map(str, polygon))}\n")

def plotThreeImages(image1, image2, image3):
    plt.close('all')
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    axes[0].imshow(image1)
    #axes[0].axis('off')
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")

    axes[1].imshow(image2)
    #axes[1].axis('off')
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")
    
    axes[2].imshow(image3)
    #axes[1].axis('off')
    axes[2].set_xlabel("X (pixels)")
    axes[2].set_ylabel("Y (pixels)")

    plt.show()

def plotALOTImages(images):
    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()



def plotTwoImages(image1, image2):
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(image1)
    #axes[0].axis('off')
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")

    axes[1].imshow(image2)
    #axes[1].axis('off')
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")

    plt.show()

def plotFourImages(image1, image2, image3, image4):

    images = [image1, image2, image3, image4]
 
    plt.figure(figsize=(10, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # plt.close('all')
    # fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    # axes[0].imshow(image1)
    # #axes[0].axis('off')
    # axes[0].set_xlabel("X (pixels)")
    # axes[0].set_ylabel("Y (pixels)")

    # axes[1].imshow(image2)
    # #axes[1].axis('off')
    # axes[1].set_xlabel("X (pixels)")
    # axes[1].set_ylabel("Y (pixels)")

    # axes[2].imshow(image3)
    # #axes[1].axis('off')
    # axes[2].set_xlabel("X (pixels)")
    # axes[1].set_ylabel("Y (pixels)")

    # axes[3].imshow(image4)
    # #axes[1].axis('off')
    # axes[3].set_xlabel("X (pixels)")
    # axes[3].set_ylabel("Y (pixels)")


    # plt.show()



