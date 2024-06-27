import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import random

def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def generate_synthetic_image(width, height, num_speckles):
    # Create a yellow background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    image[:, :, 0] = 0  # Remove the blue channel
    image[:, :, 1] = 255  # Full intensity in the green channel
    image[:, :, 2] = 255  # Full intensity in the red channel

    speckles = []
    for _ in range(num_speckles):
        # Randomly generate speckle properties
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        radius = np.random.randint(1, 5)
        
        # Randomize speckle shape
        if random.choice([True, False]):
            axes = (radius, np.random.randint(1, 5))
            angle = np.random.randint(0, 360)
            cv2.ellipse(image, (x, y), axes, angle, 0, 360, (255, 0, 0), -1)
        else:
            cv2.circle(image, (x, y), radius, (255, 0, 0), -1)
        
        # Store the speckle's ground truth
        speckles.append({'x': x, 'y': y, 'radius': radius})
    
    # Add some blur and noise to the image
    image = cv2.GaussianBlur(image, (3, 3), 0)
    noisy('speckle', image)
    
    return image, speckles

def save_data(image, speckles, image_path, gt_path):
    cv2.imwrite(image_path, image)
    with open(gt_path, 'w') as f:
        json.dump(speckles, f)

# Generate a synthetic image
width, height, num_speckles = 500, 300, 200
image, speckles = generate_synthetic_image(width, height, num_speckles)

# Save the image and ground truth data
#save_data(image, speckles, '/mnt/data/synthetic_image.jpg', '/mnt/data/ground_truth.json')

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
