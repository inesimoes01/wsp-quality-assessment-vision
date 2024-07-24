import os
import cv2
import numpy as np

def read_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    return lines

def process_dataset(images_folder, labels_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(os.path.join(output_folder, "image")):
        os.makedirs(os.path.join(output_folder, "image"))
    if not os.path.exists(os.path.join(output_folder, "label")):
        os.makedirs(os.path.join(output_folder, "label"))
    
    # List all images in images_folder
    image_files = os.listdir(images_folder)
    
    for image_file in image_files:
        parts = image_file.split(".")
        filename = parts[0]
        
        # Construct paths
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, image_file.replace(".png", ".txt"))  # Assuming labels are .txt files
        
        # Check if label file exists
        if not os.path.exists(label_path):
            continue
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Read labels
        labels = read_labels(label_path)
        square_count = 0
        square_size = 512
        height, width, _ = image.shape

        # Resize image if it is smaller than 512x512
        if height < square_size or width < square_size:
            image = cv2.resize(image, (square_size, square_size))
            height, width = square_size, square_size

            # Adjust labels
            new_labels = []
            for line in labels:
                data = line.strip().split()
                if len(data) < 2:
                    continue
                new_data = [data[0]]
                for i in range(1, len(data), 2):
                    new_data.append(float(data[i]) * width / square_size)
                    new_data.append(float(data[i+1]) * height / square_size)
                new_labels.append(new_data)
            labels = [" ".join(map(str, l)) for l in new_labels]

        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                # Extract the square
                square = image[y:y+square_size, x:x+square_size]

                original_square_shape = square.shape

                # Ensure the square is 512x512 (handle border cases)
                if square.shape[0] != square_size or square.shape[1] != square_size:
                    original_square_shape = square.shape
                    square = cv2.resize(square, (square_size, square_size))

                # Adjust labels
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
                    square_path = os.path.join(output_folder, "image", square_filename)
                    cv2.imwrite(square_path, square)

                    label_filename = f"{filename}_{square_count}.txt"
                    cropped_label_path = os.path.join(output_folder, "label", label_filename)    
                    with open(cropped_label_path, 'w') as f:
                        for coords in adjusted_labels:
                            coord_str = " ".join([f"{p}" for p in coords])
                            f.write(f"{0} {coord_str}\n")
                    
                    square_count += 1

images_folder = "data\\artificial_dataset_3\\wsp\\image"
labels_folder = "data\\artificial_dataset_3\\wsp\\label"
output_folder = "data\\artificial_dataset_3\\processed\\all"

process_dataset(images_folder, labels_folder, output_folder)
