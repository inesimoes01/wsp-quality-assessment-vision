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
        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                # Extract the square
                square = image[y:y+square_size, x:x+square_size]

                # Ensure the square is 512x512 (handle border cases)
                if square.shape[0] != square_size or square.shape[1] != square_size:
                    # Skip squares that don't meet the size requirements
                    continue



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
                            coordinates.append(float((x_label - x) / 512))
                            coordinates.append(float((y_label - y) / 512))
                    
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

        
        # # height, width, _ = image.shape
        # # num_cols = (width // 512) + 1
        # # num_rows = (height // 512) + 1
        
        # # for r in range(num_rows):
        # #     for c in range(num_cols):
        # #         start_x = c * 512
        # #         start_y = r * 512
        # #         end_x = min(start_x + 512, width)
        # #         end_y = min(start_y + 512, height)
                
        # #         cropped_image = image[start_y:end_y, start_x:end_x]
                
        # #         # Adjust labels
        # #         adjusted_labels = []

        # #         for line in labels:
        # #             data = line.strip().split()
        # #             if len(data) < 2:
        # #                 continue
                    
        # #             coordinates = []
        # #             for i in range(1, len(data), 2):
        # #                 # switched because the labels are like the original image
        # #                 x = (float(data[i])) * width
        # #                 y = (float(data[i + 1])) * height

        # #                 # coordinates.append((trans_x, trans_y))
        # #                 if start_x <= x <= end_x and start_y <= y <= end_y:
        # #                     coordinates.append(float(x / 512))
        # #                     coordinates.append(float(y / 512))
                    
        # #             adjusted_labels.append(np.array(coordinates, dtype=np.int32))
                
        #         # Save cropped image
        #         cropped_image_name = f"filename_{r}_{c}.png"
        #         cropped_label_name = f"filename_{r}_{c}.txt"
                
        #         cropped_image_path = os.path.join(output_folder, cropped_image_name)
        #         cropped_label_path = os.path.join(output_folder, cropped_label_name)
                
        #         cv2.imwrite(cropped_image_path, cropped_image)

        #         with open(cropped_label_path, 'w') as f:
        #              for coords in adjusted_labels:
        #                 coord_str = " ".join([f"{p}" for p in coords])
        #                 f.write(f"{0} {coord_str}\n")
                

# Example usage:
images_folder = "data\\artificial_dataset\\raw\\image"
labels_folder = "data\\artificial_dataset\\raw\\label"
output_folder = "data\\artificial_dataset\\processed"

process_dataset(images_folder, labels_folder, output_folder)

images_folder = "data\\artificial_dataset\\augmented\\image"
labels_folder = "data\\artificial_dataset\\augmented\\label"
output_folder = "data\\artificial_dataset\\processed"

process_dataset(images_folder, labels_folder, output_folder)
