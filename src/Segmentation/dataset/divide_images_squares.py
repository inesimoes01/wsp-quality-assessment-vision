import cv2
import os
import sys
sys.path.insert(0, 'src/common')
import config as config

def divide_image(filename, end, image_path, output_dir, square_size=512):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    height, width, _ = image.shape

    # Iterate through the image and extract 512x512 squares
    square_count = 0
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            # Extract the square
            square = image[y:y+square_size, x:x+square_size]

            # Ensure the square is 512x512 (handle border cases)
            if square.shape[0] != square_size or square.shape[1] != square_size:
                # Skip squares that don't meet the size requirements
                continue

            # Save the square
            square_filename = f"{filename}_square{square_count}.{end}"
            square_path = os.path.join(output_dir, square_filename)
            cv2.imwrite(square_path, square)
            square_count += 1

    print(f"Divided image into {square_count} squares of size {square_size}x{square_size} and saved to {output_dir}")

out_path = os.path.join(config.DATA_REAL_PROC_IMAGE_DIR)
if not os.path.exists(out_path):
    os.makedirs(out_path)

for file in os.listdir(config.DATA_REAL_RAW_IMAGE_DIR):
    parts = file.split(".")
    filename = parts[0]
    end = parts[1]

    in_path = os.path.join(config.DATA_REAL_RAW_IMAGE_DIR, file)
    

    divide_image(filename, end, in_path, out_path)