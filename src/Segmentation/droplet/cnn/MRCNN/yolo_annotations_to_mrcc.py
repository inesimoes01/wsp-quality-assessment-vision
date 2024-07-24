import json
import cv2

def yolo_to_vgg(yolo_txt_path, output_json_path, class_mapping, image_filename, image_size, width, height):
    """
    Convert YOLOv8 annotations to VGG JSON format.
    
    :param yolo_txt_path: Path to the YOLOv8 annotation txt file.
    :param output_json_path: Path to save the output VGG JSON file.
    :param class_mapping: Dictionary mapping YOLOv8 class IDs to class names.
    :param image_filename: Filename of the annotated image.
    :param image_size: Size of the annotated image.
    """
    vgg_annotations = {}
    
    # Initialize the VGG annotation for the given image
    vgg_annotations[image_filename + str(image_size)] = {
        "filename": image_filename,
        "size": image_size,
        "regions": [],
        "file_attributes": {}
    }
    
    with open(yolo_txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            coordinates = [float(x) for x in parts[1:]]

            # Ensure coordinates are in pairs
            if len(coordinates) % 2 != 0:
                raise ValueError(f"Invalid coordinates in line: {line}")
                
            # Extract x and y coordinates
            all_points_x = [int(x * width) for x in coordinates[0::2]]
            all_points_y = [int(y * height) for y in coordinates[1::2]]


            # Create a region for this object
            region = {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": all_points_x,
                    "all_points_y": all_points_y
                },
                "region_attributes": {
                    "class": class_mapping.get(class_id, "unknown")
                }
            }
            
            vgg_annotations[image_filename + str(image_size)]["regions"].append(region)

    # Save the VGG annotations to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(vgg_annotations, json_file, indent=2)

# Example usage
yolo_txt_path = 'src\\Segmentation\\Segmentation_AI\\MRCNN\\0.txt'
output_json_path = 'src\\Segmentation\\Segmentation_AI\\MRCNN\\output_vgg.json'
class_mapping = {
    0: "droplet",
    # Add other class mappings as needed
}

image_filename = 'src\\Segmentation\\Segmentation_AI\\MRCNN\\0.png'

im = cv2.imread(image_filename)
width, height = im.shape[:2]
image_size = width * height  # Replace with the actual size of the image

yolo_to_vgg(yolo_txt_path, output_json_path, class_mapping, image_filename, image_size,width, height)
