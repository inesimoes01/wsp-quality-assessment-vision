import os

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    filtered_lines = []
    for line in lines:
        values = line.split()
        if len(values) > 6:  # 1 for the initial 0 and at least 3 more values
            filtered_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(filtered_lines)

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            process_file(file_path)

folder_path = 'data\\artificial_dataset\\yolo_data\\labels\\test'  # Replace with the path to your folder
process_folder(folder_path)
folder_path = 'data\\artificial_dataset\\yolo_data\\labels\\train'  # Replace with the path to your folder
process_folder(folder_path)
folder_path = 'data\\artificial_dataset\\yolo_data\\labels\\val'  # Replace with the path to your folder
process_folder(folder_path)