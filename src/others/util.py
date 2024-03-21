import os 
import shutil

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
