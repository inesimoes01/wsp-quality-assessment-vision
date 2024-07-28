from ultralytics import YOLO
from matplotlib import pyplot as plt
#import torch
import glob
import os


yaml_file = "data\\synthetic_normal_dataset\\yolo\\data.yaml"
results_folder = "results\\yolo_droplet"
train_model_name = "50epc_droplet"
test_model_name = "50epc_droplet_test"
classes = [0]

if __name__ == '__main__': 
  
    model = YOLO("models\\yolov8s-seg.pt")
    
    test_metrics = model.train( data=yaml_file,
                                project=results_folder,
                                name=train_model_name,
                                save_period = 50,
                                epochs=50,
                                batch=8,
                                imgsz=320,
                                workers=4,  
                                lr0=0.0001,
                                weight_decay = 0.1,
                                dropout = 0.2,  
                                plots=True)
   
    model = YOLO(os.path.join("yolo_models", train_model_name, "weights", "best.pt"))

    # model.predict()
    # # Perform validation on the test set
    # test_metrics = model.val(data=yaml_file, 
    #                     project=results_folder, 
    #                     name=test_model_name,
    #                     classes=classes,
    #                     conf = 0.5,
    #                     split='test',
    #                     plots=True)


    # Get most recent test folder
    print(test_metrics)
    test_folderpaths = os.path.join(results_folder, f'*{test_model_name}*')
    test_folders = glob.glob(test_folderpaths)
    most_recent_sorted_test_folder = sorted(test_folders, key=os.path.getctime, reverse=True)[0]
    # Write validation results
    with open(os.path.join(most_recent_sorted_test_folder, 'test_results.txt'), 'w') as file:
        file.write(str(test_metrics))
    
    success = model.export(format='onnx')
