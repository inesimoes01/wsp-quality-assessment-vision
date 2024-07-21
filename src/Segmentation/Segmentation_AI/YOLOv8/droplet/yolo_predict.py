from ultralytics import YOLO
#from matplotlib import pyplot as plt
#import torch
import glob
import os

yaml_file = "data\\artificial_dataset\\yolo_data\\configuration.yaml"
results_folder = "results\\yolo\\validation"
train_model_name = "300epc_5iou_0001lr_0005wd_2drp6"
test_model_name = "300epc_5iou_0001lr_0005wd_2drp6"
classes = [0]

if __name__ == '__main__': 
    
    model = YOLO(os.path.join("results", "yolo", train_model_name, "weights", "best.pt"))

    test_metrics = model.val(data = yaml_file, 
                        project = results_folder, 
                        name = test_model_name,
                        classes = classes,
                        conf = 0.1,
                        plots = True)


    # Get most recent test folder
    print(test_metrics)
    test_folderpaths = os.path.join(results_folder, f'*{test_model_name}*')
    test_folders = glob.glob(test_folderpaths)
    most_recent_sorted_test_folder = sorted(test_folders, key=os.path.getctime, reverse=True)[0]
    # Write validation results
    with open(os.path.join(most_recent_sorted_test_folder, 'test_results.txt'), 'w') as file:
        file.write(str(test_metrics))
    
    success = model.export(format='onnx')
