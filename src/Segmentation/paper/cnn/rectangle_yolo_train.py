from ultralytics import YOLO
from matplotlib import pyplot as plt
#import torch
import glob
import os

def save_final_model(train_model_path):
    train_model_path = "results\\yolo_rectangle\\200epc_rectangle2"
    model = YOLO(os.path.join(train_model_path, "weights", "best.pt"))

    model.export(format='tflite')


yaml_file = "data\\real_rectangle_dataset\\data.yaml"
results_folder = "results\\yolo_rectangle"
val_folder = "results\\yolo_rectangle\\val"
train_model_name = "30epc_rectangle"
test_model_name = "30epc_rectangle_test"
classes = [0]

if __name__ == '__main__': 
    # Load a COCO-pretrained YOLOv8n model
    model = YOLO('models\\yolov8s-seg.pt')
    
    # test_metrics = model.train(data=yaml_file,
    #                             project=results_folder,
    #                             name=train_model_name,
    #                             epochs=50,
    #                             batch=8,
    #                             imgsz=640,
    #                             weight_decay = 0.1,
    #                             dropout = 0.8,
    #                             plots=True)
   
    model = YOLO(os.path.join("results\\yolo_rectangle\\30epc_rectangle7", "weights", "best.pt"))

    model.predict()

    # Perform validation on the test set
    test_metrics = model.val(data = yaml_file, 
                        project = results_folder, 
                        name = test_model_name,
                        classes=classes,
                        conf = 0.5,
                        split='test',
                        plots=True)


    # Get most recent test folder
    print(test_metrics)
    test_folderpaths = os.path.join(results_folder, f'*{test_model_name}*')
    test_folders = glob.glob(test_folderpaths)
    most_recent_sorted_test_folder = sorted(test_folders, key=os.path.getctime, reverse=True)[0]
    # Write validation results
    with open(os.path.join(most_recent_sorted_test_folder, 'test_results.txt'), 'w') as file:
        file.write(str(test_metrics))
    
    success = model.export(format='onnx')

