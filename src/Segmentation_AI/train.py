from ultralytics import YOLO
from matplotlib import pyplot as plt
import torch
import glob
import os

yaml_file = "src/ai/configuration.yaml"
results_folder = "yolo_models"
train_model_name = "teste1_train"
test_model_name = "teste1_test"
classes = [1]

if __name__ == '__main__': 
    # Load a COCO-pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data=yaml_file, 
                          epochs=100, 
                          imgsz=640)
    
    test_metrics = model.train(data=yaml_file,
                        project=results_folder,
                        name=train_model_name,
                        epochs=100,
                        classes=classes,
                        patience=30, 
                        batch=16,
                        dropout=0.2,
                        imgsz=512,
                        workers=0,
                        val=True)


    # Perform validation on the test set
    test_metrics = model.val(data=yaml_file, 
                        project=results_folder,
                        name=test_model_name,
                        classes=classes,
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
