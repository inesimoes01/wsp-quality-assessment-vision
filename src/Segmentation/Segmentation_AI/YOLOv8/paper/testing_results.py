from ultralytics import YOLO
import os 


train_model_path = "results\\yolo_rectangle\\200epc_rectangle2"

model = YOLO(os.path.join(train_model_path, "weights", "best.pt"))

model.export(format='tflite')
