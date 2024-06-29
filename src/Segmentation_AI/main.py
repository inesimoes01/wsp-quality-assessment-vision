import os
import sys
from ultralytics import YOLO
from PIL import ImageDraw, Image

sys.path.insert(0, 'src/common')
import config


for file in os.listdir(config.DATA_ARTIFICIAL_RAW_DIR):
    img = Image.open(os.paht.join(config.DATA_ARTIFICIAL_RAW_DIR, file))
    model = YOLO(config.YOLO_MODEL_DIR)

    result = model.predict(file)

    masks = result[0].masks
    polygon_list = []
    for mask in masks:
        mask_original = mask
        
        mask = mask_original.data[0].numpy()
        polygon = mask_original.xy[0]
      
        draw = ImageDraw.Draw(img)
        draw.polygon(polygon, outline = (0, 255, 0), width = 2)
        polygon_list.append(polygon)

    


        
        
    