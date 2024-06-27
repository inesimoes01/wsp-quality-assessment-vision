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
    seg_class = 0

    for mask in masks:
        mask_original = mask
        
        mask = mask_original.data[0].numpy()
        polygon = mask_original.xy[0]
      
        draw = ImageDraw.Draw(img)
        if seg_class == 0:
            draw.polygon(polygon, outline = (0, 255, 0), width = 2)
        else:
            draw.polygon(polygon, outline = (255, 0, 0), width = 2) 


        seg_class += 1
        
    