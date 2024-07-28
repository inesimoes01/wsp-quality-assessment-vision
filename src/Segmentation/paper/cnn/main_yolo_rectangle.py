import os
import sys
from ultralytics import YOLO
from PIL import ImageDraw, Image
from matplotlib import pyplot as plt
sys.path.insert(0, 'src/common')
import config

#for file in os.listdir(config.DATA_ARTIFICIAL_RAW_DIR):
img = Image.open("C:\\Users\\mines\\Desktop\\artificial_dataset\\raw\\image\\0.png")
#img = Image.open(os.path.join(config.DATA_ARTIFICIAL_RAW_DIR, file))
model = YOLO("C:\\Users\\mines\\Desktop\\yolo_models\\300epc_6iou_001lr_01wd_2drp\\weights\\best.pt")

results = model.predict(source = "C:\\Users\\mines\\Desktop\\artificial_dataset\\raw\\image\\0.png", 
                        conf = 0.2,
                        save=True, save_txt=False, stream=True)

for result in results:

    masks = result.masks.data
    boxes = result.boxes.data
    people_indices = torch.where(clss == 0)

    clss = boxes[:, 5]

    # get indices of results where class is 0 (people in COCO)
    people_indices = torch.where(clss == 0)
    # use these indices to extract the relevant masks
    people_masks = masks[people_indices]
    # scale for visualizing results
    people_mask = torch.any(people_masks, dim=0).int() * 255
    # save to file
    plt.imshow(people_mask.cpu().numpy())
    plt.show()
    
    # draw = ImageDraw.Draw(img)
    # draw.polygon(polygon, outline = (0, 255, 0), width = 2)
    # polygon_list.append(polygon)

# plt.imshow(img)
# plt.show()




    
    
