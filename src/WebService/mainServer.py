from flask import Flask, jsonify, request
import cv2
import base64
import sys
from matplotlib import pyplot as plt 
from PIL import Image
import io
import numpy as np
from datetime import date
import os
from ultralytics import YOLO



sys.path.insert(0, 'src/')
from Segmentation.droplet.ccv.Segmentation_CCV import Segmentation_CCV
import WebService.paper_segmentation as paper_segmentation
import WebService.droplet_segmentation as droplet_segmentation
import Common.config as config
#from Distortion import Distortion
#from Segmentation_CV import Segmentation
# from Distortion import Distortion
# from Segmentation import Segmentation
import CreateGraph

AUX_FOLDER = "src\\WebService\\aux_files"

app = Flask(__name__)

def create_json_answer(imgdata, vmd_value, rsf_value, coverage_percentage, no_droplets, droplet_sizes_list=None):    
    
    data_graph = CreateGraph.create_graph_values(droplet_sizes_list)
    data = {
        "image_bitmap": imgdata,
        "vmd": vmd_value,
        "rsf": rsf_value,
        "coverage_percentage": int(coverage_percentage),
        "number_droplets": int(no_droplets),
        "values_of_diameter": data_graph  
    }

    return data
        


def compute_function(image_uri, paper_width, paper_height, model):
    for filename in os.listdir(AUX_FOLDER):
        path = os.path.join(AUX_FOLDER, filename)
        if os.path.isfile(path):
            os.remove(path)

    imgdata = base64.b64decode(image_uri)
    im = Image.open(io.BytesIO(imgdata))
    
    image_color = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(np.array(im), cv2.IMREAD_GRAYSCALE)

    filename = str(date.today())

    image_file = os.path.join(AUX_FOLDER, "original_" + filename + ".png")
    undistorted_image_file = os.path.join(AUX_FOLDER, "undistorted_" + filename + ".png")
    droplet_segmentation_file = os.path.join(AUX_FOLDER, "segmentation_" + filename + ".png")
    cv2.imwrite(image_file, image_color)

    model_yolo_paper = YOLO(config.PAPER_YOLO_MODEL)

    image_to_analyze = cv2.imread(image_file)
    width, height = image_to_analyze.shape[:2]

    match model:
        case 0: # Cellpose model
            undistorted_image = paper_segmentation.find_paper_yolo(image_to_analyze, filename, model_yolo_paper)
            cv2.imwrite(undistorted_image_file, undistorted_image)
            
            vmd_value, rsf_value, coverage_percentage, total_no_droplets, diameter_list = droplet_segmentation.droplet_segmentation_cellpose(undistorted_image, undistorted_image_file, AUX_FOLDER, filename, paper_width, 10, width, height)
            return

        case 1: # CV original segmentation
            undistorted_image = paper_segmentation.find_paper_ccv(image_to_analyze, filename)
            cv2.imwrite(undistorted_image_file, undistorted_image)

            image_gray = cv2.imread(undistorted_image_file, cv2.IMREAD_GRAYSCALE)

            vmd_value, rsf_value, coverage_percentage, total_no_droplets, diameter_list = droplet_segmentation.droplet_segmentation_ccv(undistorted_image, image_gray, filename, paper_width, width, height, AUX_FOLDER)
            
    im = cv2.imread(droplet_segmentation_file)
    _, buffer = cv2.imencode('.png', im)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return create_json_answer(img_str, vmd_value, rsf_value, coverage_percentage, total_no_droplets, diameter_list)
        
    

@app.route('/perform_segmentation', methods=['POST'])
def compute():
    try:
        print("received request")
        # parse the JSON request payload
        settings = request.get_json()
        #print(settings)
        if not settings:
            return jsonify({"error": "Invalid input"}), 400
        
        image_uri = settings.get('image_uri')        
        paper_width = settings.get('paper_width')
        paper_height = settings.get('paper_height')
        model = settings.get('model')

    
        # perform segmentation
        result = compute_function(image_uri, paper_width, paper_height, model)
      
        return jsonify(result)
    
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)