from flask import Flask, jsonify, request
import cv2
import base64
import sys
import os
from matplotlib import pyplot as plt 
from PIL import Image
import io
import numpy as np

# sys.path.insert(0, 'src/Segmentation_CV')
# from Distortion import Distortion
# from Segmentation import Segmentation

app = Flask(__name__)

def compute_function(image_uri, paper_width, paper_height, isAI):
    # get name of file
    # file_path_parts = image_uri.split("/")
    # parts = image_uri[len(file_path_parts)].split(".")
    # filename = parts[0]
    
    # save image
   
    
    # print("here")
    imgdata = base64.b64decode(image_uri)
    im = Image.open(io.BytesIO(imgdata))
    image_color = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(np.array(im), cv2.IMREAD_GRAYSCALE)
    print("AHHHH1")
    # image_gray = cv2.imread(image_uri, cv2.IMREAD_GRAYSCALE)
    # image_color = cv2.imread(image_uri)

    if (isAI):
        return
    else:
        # find paper in image
        # dist:Distortion = Distortion(image_gray, image_color, "", False)
        # no_paper = dist.noPaper
        # if no_paper: return
 
        # segmentation:Segmentation = Segmentation(dist.undistorted_image, "0", save_images=False, create_masks= False)
    
        _, buffer = cv2.imencode('.png', image_color)
        img_str = base64.b64encode(buffer).decode('utf-8')

        

        # data = {
        #     "image_bitmap": imgdata,
        #     "vmd": segmentation.stats.vmd_value,
        #     "rsf": segmentation.stats.rsf_value,
        #     "coverage_percentage": segmentation.stats.coverage_percentage,
        #     "number_droplets": segmentation.stats.no_droplets,
        #     "overlapped_percentage": 15.8,
        #     "values_of_radius": segmentation.droplet_diameter  
        # }
    
        data = {
            "image_bitmap": img_str,
            "vmd": 0.0,
            "rsf": 0.0,
            "coverage_percentage": 12,
            "number_droplets": 222,
            "overlapped_percentage": 15.8,
            "values_of_radius": [1, 2, 3]
        }

        print("AHHHH3")
        return data



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
        isAI = settings.get('isAI')
    
        # perform segmentation
        result = compute_function(image_uri, paper_width, paper_height, isAI)
        print("AHHHH4")
        return jsonify(result)
    
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)