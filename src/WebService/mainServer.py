from flask import Flask, jsonify, request
import cv2
import base64
import sys
import os
from matplotlib import pyplot as plt 
from PIL import Image
import io
import numpy as np

sys.path.insert(0, 'src/')
from Segmentation.droplet.ccv.Segmentation_CV import Segmentation_CV
#from Distortion import Distortion
#from Segmentation_CV import Segmentation
# from Distortion import Distortion
# from Segmentation import Segmentation
import CreateGraph

app = Flask(__name__)

def create_json_answer(imgdata, vmd_value, rsf_value, coverage_percentage, no_droplets, overlapped_percentage, droplet_sizes_list=None):    
    
    data_graph = CreateGraph.create_graph_values(droplet_sizes_list)
    # data = {
    #     "image_bitmap": imgdata,
    #     "vmd": vmd_value,
    #     "rsf": rsf_value,
    #     "coverage_percentage": int(coverage_percentage),
    #     "number_droplets": int(no_droplets),
    #     "overlapped_percentage": 15.8,
    #     "values_of_radius": data_graph  
    # }

    data = {
        #"image_bitmap": imgdata,
        "vmd": 0.0,
        "rsf": 0.0,
        "coverage_percentage": 12,
        "number_droplets": 222,
        "overlapped_percentage": 15.8,
        "values_of_size": data_graph
    }

    return data
        


def compute_function(image_uri, paper_width, paper_height, model):

    imgdata = base64.b64decode(image_uri)
    im = Image.open(io.BytesIO(imgdata))
    image_color = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(np.array(im), cv2.IMREAD_GRAYSCALE)
    
    match model:
        case 0: # AI model
            return

        case 1: # CV original segmentation

                #find paper in image
            # dist:Distortion = Distortion(image_gray, image_color, "", False)
            # no_paper = dist.noPaper
            # if no_paper: return
            path = "data\\synthetic_normal_dataset_new\\wsp\\image\\270.png"
            # read image
            image_gray = cv2.imread(path ,cv2.IMREAD_GRAYSCALE)
            image_colors = cv2.imread(path)  

            image_colors = cv2.cvtColor(image_colors, cv2.COLOR_BGR2RGB)

            calculated = Segmentation_CV(image_colors, image_gray, "0", True, True, 0, "results\\computer_vision_algorithm")
            
            circle_areas = [droplet.area for droplet in calculated.droplets_data]
            #segmentation:Segmentation = Segmentation(dist.undistorted_image, "0", save_images=False, create_masks= False)
            im = cv2.imread("results\\latex\\pipeline\\0detected.png")
            _, buffer = cv2.imencode('.png', im)
            img_str = base64.b64encode(buffer).decode('utf-8')

            return create_json_answer(img_str, 449.9, 0.75, 8.63, 1057, 25.3, circle_areas)

            # return create_json_answer(img_str, segmentation.stats.vmd_value, segmentation.stats.rsf_value, 
            #                    segmentation.stats.coverage_percentage, segmentation.stats.final_no_droplets, 
            #                    segmentation.stats.overlaped_percentage, segmentation.droplet_area)

        
            
        case 2: # CV paper segmentation
            return
        
    

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