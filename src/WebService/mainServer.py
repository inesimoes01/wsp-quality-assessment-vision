from flask import Flask, jsonify, request
import cv2
import base64
import sys
import os
from matplotlib import pyplot as plt 
from PIL import Image
import io
import numpy as np

sys.path.insert(0, 'src/Segmentation_CV')
from Distortion import Distortion
from Segmentation import Segmentation

app = Flask(__name__)

def compute_function(image_uri, paper_width, paper_height, isAI):
    # get name of file
    # file_path_parts = image_uri.split("/")
    # parts = image_uri[len(file_path_parts)].split(".")
    # filename = parts[0]
    
    # save image
    imgdata = base64.b64decode(image_uri)
    im = Image.open(io.BytesIO(imgdata))
    image_color = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(np.array(im), cv2.IMREAD_GRAYSCALE)

    # image_gray = cv2.imread(image_uri, cv2.IMREAD_GRAYSCALE)
    # image_color = cv2.imread(image_uri)

    if (isAI):
        return
    else:
        # find paper in image
        dist:Distortion = Distortion(image_gray, image_color, "", False)
        no_paper = dist.noPaper
        if no_paper: return
 
        segmentation:Segmentation = Segmentation(dist.undistorted_image, "0", save_images=False, create_masks= False)
    
        _, buffer = cv2.imencode('.png', segmentation.detected_image)
        img_str = base64.b64encode(buffer).decode('utf-8')

        data = {
            "image_bitmap": img_str,
            "vmd": segmentation.stats.vmd_value,
            "rsf": segmentation.stats.rsf_value,
            "coverage_percentage": segmentation.stats.coverage_percentage,
            "number_droplets": segmentation.stats.no_droplets,
            "overlapped_percentage": 15.8,
            "values_of_radius": segmentation.droplet_diameter  
        }

        return data



@app.route('/perform_segmentation', methods=['POST'])
def compute():
    try:
        
        # parse the JSON request payload
        settings = request.get_json()
        if not settings:
            print("no")
            return jsonify({"error": "Invalid input"}), 400
        
        image_uri = settings.get('image_uri')
        paper_width = settings.get('paper_width')
        paper_height = settings.get('paper_height')
        isAI = settings.get('isAI')
  
        # perform segmentation
        result = compute_function(image_uri, paper_width, paper_height, isAI)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/tasks', methods=['GET'])
# def get_tasks():
#     return jsonify({'tasks': tasks})

# @app.route('/tasks', methods=['POST'])
# def create_task():
#     new_task = {
#         'id': uuid.uuid4().hex,
#         'title': request.json['title'],
#         'description': request.json['description'],
#         'completed': request.json.get('completed', False)
#     }
#     tasks.append(new_task)
#     return jsonify({'task': new_task})

# @app.route('/tasks/<string:task_id>', methods=['GET'])
# def get_task(task_id):
#     task = [task for task in tasks if task['id'] == task_id]
#     if len(task) == 0:
#         return jsonify({'error': 'Task not found'})
#     return jsonify({'task': task[0]})

# @app.route('/tasks/<string:task_id>', methods=['PUT'])
# def update_task(task_id):
#     task = [task for task in tasks if task['id'] == task_id]
#     if len(task) == 0:
#         return jsonify({'error': 'Task not found'})
#     task[0]['title'] = request.json.get('title', task[0]['title'])
#     task[0]['description'] = request.json.get('description', task[0]['description'])
#     task[0]['completed'] = request.json.get('completed', task[0]['completed'])
#     return jsonify({'task': task[0]})

# @app.route('/tasks/<string:task_id>', methods=['DELETE'])
# def delete_task(task_id):
#     task = [task for task in tasks if task['id'] == task_id]
#     if len(task) == 0:
#         return jsonify({'error': 'Task not found'})
#     tasks.remove(task[0])
#     return jsonify({'result': 'Task deleted'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)