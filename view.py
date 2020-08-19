from app import app, model, session
import os

from flask import render_template
from flask import request, redirect, jsonify

import segmentation_models as sm

import numpy as np
import cv2

from segmentation_utils import *
from text_utils import *

app.config["IMAGE_UPLOADS"] = os.path.join(os.path.dirname(os.path.abspath(__file__)),'static/upload')

project_path = 'data'
dataset_path = f'{project_path}/dataset'
images_path = f'{dataset_path}/images'
masks_path = f'{dataset_path}/masks'
segments_path = f'{dataset_path}/segments'
BACKBONE = 'mobilenet'
preprocess_input = sm.get_preprocessing(BACKBONE)

threshold = 0.5

@app.route('/')
def index():
    return render_template('index.html')

# define a predict function as an endpoint 
@app.route("/extract_text", methods=["GET","POST"])
def extract_text():
    data = {"success": False}

    params = request.json
    if (params == None):
        img_file = request.args.get('img_file')
        print (f'params: {img_file}')

    # if parameters are found, return a prediction
    if (img_file != None):
        
        img = cv2.cvtColor(cv2.imread(f'{images_path}/{img_file}'), cv2.COLOR_BGR2RGB)
        image_r = cv2.resize(img, (224,224))
        image_r = np.expand_dims(preprocess_image(image_r, preprocess_input), axis=0)
        print(f'Input image: {image_r.shape}')
        
        with session.as_default():
            with session.graph.as_default():
                p=model.predict(image_r)
                print(f'Predicitiong shape: {p.shape}')
        
        mask = preprocess_mask(np.squeeze(p), threshold)
        mask = cv2.resize(mask, (img.shape[1],img.shape[0]))
        mask = mask[..., 1]
        points = approx_polygon(mask)
  
        segment = {'data': [], 'points': points, 'mask': mask}
        
        if(len(points) == 4):
            solid_mask = np.zeros((img.shape[0],img.shape[1]))
            solid_mask = cv2.fillPoly(solid_mask, np.int32([points]), color=255).astype(np.uint8)
            segment['data'] = crop_segment(img, solid_mask)

        print(f"Segment shape {segment['data'].shape}")
        segment_gray_inv = 255 - cv2.cvtColor(segment['data'],cv2.COLOR_RGB2GRAY)
        text = extract_score(segment_gray_inv, lang='fifa_score')

        data["extracted_text"] = str(text)
        data["success"] = True

    # return a response in json format 
    return jsonify(data)

@app.route('/get_score', methods=["GET", "POST"])
def get_score():
    if request.method == "POST":

        if request.files:
            data = {"success": False}
            image = request.files["image"]
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

            img_link = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
            img = cv2.cvtColor(cv2.imread(f'{img_link}'), cv2.COLOR_BGR2RGB)
            resp = img.shape
            print(f'{resp}')

            if resp:

                image_r = cv2.resize(img, (224,224))
                image_r = np.expand_dims(preprocess_image(image_r, preprocess_input), axis=0)
                print(f'Input image: {image_r.shape}')
                
                with session.as_default():
                    with session.graph.as_default():
                        p=model.predict(image_r)
                        print(f'Predicitiong shape: {p.shape}')
                
                mask = preprocess_mask(np.squeeze(p), threshold)
                mask = cv2.resize(mask, (img.shape[1],img.shape[0]))
                mask = mask[..., 1]
                points = approx_polygon(mask)
        
                segment = {'data': [], 'points': points, 'mask': mask}
                
                if(len(points) == 4):
                    solid_mask = np.zeros((img.shape[0],img.shape[1]))
                    solid_mask = cv2.fillPoly(solid_mask, np.int32([points]), color=255).astype(np.uint8)
                    segment['data'] = crop_segment(img, solid_mask)

                print(f"Segment shape {segment['data'].shape}")
                segment_gray_inv = 255 - cv2.cvtColor(segment['data'],cv2.COLOR_RGB2GRAY)
                text = str(extract_score(segment_gray_inv, lang='fifa_score'))

                data["extracted_text"] = text
                data["success"] = True

                return render_template("index.html", data=text, file_name=image.filename)

    return render_template("index.html") 
