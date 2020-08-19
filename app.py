import os
from flask import Flask
from flask import render_template
from flask import request, redirect, jsonify


from config import Configuration

import tensorflow as tf
import segmentation_models as sm

app = Flask(__name__)
app.config.from_object(Configuration)

os.environ['TF_KERAS'] = '1'

sm.set_framework('tf.keras')
BACKBONE = 'mobilenet'
 
session = tf.Session()
tf.keras.backend.set_session(session)

model = tf.keras.models.load_model('data/fifa_scores.h5')

