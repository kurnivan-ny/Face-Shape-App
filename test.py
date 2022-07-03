from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os

model = load_model('face_shape.h5')

class_dict = {0 : 'Heart', 1 : 'Oblong', 2 : 'Oval', 3 : 'Round', 4 : 'Square'}

def predict_label(img_path):
    loaded_img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(loaded_img) / 255.0
    img_array = expand_dims(img_array, 0)
    predicted_bit = np.argmax(model.predict(img_array))
    return class_dict[predicted_bit]

img_path = './static/uploads/round face.jpg'
print(predict_label(img_path))