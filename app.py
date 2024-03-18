# Flask utils
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import re

#image convert
import base64
from PIL import Image
from io import BytesIO

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

import numpy as np

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Skin Sense ML model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/')
def home():
    return 'Skin Sense ML API is running.'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.json:
            print('No image data found in request')
            return jsonify({'error': 'No image data found'})

        # Extract base64 image data
        base64_img = request.json['image']
        base64_data = re.sub('^data:image/.+;base64,', '', base64_img)

        # Decode base64 image data
        try:
            img_data = base64.b64decode(base64_data)
        except Exception as e:
            print(f'Error decoding base64 data: {e}')
            return jsonify({'error': 'Error decoding base64 data'})
        
        # Convert binary data to image
        try:
            img = Image.open(BytesIO(img_data))
        except Exception as e:
            print(f'Error opening image: {e}')
            return jsonify({'error': 'Error opening image'})

        # Save the image temporarily
        temp_img_path = 'temp_image.jpg'
        img.save(temp_img_path)

        # Make prediction
        preds = model_predict(temp_img_path, model)

        # Process your result for human
        class_labels = ['Cellulitis', 'Impetigo', 'Athlete Foot', 'Nail Fungus', 'Ringworm', 'Cutaneous Larva Migrans', 'Chickenpox', 'Shingles']
        predicted_class_index = np.argmax(preds)
        predicted_class_label = class_labels[predicted_class_index]

        return jsonify({'predicted_class': predicted_class_label})

    return jsonify({'error': 'Unexpected error occurred.'})


if __name__ == '__main__':
    app.run(debug=True)
