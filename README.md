# Project Overview
SkinSense ML Deploy is a Flask API that serves a pre-trained Keras model for dermatological image classification. It accepts base64-encoded images and returns a predicted skin condition label.

# Problem Statement
Provide a lightweight API that can classify skin disease images so downstream clients can submit an image payload and receive a diagnosis label programmatically.

# Solution Summary
- Load the Keras `.h5` model at application startup.
- Expose a `/predict` endpoint that accepts JSON with a base64 image string.
- Preprocess images to 224×224 and run inference with the InceptionV3 pipeline.
- Return a JSON response containing the predicted class label.

# Technical Architecture
- **Flask app (`app.py`)** initializes a single application instance and loads the model once.
- **Model inference pipeline** decodes base64, converts to a PIL image, saves a temporary file, and runs TensorFlow/Keras inference.
- **Prediction output** maps the argmax index to one of eight hardcoded class labels.
- **Deployment** is configured for Gunicorn via `Procfile` and Python 3.12.2 via `runtime.txt`.

# Key Features
- `/predict` POST endpoint for image classification requests.
- Base64 decoding and image validation with explicit error handling.
- Eight-class dermatological classification output in JSON.
- Root `/` endpoint for quick service health check.

# Tech Stack
- Python 3.12.2
- Flask 3.0.1
- TensorFlow (CPU) 2.16.1 / Keras
- Pillow 10.2.0
- NumPy 1.26.3
- Gunicorn 21.2.0
- Requests 2.31.0

# Setup and Run Instructions
1. Create and activate a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Start the API locally.
   ```bash
   python app.py
   ```
4. Send a prediction request.
   ```bash
   curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"image":"data:image/jpeg;base64,<BASE64_DATA>"}'
   ```

# Challenges Faced
- Ensuring base64 image payloads decode reliably before inference.
- Aligning image preprocessing (224×224 and InceptionV3 normalization) with the trained model’s expectations.

# Key Learnings
- Packaging a TensorFlow/Keras model behind a Flask REST API.
- Building a deterministic preprocessing pipeline for image inference.
- Handling binary image data over JSON APIs.
