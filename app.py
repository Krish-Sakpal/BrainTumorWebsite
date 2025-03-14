import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = joblib.load('svm_model.pkl')

def process_image(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path, 0)
    if img is None:
        return "Error: Image not found", image_path, 0
    
    img_resized = cv2.resize(img, (200, 200))
    img_flattened = img_resized.reshape(1, -1) / 255
    prediction_proba = model.decision_function(img_flattened)  # Get confidence score
    confidence = abs(prediction_proba[0]) / max(abs(prediction_proba))  # Normalize confidence score
    
    prediction = model.predict(img_flattened)
    label = "No Tumor" if prediction[0] == 0 else "Pituitary Tumor"
    
    return label, image_path, round(confidence * 100, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    result, processed_path, confidence = process_image(filepath)
    
    return render_template('result.html', prediction=result, confidence=confidence, image_path=processed_path)

if __name__ == '__main__':
    app.run(debug=True)