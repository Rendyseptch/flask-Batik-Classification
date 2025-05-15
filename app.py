from flask import Flask, request, jsonify
from flask_cors import CORS  
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import time
import json
import io
import base64
import psutil
import os

app = Flask(__name__)
CORS(app) 

# Global variables for model and label map
model = None
label_map = {}

# Initialize model and label map
def initialize_model():
    global model, label_map
    try:
        model = load_model('ModelActivationElu.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None

    try:
        with open("label_mapSequenceElu.json", "r") as f:
            label_data = json.load(f)
        label_map = {int(v): k for k, v in label_data.items()}
        print("Label map loaded successfully")
    except Exception as e:
        print(f"Failed to load label map: {e}")
        label_map = {}

# Call the initialization function
initialize_model()

def preprocess_image(image, target_size=(150, 150)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def process_image(image):
    if model is None:
        return {"success": False, "error": "Model not loaded"}
    if not label_map:
        return {"success": False, "error": "Label map not loaded"}

    start_time = time.time()
    try:
        processed = preprocess_image(image)
        preds = model.predict(processed)[0]
        top_indices = preds.argsort()[-3:][::-1]
        results = []
        for i in top_indices:
            results.append({
                "class_name": label_map.get(i, "Unknown"),
                "confidence_score": float(preds[i]),
                "confidence_percentage": f"{preds[i]*100:.2f}%"
            })
        inference_time_ms = round((time.time() - start_time) * 1000, 2)
        memory_usage_mb = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
        return {
            "success": True,
            "predictions": results,
            "system_info": {
                "inference_time_ms": inference_time_ms,
                "memory_usage_mb": memory_usage_mb
            }
        }
    except Exception as e:
        return {"success": False, "error": f"Image processing failed: {str(e)}"}

@app.route('/api/upload', methods=['POST'])
def upload_file():
    file = None
    if 'file' in request.files:
        file = request.files['file']
    elif 'image' in request.files:
        file = request.files['image']

    if file is None:
        return jsonify({
            'success': False,
            'error': 'No file uploaded',
            'message': 'Please provide a file with the "file" or "image" key'
        }), 400

    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename',
            'message': 'The uploaded file has no filename'
        }), 400

    try:
        image = Image.open(file.stream)
        result = process_image(image)
        if not result['success']:
            return jsonify({
                'success': False,
                'error': result['error'],
                'message': 'Failed to process image'
            }), 400
        return jsonify({
            'success': True,
            'message': 'Image processed successfully',
            'data': {
                'top_predictions': result['predictions'],
                'performance_metrics': result['system_info']
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred while processing the image'
        }), 500

@app.route('/api/scan', methods=['POST'])
def scan_image():
    if not request.is_json:
        return jsonify({
            'success': False,
            'error': 'Request must be JSON',
            'message': 'Please send a JSON request with Content-Type: application/json'
        }), 400

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({
            'success': False,
            'error': 'No image data',
            'message': 'Please provide an image in the "image" field'
        }), 400

    try:
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        result = process_image(image)
        if not result['success']:
            return jsonify({
                'success': False,
                'error': result['error'],
                'message': 'Failed to process image'
            }), 400
        return jsonify({
            'success': True,
            'message': 'Image processed successfully',
            'data': {
                'top_predictions': result['predictions'],
                'performance_metrics': result['system_info']
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred while processing the image'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'message': 'Service is healthy',
        'data': {
            'status': 'healthy',
            'model_loaded': model is not None,
            'classes_loaded': len(label_map),
            'system_info': {
                'timestamp': time.time(),
                'memory_usage_mb': round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
            }
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)