from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import base64
import time
import psutil
import json

app = Flask(__name__)
CORS(app)

# Load model
model_path = 'ModelActivationElu.h5'
model = load_model(model_path)
num_classes = model.output_shape[-1]

# Load class names
def load_class_names(json_file, expected_num_classes=None):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        if isinstance(label_data, list):
            class_names = label_data
        elif isinstance(label_data, dict):
            if all(k.isdigit() for k in label_data.keys()):
                class_names = [label_data[str(i)] for i in range(len(label_data))]
            else:
                class_names = [k for k, v in sorted(label_data.items(), key=lambda item: item[1])]
        else:
            raise ValueError("Unsupported JSON format for label map")

        if expected_num_classes is not None and len(class_names) != expected_num_classes:
            raise ValueError(f"Label count mismatch: {len(class_names)} vs model's {expected_num_classes}")

        return class_names
    except Exception as e:
        print(f"Error loading class names: {str(e)}")
        return []

class_names = load_class_names('label_mapSequenceElu.json', num_classes)
if not class_names:
    raise RuntimeError("Failed to load valid class names")
print(f"✅ Loaded {len(class_names)} class names")

# Image processing
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def process_image(image):
    try:
        processed_image = preprocess_image(image)
        start_time = time.time()
        predictions = model.predict(processed_image)
        probs = tf.nn.softmax(predictions[0]).numpy()
        
        top_k = 3  # Return top 3 predictions
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        results = [{
            'class_name': class_names[i],  # Changed from 'class' to 'class_name'
            'confidence_percentage': f"{probs[i] * 100:.2f}%",  # More descriptive
            'confidence_score': float(probs[i]),  # Raw score
            'processing_time_ms': round((time.time() - start_time) * 1000, 2),
            'memory_usage_mb': round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
        } for i in top_indices]

        return {
            'success': True,
            'predictions': results,
            'system_info': {
                'processing_time_ms': round((time.time() - start_time) * 1000, 2),
                'memory_usage_mb': round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# API Endpoints
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        result = process_image(image)
        
        # Format the response for better readability
        formatted_response = {
            'status': 'success',
            'top_predictions': result['predictions'],
            'performance_metrics': result['system_info']
        }
        
        return jsonify(formatted_response)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/scan', methods=['POST'])
def scan_image():
    if not request.is_json:
        return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image data'}), 400
    
    try:
        if 'data:image' in data['image']:
            image_data = data['image'].split(',')[1]
        else:
            image_data = data['image']
            
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        result = process_image(image)
        
        # Format the response similarly to upload endpoint
        formatted_response = {
            'status': 'success',
            'top_predictions': result['predictions'],
            'performance_metrics': result['system_info']
        }
        
        return jsonify(formatted_response)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'classes_loaded': len(class_names),
        'system_info': {
            'timestamp': time.time(),
            'memory_usage_mb': round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)