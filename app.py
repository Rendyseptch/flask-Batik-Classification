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

# Global variables
model = None
label_map = {}

def initialize_model():
    global model, label_map
    try:
        model = load_model('augmentTest_batik_cnn_pararel_elu12.h5')
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        model = None

    try:
        with open("label_mapping_pararelEluAugment12.json", "r") as f:
            label_data = json.load(f)
        # Invert the mapping: JSON stored as {class_name: index}, we want {index: class_name}
        # Depending on your JSON, adapt if keys are strings or ints
        label_map.clear()
        for k, v in label_data.items():
            label_map[int(v)] = k
        print("[INFO] Label map loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load label map: {e}")
        label_map.clear()

initialize_model()

def preprocess_image(image, target_size=(224, 224)):
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize image
    image = image.resize(target_size)
    # Convert to numpy array
    image_array = img_to_array(image).astype("float32") / 255.0  # Normalize as done during training
    # Expand dims to add batch axis
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(image):
    if model is None:
        return {"success": False, "error": "Model not loaded"}

    if not label_map:
        return {"success": False, "error": "Label map not loaded"}

    try:
        start = time.time()
        preprocessed = preprocess_image(image)
        preds = model.predict(preprocessed)[0]
        inference_time_ms = round((time.time() - start) * 1000, 2)
        memory_usage_mb = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)

        # Get top 3 predictions
        top_indices = preds.argsort()[-3:][::-1]
        top_predictions = []
        for idx in top_indices:
            class_name = label_map.get(idx, "Unknown")
            confidence = float(preds[idx])
            confidence_pct = f"{confidence * 100:.2f}%"
            top_predictions.append({
                "class_name": class_name,
                "confidence_score": confidence,
                "confidence_percentage": confidence_pct
            })

        return {
            "success": True,
            "predictions": top_predictions,
            "system_info": {
                "inference_time_ms": inference_time_ms,
                "memory_usage_mb": memory_usage_mb
            }
        }
    except Exception as e:
        return {"success": False, "error": f"Prediction failed: {str(e)}"}

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' in request.files:
        file = request.files['file']
    elif 'image' in request.files:
        file = request.files['image']
    else:
        return jsonify({
            'success': False,
            'error': 'No file uploaded',
            'message': 'Please provide a file with key "file" or "image".'
        }), 400

    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename',
            'message': 'The uploaded file has no filename.'
        }), 400

    try:
        image = Image.open(file.stream)
        result = predict_image(image)
        if not result['success']:
            return jsonify({
                'success': False,
                'error': result['error'],
                'message': 'Failed to process image.'
            }), 400

        return jsonify({
            'success': True,
            'message': 'Image processed successfully.',
            'data': {
                'top_predictions': result['predictions'],
                'performance_metrics': result['system_info']
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error processing the uploaded image.'
        }), 500

@app.route('/api/scan', methods=['POST'])
def scan_image():
    if not request.is_json:
        return jsonify({
            'success': False,
            'error': 'Request must be JSON',
            'message': 'Send a JSON request with Content-Type: application/json'
        }), 400

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({
            'success': False,
            'error': 'No image data',
            'message': 'Provide an image in the "image" field.'
        }), 400

    image_data = data['image']
    if image_data.startswith("data:image"):
        # Remove base64 header if present
        image_data = image_data.split(",")[1]

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        result = predict_image(image)
        if not result['success']:
            return jsonify({
                'success': False,
                'error': result['error'],
                'message': 'Failed to process image.'
            }), 400

        return jsonify({
            'success': True,
            'message': 'Image processed successfully.',
            'data': {
                'top_predictions': result['predictions'],
                'performance_metrics': result['system_info']
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error processing base64 image data.'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'message': 'Service is healthy.',
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

if __name__ == "__main__":
    # Use debug=False in production
    app.run(host="0.0.0.0", port=5000, debug=True)

