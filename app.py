#!/usr/bin/env python3
"""
Flask Web Application for Lung Cancer Prediction
Provides web interface and API for real-time predictions
"""

import os
import numpy as np
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import base64
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configuration
app = Flask(__name__)
app.secret_key = 'lung_cancer_prediction_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Model configuration
IMAGE_SIZE = (350, 350)
MODEL_PATH = 'best_model.hdf5'
CLASS_LABELS = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global model variable
model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model"""
    global model
    if model is None:
        try:
            # Create model architecture
            pretrained_model = tf.keras.applications.Xception(
                weights='imagenet', 
                include_top=False, 
                input_shape=[*IMAGE_SIZE, 3]
            )
            pretrained_model.trainable = False
            
            model = tf.keras.Sequential([
                pretrained_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(4, activation='softmax')
            ])
            
            # Load weights
            model.load_weights(MODEL_PATH)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return model

def preprocess_image(img):
    """Preprocess image for prediction"""
    try:
        # Resize image
        img = img.resize(IMAGE_SIZE)
        # Convert to array
        img_array = image.img_to_array(img)
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        # Normalize
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(img_array):
    """Make prediction on preprocessed image"""
    try:
        model = load_model()
        if model is None:
            return None, None
        
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        predicted_label = CLASS_LABELS[predicted_class]
        
        # Get all class probabilities
        all_predictions = {
            CLASS_LABELS[i]: float(predictions[0][i]) 
            for i in range(len(CLASS_LABELS))
        }
        
        return predicted_label, confidence, all_predictions
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None, None

def create_prediction_chart(all_predictions):
    """Create a bar chart of predictions"""
    try:
        plt.figure(figsize=(10, 6))
        classes = list(all_predictions.keys())
        probabilities = list(all_predictions.values())
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        bars = plt.bar(classes, probabilities, color=colors, alpha=0.8)
        
        plt.title('Lung Cancer Classification Probabilities', fontsize=16, fontweight='bold')
        plt.xlabel('Cancer Types', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        plot_url = base64.b64encode(plot_data).decode()
        return plot_url
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Load and preprocess image
            img = Image.open(file.stream).convert('RGB')
            img_array = preprocess_image(img)
            
            if img_array is None:
                flash('Error processing image')
                return redirect(url_for('index'))
            
            # Make prediction
            predicted_label, confidence, all_predictions = predict_image(img_array)
            
            if predicted_label is None:
                flash('Error making prediction')
                return redirect(url_for('index'))
            
            # Create visualization
            plot_url = create_prediction_chart(all_predictions)
            
            # Convert image to base64 for display
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_data = base64.b64encode(img_buffer.getvalue()).decode()
            
            return render_template('result.html',
                                 predicted_label=predicted_label,
                                 confidence=confidence,
                                 all_predictions=all_predictions,
                                 plot_url=plot_url,
                                 img_data=img_data)
        else:
            flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files.')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Error processing request: {str(e)}')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Process image
        img = Image.open(file.stream).convert('RGB')
        img_array = preprocess_image(img)
        
        if img_array is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        # Make prediction
        predicted_label, confidence, all_predictions = predict_image(img_array)
        
        if predicted_label is None:
            return jsonify({'error': 'Error making prediction'}), 500
        
        return jsonify({
            'predicted_class': predicted_label,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    # Load model on startup
    print("Loading model...")
    load_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5001)