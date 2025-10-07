#!/usr/bin/env python3
"""
Quick test script for lung cancer prediction
"""

import os
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Configuration
IMAGE_SIZE = (350, 350)
MODEL_PATH = 'best_model.hdf5'
CLASS_LABELS = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

def load_model_and_predict():
    """Load model and run quick test"""
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found!")
        return
    
    print("Loading model...")
    
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
    print("Model loaded successfully!")
    
    # Test on a sample image from the dataset
    test_path = 'dataset/test'
    if os.path.exists(test_path):
        # Find first available image
        for class_name in CLASS_LABELS:
            class_path = os.path.join(test_path, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith('.png')]
                if images:
                    img_path = os.path.join(class_path, images[0])
                    
                    # Load and preprocess image
                    img = image.load_img(img_path, target_size=IMAGE_SIZE)
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array /= 255.0
                    
                    # Make prediction
                    predictions = model.predict(img_array, verbose=0)
                    predicted_class = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class]
                    predicted_label = CLASS_LABELS[predicted_class]
                    
                    print(f"\nTest Image: {os.path.basename(img_path)}")
                    print(f"True Class: {class_name}")
                    print(f"Predicted: {predicted_label}")
                    print(f"Confidence: {confidence:.3f}")
                    print(f"Correct: {'Yes' if predicted_label == class_name else 'No'}")
                    return
    
    print("No test images found in dataset/test/")

if __name__ == "__main__":
    load_model_and_predict()