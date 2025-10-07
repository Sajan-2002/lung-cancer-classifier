#!/usr/bin/env python3
"""
Lung Cancer Prediction - Local Version
Loads the pre-trained model and makes predictions on test images
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Configuration
IMAGE_SIZE = (350, 350)
MODEL_PATH = 'best_model.hdf5'
DATASET_PATH = 'dataset'

# Class labels (based on your dataset structure)
CLASS_LABELS = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

def load_trained_model():
    """Load the pre-trained model"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        print("Please ensure the trained model exists or train a new one.")
        return None
    
    try:
        # Create the model architecture (same as training)
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
        
        # Load the weights
        model.load_weights(MODEL_PATH)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_and_preprocess_image(img_path, target_size):
    """Load and preprocess an image for prediction"""
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale like training images
        return img_array, img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None, None

def predict_image(model, img_path):
    """Make prediction on a single image"""
    img_array, original_img = load_and_preprocess_image(img_path, IMAGE_SIZE)
    
    if img_array is None:
        return None, None, None
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    predicted_label = CLASS_LABELS[predicted_class]
    
    return predicted_label, confidence, original_img

def test_sample_images(model):
    """Test the model on sample images from the test dataset"""
    test_path = os.path.join(DATASET_PATH, 'test')
    
    if not os.path.exists(test_path):
        print(f"Test dataset not found at {test_path}")
        return
    
    # Get a few sample images from each class
    sample_images = []
    for class_name in CLASS_LABELS:
        class_path = os.path.join(test_path, class_name)
        if os.path.exists(class_path):
            images = os.listdir(class_path)[:2]  # Take first 2 images
            for img_name in images:
                sample_images.append((os.path.join(class_path, img_name), class_name))
    
    print(f"\nTesting on {len(sample_images)} sample images:")
    print("-" * 60)
    
    correct_predictions = 0
    total_predictions = len(sample_images)
    
    for img_path, true_label in sample_images:
        predicted_label, confidence, _ = predict_image(model, img_path)
        
        if predicted_label:
            is_correct = predicted_label == true_label
            if is_correct:
                correct_predictions += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} {os.path.basename(img_path)}")
            print(f"  True: {true_label}")
            print(f"  Predicted: {predicted_label} (confidence: {confidence:.3f})")
            print()
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Sample Test Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")

def predict_custom_image(model, img_path):
    """Predict on a custom image and display result"""
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    predicted_label, confidence, original_img = predict_image(model, img_path)
    
    if predicted_label:
        print(f"\nPrediction for {os.path.basename(img_path)}:")
        print(f"Class: {predicted_label}")
        print(f"Confidence: {confidence:.3f}")
        
        # Display the image
        plt.figure(figsize=(8, 6))
        plt.imshow(original_img)
        plt.title(f"Predicted: {predicted_label} (confidence: {confidence:.3f})")
        plt.axis('off')
        plt.show()

def main():
    """Main function to run the lung cancer prediction"""
    print("Lung Cancer Prediction System")
    print("=" * 40)
    
    # Load the trained model
    model = load_trained_model()
    if model is None:
        return
    
    # Test on sample images from the dataset
    test_sample_images(model)
    
    # Interactive prediction
    print("\n" + "=" * 40)
    print("Interactive Prediction Mode")
    print("Enter image path (or 'quit' to exit):")
    
    while True:
        img_path = input("\nImage path: ").strip()
        
        if img_path.lower() in ['quit', 'exit', 'q']:
            break
        
        if img_path:
            predict_custom_image(model, img_path)

if __name__ == "__main__":
    main()