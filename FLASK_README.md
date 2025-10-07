# Flask Web Application - Lung Cancer Prediction

## 🌐 Web Interface Features

### **What's New:**
- **Web-based Interface**: Upload images through browser
- **Real-time Predictions**: Instant results with confidence scores
- **Interactive Visualizations**: Bar charts showing all class probabilities
- **REST API**: JSON endpoints for integration with other applications
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### **How to Run:**

1. **Install Dependencies:**
   ```bash
   pip install Flask Werkzeug
   ```

2. **Start the Web Server:**
   ```bash
   python app.py
   ```

3. **Access the Application:**
   - Open your browser and go to: `http://localhost:5000`
   - Upload a lung cancer image and get instant predictions

### **API Endpoints:**

- **Web Interface**: `GET /` - Main upload page
- **Prediction API**: `POST /api/predict` - JSON API for predictions
- **Health Check**: `GET /health` - Server status

### **API Usage Example:**
```python
import requests

url = "http://localhost:5000/api/predict"
files = {"file": open("lung_image.png", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### **Features:**
- ✅ Drag & drop file upload
- ✅ Image preview before prediction
- ✅ Confidence scores and probability distribution
- ✅ Professional medical-themed UI
- ✅ Mobile-responsive design
- ✅ Error handling and validation
- ✅ Medical disclaimer for ethical use

### **Technology Stack:**
- **Backend**: Flask (Python web framework)
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **AI Model**: TensorFlow/Keras with Xception CNN
- **Visualization**: Matplotlib charts
- **Image Processing**: PIL/Pillow

This transforms your command-line tool into a professional web application suitable for deployment and real-world use!