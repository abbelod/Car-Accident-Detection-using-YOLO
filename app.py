import os
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from ultralytics import YOLO
import traceback
import base64
from io import BytesIO

app = Flask(__name__)
# Increase maximum content length to 32MB
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
# Add timeout configuration
app.config['TIMEOUT'] = 300  # 5 minutes timeout
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your YOLO model here
model = YOLO('./model/my_model.pt')  # Update this path to your .pt file

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    try:
        print(f"Processing image at path: {image_path}")
        print(f"File exists: {os.path.exists(image_path)}")
        
        # Verify the image can be opened
        try:
            with Image.open(image_path) as img:
                # Verify the image is valid
                img.verify()
            print("Image verified successfully")
        except Exception as e:
            print(f"Error verifying image: {str(e)}")
            raise ValueError("Invalid image file")

        # Process image with YOLO
        results = model(image_path, verbose=False)  # Add verbose=False to reduce console output
        print(f"Model prediction completed")
        
        # Process the results
        result = results[0]  # Get the first image result
        print(f"Processing results")
        
        # Get the plotted image with annotations
        plotted_image = result.plot()  # This returns a numpy array of the annotated image
        
        # Convert numpy array to base64 string
        img = Image.fromarray(plotted_image)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Check if any accidents were detected
        detections = result.boxes.data.tolist()
        print(f"Number of detections: {len(detections)}")
        
        confidence_scores = [detection[4] for detection in detections] if detections else [0.0]
        max_confidence = max(confidence_scores) if confidence_scores else 0.0
        accident_detected = max_confidence > 0.5  # You can adjust this threshold
        
        print(f"Max confidence: {max_confidence}")
        print(f"Accident detected: {accident_detected}")
        
        return {
            "accident_detected": accident_detected,
            "confidence": float(max_confidence),
            "annotated_image": img_str
        }
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure the uploads directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            try:
                print(f"Saving file to: {filepath}")
                file.save(filepath)
                print(f"File saved successfully")
                
                # Check if file was actually saved
                if not os.path.exists(filepath):
                    raise FileNotFoundError("Failed to save uploaded file")
                
                # Check file size
                file_size = os.path.getsize(filepath)
                print(f"File size: {file_size / (1024*1024):.2f} MB")
                
                result = process_image(filepath)
                print(f"Processing completed successfully")
                
                # Clean up the uploaded file
                os.remove(filepath)
                return jsonify(result)
            except Exception as e:
                print(f"Error in predict route: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                # Clean up the uploaded file in case of error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': str(e)}), 500
        
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True) 