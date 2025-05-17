# Traffic Accident Detection Web Application

This is a web application that allows users to upload images and detect traffic accidents using a machine learning model.

## Features

- Modern, responsive UI with drag-and-drop file upload
- Real-time image preview
- Support for PNG, JPG, and JPEG formats
- Visual confidence score display
- Clean error handling

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional)Place your trained model file in the project directory and update the model loading code in `app.py`:
```python
model = tf.keras.models.load_model('path_to_your_model')
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Open your web browser and navigate to `http://localhost:5000`
2. Click the upload zone or drag and drop an image file
3. Wait for the model to process the image
4. View the detection results and confidence score

## Technical Details

- Built with Flask
- Uses TensorFlow for model inference
- Frontend styled with Tailwind CSS
- Supports images up to 16MB
- Includes proper error handling and file cleanup

## Model Integration

To integrate your traffic accident detection model:

1. Update the `process_image` function in `app.py` with your model's preprocessing requirements
2. Ensure your model is properly loaded using the correct path
3. Adjust the image preprocessing (size, normalization, etc.) according to your model's requirements
4. Update the prediction logic to match your model's output format

## Security Considerations

- File size is limited to 16MB
- Only specific image formats are allowed
- Uploaded files are automatically cleaned up after processing
- Input validation is implemented for all file uploads 