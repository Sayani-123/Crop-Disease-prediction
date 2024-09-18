from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model(r'D:\disease\Models\trained_plant_disease_model.keras')

# Define the class names based on your model's output
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
               'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
               'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
               'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Route for the home page
@app.route('/')
def home():
    return render_template('disease.html')

# Route to handle prediction after image upload
@app.route('/disease/predict', methods=['POST'])
def predict_image():
    # Get the uploaded image file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    
    # Check if the file is not empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the image temporarily
    img_path = os.path.join('static', 'uploads', file.filename)
    file.save(img_path)
    
    # Load and preprocess the image
    try:
        image = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))  # Adjust target size
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        # Predict using the model
        predictions = model.predict(input_arr)
        result_index = np.argmax(predictions)

        # Ensure the result index is within bounds
        if result_index < len(class_names):
            predicted_class_label = class_names[result_index]
            return jsonify({'result': predicted_class_label}), 200
        else:
            return jsonify({'error': 'Prediction index out of range'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure uploads directory exists
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    app.run(debug=True)
