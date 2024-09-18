from io import BytesIO
from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Load MobileNetV2 feature extractor
pretrained_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the pretrained model
pretrained_model.trainable = False

# Define custom layers
global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(9, activation='softmax')

# Build the model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = pretrained_model(inputs, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)
model = Model(inputs, outputs)

# Load your trained model
model.load_weights('my_model.weights.h5')

# Dictionary mapping disease index to disease name
disease_labels = {
    0: 'Cellulitis',
    1: 'Impetigo',
    2: 'Athlete\'s Foot',
    3: 'Nail Fungus',
    4: 'Ringworm',
    5: 'Cutaneous Larva Migrans',
    6: 'Chickenpox',
    7: 'Shingles',
    8: 'Normal'
}

# Dictionary mapping disease name to Wikipedia URL
disease_wikipedia_urls = {
    'Cellulitis': 'https://en.wikipedia.org/wiki/Cellulitis',
    'Impetigo': 'https://en.wikipedia.org/wiki/Impetigo',
    'Athlete\'s Foot': 'https://en.wikipedia.org/wiki/Athlete%27s_foot',
    'Nail Fungus': 'https://en.wikipedia.org/wiki/Nail_fungus',
    'Ringworm': 'https://en.wikipedia.org/wiki/Dermatophytosis',
    'Cutaneous Larva Migrans': 'https://en.wikipedia.org/wiki/Cutaneous_larva_migrans',
    'Chickenpox': 'https://en.wikipedia.org/wiki/Chickenpox',
    'Shingles': 'https://en.wikipedia.org/wiki/Herpes_zoster',
    'Normal': 'https://en.wikipedia.org/wiki/Skin_disease'
}

# Create Flask app
app = Flask(__name__)

def predict_disease(image):
    # Convert image to numpy array and resize
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return predicted_class

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/diseases", methods=['GET', 'POST'])
def diseases():
    return render_template('diseases.html')   
@app.route("/diagnosis", methods=['GET', 'POST'])
def diagnosis():
    if request.method == 'POST':
        # Handle file upload
        uploaded_image = request.files['image']
        
        if uploaded_image:
            # Convert FileStorage object to a file-like object (BytesIO)
            image_file = BytesIO(uploaded_image.read())
            
            # Load the image using keras_image.load_img
            image = keras_image.load_img(image_file, target_size=(224, 224))
            
            # Make prediction
            prediction = predict_disease(image)
            predicted_disease = disease_labels[prediction]
            
            # Return the predicted disease and a link to Google Maps
            google_maps_link = f"https://www.google.com/maps/search/{predicted_disease}+doctors"
            return render_template('result.html', predicted_disease=predicted_disease, google_maps_link=google_maps_link)
    
    # Render the home page template
    return render_template('diagnosis.html')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
