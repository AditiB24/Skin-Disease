import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Background image
background_image = "bg.jpg"

# Display background image
st.image(background_image, use_column_width=True)

# Load MobileNetV2 feature extractor
pretrained_model = MobileNetV2(input_shape=(224, 224, 3),
                               include_top=False,
                               weights='imagenet')

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

def predict_disease(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return predicted_class

def main():
    st.sidebar.title("Diseases")
    for disease in disease_labels.values():
        if st.sidebar.button(disease, key=f"{disease}_button"):
            disease_url = disease_wikipedia_urls[disease]
            st.sidebar.markdown(f"**[Learn more about {disease}]( {disease_url} )**")
            st.sidebar.markdown("---")
            st.sidebar.markdown("[Go back to prediction](#prediction_section)")

    st.title("DermaSure")
    st.markdown("---")
    st.markdown("<div style='text-align: center;'><h2 id='prediction_section' style='color: black; font-size: 24px;'>Prediction</h2></div>", unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

    if uploaded_image is not None:
        st.markdown("---")
        st.subheader("Uploaded Image")
        image = keras_image.load_img(uploaded_image, target_size=(224, 224))
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Predict", key="predict_button"):
            prediction = predict_disease(image)
            predicted_disease = disease_labels[prediction]
            st.success(f"Predicted Disease: {predicted_disease}")

            # Show Google Maps link to search for doctors
            google_maps_link = f"https://www.google.com/maps/search/{predicted_disease}doctors"
            st.markdown(f"**[Find doctors for {predicted_disease} on Google Maps]({google_maps_link})**", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
