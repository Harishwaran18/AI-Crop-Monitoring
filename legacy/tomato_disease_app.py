import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

# --- Step 1: Generate synthetic dataset ---
def generate_synthetic_data(num_samples=100):
    img_size = 64
    num_classes = 4
    diseases = ['Healthy', 'Bacterial Spot', 'Late Blight', 'Yellow Leaf Curl']
    
    X = np.random.rand(num_samples, img_size, img_size, 3)
    y_class = np.random.randint(0, num_classes, size=(num_samples,))
    y_metrics = np.random.rand(num_samples, 4) * np.array([40, 1, 100, 100])  # temp, leafwet, humidity, soil moisture
    
    return X, y_class, y_metrics, diseases

X, y_class, y_metrics, disease_names = generate_synthetic_data(200)

# --- Step 2: Build model ---
def create_model(input_shape=(64,64,3), num_classes=4):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    
    class_output = layers.Dense(num_classes, activation='softmax', name='disease')(x)
    metrics_output = layers.Dense(4, activation='linear', name='metrics')(x)
    
    model = models.Model(inputs=inputs, outputs=[class_output, metrics_output])
    model.compile(optimizer='adam',
                  loss={'disease':'sparse_categorical_crossentropy', 'metrics':'mse'},
                  metrics={'disease':'accuracy', 'metrics':'mae'})
    return model

model = create_model()
model.fit(X, {'disease': y_class, 'metrics': y_metrics}, epochs=5, batch_size=16)

# --- Step 3: Streamlit GUI ---
st.title("Tomato Plant Disease & Health Predictor")

uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((64,64))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred_class, pred_metrics = model.predict(img_array)
    
    disease_idx = np.argmax(pred_class)
    st.subheader(f"Disease Prediction: {disease_names[disease_idx]}")
    
    temp, leafwet, humidity, soil_moisture = pred_metrics[0]
    st.subheader(f"Temperature: {temp:.1f} °C")
    st.subheader(f"Leaf Wetness: {leafwet:.2f}")
    st.subheader(f"Humidity: {humidity:.1f} %")
    st.subheader(f"Soil Moisture: {soil_moisture:.1f} %")
