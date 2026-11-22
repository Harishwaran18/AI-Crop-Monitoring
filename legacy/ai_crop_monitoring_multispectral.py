# ai_crop_monitoring_multispectral.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

st.set_page_config(page_title="AI Crop Monitoring", layout="wide")
st.title("🌱 AI Crop Monitoring with Multispectral & Hyperspectral Simulation")

# -------------------------------
# Step 1: Synthetic spectral bands
# -------------------------------
def generate_spectral_bands(img_array, num_bands=10):
    """
    Simulate multispectral/hyperspectral bands from RGB image
    """
    H, W, C = img_array.shape
    bands = []
    for i in range(num_bands):
        band = (img_array[:,:,0]*(0.3+i*0.05) +
                img_array[:,:,1]*(0.4+i*0.03) +
                img_array[:,:,2]*(0.3-i*0.02) +
                np.random.rand(H,W)*0.05)
        band = np.clip(band, 0, 1)
        bands.append(band)
    return np.stack(bands, axis=-1)

# -------------------------------
# Step 2: Create spectral model
# -------------------------------
def create_spectral_model(input_shape=(64,64,10), num_classes=4):
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

# -------------------------------
# Step 3: Load or initialize model
# -------------------------------
# For now, we initialize a new model
spectral_model = create_spectral_model(input_shape=(64,64,10), num_classes=4)
st.info("Model initialized. You can train it with your dataset or load pretrained weights.")

# -------------------------------
# Step 4: Image upload
# -------------------------------
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg","png","jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Resize and normalize
    img_array = np.array(image.resize((64,64)))/255.0
    spectral_img = generate_spectral_bands(img_array, num_bands=10)
    spectral_img = np.expand_dims(spectral_img, axis=0)  # add batch dimension

    st.info("Generating predictions...")
    # Make prediction
    pred_class, pred_metrics = spectral_model.predict(spectral_img)
    disease_idx = np.argmax(pred_class)
    
    # Dummy disease names (replace with real labels)
    disease_names = ["Healthy", "Bacterial Spot", "Late Blight", "Leaf Mold"]
    
    st.success(f"Predicted Disease: {disease_names[disease_idx]}")
    st.write("Plant Metrics (simulated):")
    st.write({
        "Temperature (°C)": round(float(pred_metrics[0][0]),2),
        "Humidity (%)": round(float(pred_metrics[0][1]),2),
        "Leaf Wetness (%)": round(float(pred_metrics[0][2]),2),
        "Soil Moisture (%)": round(float(pred_metrics[0][3]),2)
    })

st.caption("⚡ Note: Multispectral & hyperspectral channels are simulated. For real data, integrate with NIR/multispectral camera.")
