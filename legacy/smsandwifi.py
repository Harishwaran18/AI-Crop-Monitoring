import os

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
from twilio.rest import Client
from PIL import Image

# -------------------------
# Twilio SMS Setup
# -------------------------
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_NUMBER = os.environ.get("TWILIO_FROM_NUMBER", "")
USER_NUMBER = os.environ.get("ALERT_RECIPIENT", "")

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN) if TWILIO_SID and TWILIO_AUTH_TOKEN else None

# -------------------------
# Disease Labels
# -------------------------
disease_names = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold"]

# -------------------------
# Build Model (MobileNetV2 backbone)
# -------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

inputs = layers.Input(shape=(224,224,3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)

class_output = layers.Dense(len(disease_names), activation='softmax', name='disease')(x)
metrics_output = layers.Dense(4, activation='linear', name='metrics')(x)

model = models.Model(inputs=inputs, outputs=[class_output, metrics_output])
model.compile(optimizer='adam',
              loss={'disease':'sparse_categorical_crossentropy', 'metrics':'mse'},
              metrics={'disease':'accuracy', 'metrics':'mae'})

# -------------------------
# Streamlit GUI
# -------------------------
st.set_page_config(page_title="AI Crop Monitor", layout="wide")
st.title("🌱 AI Crop Monitoring: RGB + Multispectral + Hyperspectral")

# Sidebar Options
st.sidebar.header("Settings")
channel_type = st.sidebar.selectbox("Select Image Type", ["RGB", "Multispectral (Simulated)", "Hyperspectral (Simulated)"])
send_sms = st.sidebar.checkbox("Send SMS if disease detected", True)

# Image Upload
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg","tiff"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)

    # Preprocess Image
    if channel_type == "RGB":
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((224,224))
        img_array = np.array(img)
    else:
        # Simulated multispectral/hyperspectral: add extra channels
        img = Image.open(uploaded_file).convert("RGB").resize((224,224))
        img_array = np.array(img)
        extra_channels = np.random.rand(224,224,1)  # simulate NIR or other channels
        img_array = np.concatenate([img_array, extra_channels*255], axis=2)
        # Use only first 3 channels for model, extra channels could be fed to another model if available
        img_array = img_array[:,:,:3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    st.write("Generating predictions...")
    pred_class, pred_metrics = model.predict(img_array)
    disease_idx = np.argmax(pred_class[0])

    st.success(f"Predicted Disease: **{disease_names[disease_idx]}**")

    # Simulated Plant Metrics
    metrics = {
        "Temperature (°C)": round(pred_metrics[0][0]*10 + 25, 2),
        "Humidity (%)": round((pred_metrics[0][1]+0.5)*100, 2),
        "Leaf Wetness (%)": round((pred_metrics[0][2]+0.5)*100, 2),
        "Soil Moisture (%)": round((pred_metrics[0][3]+0.5)*100, 2)
    }
    st.json(metrics)

    # Send Mobile Alert
    if disease_names[disease_idx] != "Healthy" and send_sms and client is not None and USER_NUMBER:
        st.warning("🚨 Disease detected! Sending mobile alert...")
        message = client.messages.create(
            body=f"Alert! Tomato leaf disease detected: {disease_names[disease_idx]}\nMetrics: {metrics}",
            from_=TWILIO_NUMBER,
            to=USER_NUMBER
        )
        st.info(f"SMS sent! SID: {message.sid}")

st.sidebar.markdown("---")
st.sidebar.write("This app supports simulated multispectral & hyperspectral channels for research purposes.")
