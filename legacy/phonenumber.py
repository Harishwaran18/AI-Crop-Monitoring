import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "model.h5"  # Replace with your trained CNN model path
DISEASE_NAMES = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold"]  # Replace with your classes

# Secure API Key (set as environment variable)
FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")

# Your mobile number (must be verified in Fast2SMS free account)
USER_NUMBER = os.environ.get("ALERT_RECIPIENT", "")


# -------------------------
# FUNCTIONS
# -------------------------
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def predict_disease(model, img_array):
    predictions = model.predict(img_array)
    idx = np.argmax(predictions)
    return DISEASE_NAMES[idx], predictions[0][idx]


def send_sms(message_text):
    """Send SMS using Fast2SMS API"""
    if not FAST2SMS_API_KEY or not USER_NUMBER:
        return {"error": "❌ API key not set. Please set FAST2SMS_API_KEY environment variable."}

    url = "https://www.fast2sms.com/dev/bulkV2"
    payload = {
        "sender_id": "TXTIND",
        "message": message_text,
        "route": "v3",
        "numbers": USER_NUMBER
    }
    headers = {
        "authorization": FAST2SMS_API_KEY
    }

    try:
        response = requests.post(url, data=payload, headers=headers)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# -------------------------
# STREAMLIT APP
# -------------------------
st.title("🌿 AI Crop Disease Detection with SMS Alerts")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
send_alert = st.checkbox("Send SMS alert if disease detected")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Leaf", use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        model = load_model()
        img, img_array = preprocess_image(uploaded_file)
        disease, confidence = predict_disease(model, img_array)

    st.success(f"Prediction: **{disease}** (Confidence: {confidence:.2f})")

    if disease != "Healthy" and send_alert:
        st.warning("🚨 Disease detected! Sending SMS alert...")
        response = send_sms(f"🚨 Alert! Tomato leaf disease detected: {disease}")
        st.info(f"SMS Response: {response}")
