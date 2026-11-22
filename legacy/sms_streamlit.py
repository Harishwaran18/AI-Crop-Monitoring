import streamlit as st
from twilio.rest import Client
import os

# --- Twilio Credentials ---
# Either set as environment variables or put them here directly
TWILIO_SID = os.environ.get('TWILIO_ACCOUNT_SID', 'YOUR_TWILIO_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', 'YOUR_TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_FROM_NUMBER', '')

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# --- Streamlit App ---
st.title("AI Crop Monitoring SMS Alert System")

st.write("Upload an image of the plant leaf and get SMS alerts if a disease is detected.")

# Upload leaf image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

# Input phone number to send SMS
phone_number = st.text_input("Enter your phone number with country code (e.g., +91XXXXXXXXXX)")

if uploaded_file and phone_number:
    # Simulate AI model prediction
    st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)
    
    # Dummy prediction for demo purposes
    predicted_disease = "Leaf Mold"
    plant_metrics = {
        "Temperature (°C)": 28.5,
        "Humidity (%)": 78,
        "Leaf Wetness (%)": 65,
        "Soil Moisture (%)": 42
    }
    
    st.subheader("Predicted Disease")
    st.success(predicted_disease)
    
    st.subheader("Plant Metrics")
    st.json(plant_metrics)

    # Compose SMS message
    sms_message = f"⚠ Alert! Your plant shows signs of {predicted_disease}.\nMetrics:\n"
    for key, value in plant_metrics.items():
        sms_message += f"{key}: {value}\n"

    # Send SMS button
    if st.button("Send SMS Alert"):
        try:
            message = client.messages.create(
                body=sms_message,
                from_=TWILIO_PHONE_NUMBER,
                to=phone_number
            )
            st.success(f"SMS sent successfully to {phone_number} ✅")
        except Exception as e:
            st.error(f"Failed to send SMS: {e}")
