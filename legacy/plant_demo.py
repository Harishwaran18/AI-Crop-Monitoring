import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cursor-friendly
plt.style.use('default')  # Avoid style errors

# Paths
BASE_DIR = os.environ.get("CROPMON_BASE_DIR", str(Path(__file__).resolve().parents[1]))
DATA_PATH = os.path.join(BASE_DIR, "data")
CSV_FILE = os.path.join(DATA_PATH, "sensor_data.csv")

# Ensure data folder exists
os.makedirs(DATA_PATH, exist_ok=True)

print("🚀 sensor_demo.py started (Cursor mode)")

# Step 1: Load or Generate Data
if not os.path.exists(CSV_FILE):
    print("⚠️ sensor_data.csv missing. Generating sample data...")
    rng = pd.date_range("2025-09-01", periods=48, freq="h")  # Hourly data
    df = pd.DataFrame({
        "timestamp": rng,
        "sensor_id": ["SENSOR_001"] * len(rng),
        "latitude": [10.8505] * len(rng),
        "longitude": [78.7047] * len(rng),
        "soil_moisture": np.random.uniform(15, 35, len(rng)).round(2),
        "temperature": np.random.uniform(20, 40, len(rng)).round(2),
        "humidity": np.random.uniform(40, 90, len(rng)).round(1),
        "leaf_wetness": np.random.randint(0, 2, len(rng))
    })
    df.to_csv(CSV_FILE, index=False)
else:
    df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
    print(f"✅ Loaded sensor data from {CSV_FILE}")

# Step 2: Show Table in Cursor
print("\n📊 Sensor Data Preview:")
print(df.head())  # Cursor-friendly print

# Step 3: Soil Moisture Plot
plt.figure(figsize=(10,4))
plt.plot(df["timestamp"], df["soil_moisture"], color='green', marker='o')
plt.title("Soil Moisture Trend")
plt.xlabel("Timestamp")
plt.ylabel("Soil Moisture (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 4: Temperature & Humidity Plot
plt.figure(figsize=(10,4))
plt.plot(df["timestamp"], df["temperature"], label="Temperature (°C)", color='red', marker='s')
plt.plot(df["timestamp"], df["humidity"], label="Humidity (%)", color='blue', marker='x')
plt.title("Temperature & Humidity Trend")
plt.xlabel("Timestamp")
plt.ylabel("Values")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 5: Leaf Wetness Plot
plt.figure(figsize=(10,4))
plt.bar(df["timestamp"].dt.strftime("%m-%d %H:%M"), df["leaf_wetness"], color='purple')
plt.title("Leaf Wetness")
plt.xlabel("Timestamp")
plt.ylabel("0 = Dry, 1 = Wet")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 6: Generate Recommendations
def generate_recommendations(soil_moisture, temperature, humidity, disease_risk, plant_condition):
    print("\n🌱 Practical Recommendations:\n")

    # Soil moisture remedies
    if soil_moisture < 20:
        print("- Soil is too dry. Consider drip irrigation or mulching to retain water.")
    elif soil_moisture > 70:
        print("- Soil is waterlogged. Improve drainage to prevent root rot.")
    else:
        print("- Soil moisture is optimal. Keep monitoring regularly.")

    # Temperature remedies
    if temperature > 35:
        print("- High temperature detected. Provide shade nets or timely irrigation to cool plants.")
    elif temperature < 10:
        print("- Low temperature detected. Use protective covers or greenhouse techniques.")
    else:
        print("- Temperature is within the healthy range.")

    # Humidity remedies
    if humidity > 80:
        print("- High humidity may cause fungal diseases. Improve ventilation and monitor leaf wetness.")
    elif humidity < 30:
        print("- Low humidity can stress crops. Use misting or cover crops to maintain moisture.")
    else:
        print("- Humidity is suitable for plant growth.")

    # Disease risk remedies
    if disease_risk > 0.5:
        print("- High disease risk detected. Apply biological control or safe fungicides early.")
    else:
        print("- Disease risk is low. Maintain good field hygiene.")

    # Plant condition remedies
    if plant_condition == "stressed":
        print("- Plants are stressed. Apply balanced fertilizer and check for nutrient deficiencies.")
    else:
        print("- Plants appear healthy. Continue current practices.")

# Use last row of data for recommendations
last_row = df.iloc[-1]
generate_recommendations(
    soil_moisture=last_row["soil_moisture"],
    temperature=last_row["temperature"],
    humidity=last_row["humidity"],
    disease_risk=0,  # Placeholder, you can link this with your ML model
    plant_condition="healthy"  # Replace with model output later
)

print("\n🎉 Sensor demo completed — all output visible inside Cursor!")
