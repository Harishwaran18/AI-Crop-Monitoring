import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# --------------------------
# Step 1: Load plant image
# --------------------------
def load_image(path, img_size=(128, 128)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, img_size)
    return img, img_resized / 255.0  # normalized

# --------------------------
# Step 2: Dummy CNN Model
# --------------------------
def build_dummy_model(input_shape=(128, 128, 3), num_classes=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --------------------------
# Step 3: Simulated sensor data
# --------------------------
def get_sensor_data():
    return {
        "Temperature": round(random.uniform(20, 35), 2),  # °C
        "Humidity": round(random.uniform(40, 80), 2),    # %
        "Soil Moisture": round(random.uniform(20, 60), 2), # %
        "Leaf Wetness": round(random.uniform(0, 1), 2)   # 0=dry, 1=wet
    }

# --------------------------
# Step 4: Fusion rule
# --------------------------
def fuse_predictions(ai_pred, sensor_data):
    label = "Healthy" if np.argmax(ai_pred) == 0 else "Diseased"
    
    # Sensor logic: if too humid + wet leaf, raise risk
    if sensor_data["Humidity"] > 70 and sensor_data["Leaf Wetness"] > 0.7:
        label = "High Disease Risk"
    
    return label

# --------------------------
# Step 5: Run everything
# --------------------------
if __name__ == "__main__":
    # Load image (replace with your test leaf image path)
    img_path = "test_leaf.jpg"
    orig_img, input_img = load_image(img_path)

    # Dummy CNN (untrained for now, just to simulate pipeline)
    model = build_dummy_model()
    dummy_pred = model.predict(np.expand_dims(input_img, axis=0), verbose=0)

    # Fake prediction (since model isn’t trained)
    dummy_pred = np.array([[0.3, 0.7]])  # pretend: 70% diseased

    # Sensor data
    sensor_data = get_sensor_data()

    # Fusion
    decision = fuse_predictions(dummy_pred, sensor_data)

    # --------------------------
    # Step 6: Visualization
    # --------------------------
    plt.figure(figsize=(8,6))

    plt.subplot(1,2,1)
    plt.imshow(orig_img)
    plt.title(f"AI Decision: {decision}")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.bar(sensor_data.keys(), sensor_data.values(), color='green')
    plt.title("Sensor Data")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()
