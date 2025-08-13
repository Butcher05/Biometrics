import os
import cv2
import json
import numpy as np
import tensorflow as tf

# Paths
save_dir = r"D:/Coding/Python/Code/models"
model_path = os.path.join(save_dir, "fingerprint_cnn_model.h5")
labels_path = os.path.join(save_dir, "class_labels.json")

# Load model & labels
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Labels file not found at {labels_path}")

model = tf.keras.models.load_model(model_path)
with open(labels_path, "r") as f:
    class_names = json.load(f)

# Function to preprocess image
def preprocess_image(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, img_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # channel dimension
    img = np.expand_dims(img, axis=0)   # batch dimension
    return img

# Test with a new image
test_image_path = r"D:\College\YEAR 3\Main Subjects\Biometrics\fingerprint.jpg"
image = preprocess_image(test_image_path)
predictions = model.predict(image)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print(f"🖼 Predicted Class: {predicted_class} ({confidence:.2f}% confidence)")
