import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==== Paths ====
train_dir = r"{Train set path}"
test_dir = r"{Test set path}"
save_dir = r"{MOdel access path}"
os.makedirs(save_dir, exist_ok=True)

# ==== Image Loader ====
def load_images(folder_path, img_size=(128, 128)):
    images, labels = [], []
    class_names = sorted(os.listdir(folder_path))
    for label, class_folder in enumerate(class_names):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
    images = np.array(images, dtype="float32") / 255.0
    images = np.expand_dims(images, -1)
    labels = np.array(labels)
    return images, labels, class_names

# ==== Load train and test sets ====
X_train_full, y_train_full, class_names = load_images(train_dir)
X_test, y_test, _ = load_images(test_dir)

# Save class names for later classification
with open(os.path.join(save_dir, "class_labels.json"), "w") as f:
    json.dump(class_names, f)
print("✅ Class labels saved:", class_names)

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full
)

# ==== CNN Model ====
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

input_shape = (X_train.shape[1], X_train.shape[2], 1)
num_classes = len(class_names)

model = build_cnn(input_shape, num_classes)

# ==== Train ====
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# ==== Evaluate ====
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==== Save Model ====
model_path = os.path.join(save_dir, "fingerprint_cnn_model.h5")
model.save(model_path)
print(f"\n✅ Model saved at: {model_path}")

