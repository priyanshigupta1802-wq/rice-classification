"""
Rice Classification - Final Prediction Script
=============================================
• Uses trained MobileNetV2 model
• Takes image path as input
• Outputs class + confidence
"""

import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ── CONFIG ─────────────────────────────────────────
MODEL_PATH = r"D:\grain analysis\rice_mobilenet_model.h5"
IMAGE_SIZE = (128, 128)

# MUST match training order
CLASS_NAMES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']


# ── LOAD MODEL ─────────────────────────────────────
print("🔄 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!\n")


# ── IMAGE PREPROCESSING ────────────────────────────
"""def preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img, channels=3)
    img.set_shape([None, None, 3])

    img = tf.image.resize(img, IMAGE_SIZE)

    # ✅ IMPORTANT: same as training
    img = tf.cast(img, tf.float32) / 255.0

    img = tf.expand_dims(img, axis=0)
    return img"""
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # ✅ correct size
    img = image.img_to_array(img)
    img = img / 255.0  # ✅ normalization
    img = np.expand_dims(img, axis=0)
    return img

# ── PREDICTION FUNCTION ───────────────────────────
def predict(img_path):
    if not os.path.exists(img_path):
        print("❌ Error: Image not found!")
        return

    img = preprocess_image(img_path)

    pred = model.predict(img)[0]

    class_idx = np.argmax(pred)
    confidence = pred[class_idx]

    print("\n" + "="*50)
    print(f"📷 Image: {img_path}")
    print(f"🔍 Prediction: {CLASS_NAMES[class_idx]}")
    print(f"📊 Confidence: {confidence*100:.2f}%")
    print("="*50)

    # Show all class probabilities
    print("\nDetailed Probabilities:")
    for i, prob in enumerate(pred):
        print(f"{CLASS_NAMES[i]}: {prob*100:.2f}%")


# ── MAIN ──────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py predict.py <image_path>")
    else:
        image_path = sys.argv[1]
        predict(image_path)