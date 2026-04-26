"""from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model — prefers MobileNetV2, falls back to old model
if os.path.exists("rice_mobilenet_model.h5"):
    MODEL_PATH = "rice_mobilenet_model.h5"
    MODEL_SIZE = (128, 128)   # matches full training script
elif os.path.exists("rice_classification_model.h5"):
    MODEL_PATH = "rice_classification_model.h5"
    MODEL_SIZE = (64, 64)
else:
    raise FileNotFoundError("No trained model found. Run train_mobilenet.py first.")

model = load_model(MODEL_PATH)

# Must match training class_indices (alphabetical order)
classes = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

print(f"✅ Model loaded: {MODEL_PATH}")
print(f"   Input shape : {model.input_shape}")


def extract_patches(image, grid=4):
    ###
    Split the uploaded image into grid×grid patches + a center crop.
    Each patch is resized to MODEL_SIZE and predicted independently.
    This bridges the gap between:
      - Training data  → single grain close-ups
      - User photos    → groups/bowls of rice grains
    ###
    w, h = image.size
    patch_w = w // grid
    patch_h = h // grid

    patches = []
    for row in range(grid):
        for col in range(grid):
            left  = col * patch_w
            upper = row * patch_h
            patch = image.crop((left, upper, left + patch_w, upper + patch_h))
            patch = patch.resize(MODEL_SIZE)
            patches.append(np.array(patch) / 255.0)

    # Center crop — full image scaled down
    patches.append(np.array(image.resize(MODEL_SIZE)) / 255.0)

    return np.array(patches)   # (17, 224, 224, 3)


@app.route("/predict", methods=["POST"])
def predict():
    file  = request.files["file"]
    image = Image.open(file).convert("RGB")
    w, h  = image.size

    print(f"\n--- REQUEST: image {w}x{h} px ---")

    # Patch-based inference: average predictions across all patches
    patches     = extract_patches(image, grid=4)          # (17, 224, 224, 3)
    predictions = model.predict(patches, verbose=0)        # (17, 5)
    avg_pred    = predictions.mean(axis=0)                 # (5,)

    print("Averaged class scores:")
    for i, score in enumerate(avg_pred):
        print(f"  {classes[i]}: {score * 100:.2f}%")

    class_index = int(np.argmax(avg_pred))
    confidence  = float(avg_pred[class_index])

    print(f"  => {classes[class_index]} ({confidence * 100:.1f}%)\n")

    return jsonify({
        "class":       classes[class_index],
        "confidence":  confidence,
        "all_scores":  {classes[i]: float(avg_pred[i]) for i in range(len(classes))},
        "patches_used": len(patches),
        "image_size":  f"{w}x{h}"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
"""
    
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # allow React frontend

# ==============================
# 🔹 LOAD MODEL
# ==============================
if os.path.exists("rice_mobilenet_model.h5"):
    MODEL_PATH = "rice_mobilenet_model.h5"
    MODEL_SIZE = (128, 128)
elif os.path.exists("rice_classification_model.h5"):
    MODEL_PATH = "rice_classification_model.h5"
    MODEL_SIZE = (64, 64)
else:
    raise FileNotFoundError("❌ No trained model found!")

model = load_model(MODEL_PATH)

# ⚠️ MUST match training class order
classes = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

print(f"✅ Model loaded: {MODEL_PATH}")
print(f"📏 Input shape: {model.input_shape}")


# ==============================
# 🔹 IMAGE PATCHING FUNCTION
# ==============================
def extract_patches(image, grid=4):
    """
    Split image into grid patches + center crop
    Helps model handle real-world rice images
    """
    w, h = image.size
    patch_w = w // grid
    patch_h = h // grid

    patches = []

    for row in range(grid):
        for col in range(grid):
            left = col * patch_w
            upper = row * patch_h

            patch = image.crop((left, upper, left + patch_w, upper + patch_h))
            patch = patch.resize(MODEL_SIZE)

            patches.append(np.array(patch) / 255.0)

    # center image
    patches.append(np.array(image.resize(MODEL_SIZE)) / 255.0)

    return np.array(patches)   # (17, 128, 128, 3)


# ==============================
# 🔹 PREDICTION API
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(file).convert("RGB")

        w, h = image.size
        print(f"\n📸 Image received: {w}x{h}")

        # 🔥 Patch-based prediction
        patches = extract_patches(image, grid=4)
        predictions = model.predict(patches, verbose=0)

        # ✅ Use MAX instead of mean (better accuracy)
        final_pred = np.max(predictions, axis=0)

        # DEBUG
        print("📊 Class scores:")
        for i, score in enumerate(final_pred):
            print(f"   {classes[i]}: {score*100:.2f}%")

        class_index = int(np.argmax(final_pred))
        confidence = float(final_pred[class_index])

        print(f"✅ Prediction: {classes[class_index]} ({confidence*100:.2f}%)")

        # 🔥 RESPONSE (matches React UI)
        return jsonify({
            "class": classes[class_index],
            "confidence": confidence,
            "all_scores": {
                classes[i]: float(final_pred[i])
                for i in range(len(classes))
            },
            "patches_used": len(patches),
            "image_size": f"{w}x{h}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# 🔹 RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)