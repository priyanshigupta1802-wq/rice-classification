"""
Evaluate Rice Classification Model (No Training)
==============================================
• Loads saved model
• Builds validation dataset
• Generates confusion matrix + classification report
• Saves confusion matrix image

Run: py evaluate_model.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ── Config ──────────────────────────────────────────────────────
DATASET_PATH = r"D:\grain analysis\Rice_Image_Dataset\Rice_Image_Dataset"
MODEL_PATH   = r"D:\grain analysis\rice_mobilenet_model.h5"
IMAGE_SIZE   = (128, 128)
BATCH_SIZE   = 64
AUTOTUNE     = tf.data.AUTOTUNE

# ── Load dataset paths ──────────────────────────────────────────
images, labels = [], []

for cls in sorted(os.listdir(DATASET_PATH)):
    cls_path = os.path.join(DATASET_PATH, cls)
    if not os.path.isdir(cls_path):
        continue
    for fname in os.listdir(cls_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(os.path.join(cls_path, fname))
            labels.append(cls)

df = pd.DataFrame({"image": images, "label": labels})

# ── Encode labels ───────────────────────────────────────────────
class_names = sorted(df["label"].unique())
num_classes = len(class_names)
label2idx = {c: i for i, c in enumerate(class_names)}
df["label_idx"] = df["label"].map(label2idx)

print("\nClasses:", label2idx)

# ── SAME split as training ──────────────────────────────────────
df_train, df_val = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

print(f"Validation samples: {len(df_val)}")

# ── tf.data pipeline ────────────────────────────────────────────
H, W = IMAGE_SIZE

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [H, W])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def make_dataset(df_part):
    paths  = tf.constant(df_part["image"].values)
    idxs   = tf.constant(df_part["label_idx"].values, dtype=tf.int32)
    labels = tf.one_hot(idxs, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

val_ds = make_dataset(df_val)

# ── Load trained model ──────────────────────────────────────────
print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ── Predict ─────────────────────────────────────────────────────
print("\nRunning predictions...")
y_true = df_val["label_idx"].values

y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# ── Confusion Matrix ────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# ── Plot ────────────────────────────────────────────────────────
plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Rice Classification")

plt.tight_layout()
save_path = r"D:\grain analysis\confusion_matrix.png"
plt.savefig(save_path)

print(f"\n📊 Confusion matrix saved at:\n{save_path}")

plt.show()

# ── Classification Report ───────────────────────────────────────
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))