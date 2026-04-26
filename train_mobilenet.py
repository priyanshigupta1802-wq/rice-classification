"""
Rice Classification — MobileNetV2 Full Training (CPU Optimized)
================================================================
• All 75,000 images (no sampling)
• 128x128 input — enough detail for grain texture, much faster than 224x224
• tf.data pipeline with prefetch — 2-3x faster than ImageDataGenerator on CPU
• Two-phase training: head first → then fine-tune backbone
• Estimated time on CPU: 1.5 – 2 hours

Run: py train_mobilenet.py
"""

import os
import numpy as np
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings("ignore")

# ── enable all CPU threads ──────────────────────────────────────
num_cores = os.cpu_count() or 4
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)

# ── Config ──────────────────────────────────────────────────────
DATASET_PATH  = r"D:\grain analysis\Rice_Image_Dataset\Rice_Image_Dataset"
MODEL_SAVE    = r"D:\grain analysis\rice_mobilenet_model.h5"
IMAGE_SIZE    = (128, 128)   # best balance: detail vs. speed
BATCH_SIZE    = 64           # larger batch = faster on CPU
AUTOTUNE      = tf.data.AUTOTUNE
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 15

# ── Load all file paths & labels ────────────────────────────────
images, labels = [], []
for cls in sorted(os.listdir(DATASET_PATH)):
    cls_path = os.path.join(DATASET_PATH, cls)
    if not os.path.isdir(cls_path):
        continue
    for fname in os.listdir(cls_path):
        fpath = os.path.join(cls_path, fname)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(fpath)
            labels.append(cls)

df = pd.DataFrame({"image": images, "label": labels})
print(f"\nTotal images: {len(df)}")
print(df["label"].value_counts())

# ── Encode labels ───────────────────────────────────────────────
class_names = sorted(df["label"].unique())
num_classes = len(class_names)
label2idx   = {c: i for i, c in enumerate(class_names)}
df["label_idx"] = df["label"].map(label2idx)
print(f"\nClasses: {label2idx}")

# ── Train / Val split ───────────────────────────────────────────
df_train, df_val = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"Train: {len(df_train)}  |  Val: {len(df_val)}")

# ── Class weights ───────────────────────────────────────────────
cw_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.array(class_names),
    y=df_train["label"].values
)
class_weights = dict(enumerate(cw_arr))
print("Class weights:", {class_names[i]: f"{v:.2f}" for i, v in class_weights.items()})

# ── Fast tf.data pipeline ───────────────────────────────────────
H, W = IMAGE_SIZE

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [H, W])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def make_dataset(df_part, training=True):
    paths  = tf.constant(df_part["image"].values)
    idxs   = tf.constant(df_part["label_idx"].values, dtype=tf.int32)
    labels = tf.one_hot(idxs, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(2000, seed=42)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)   # overlap CPU preprocessing + model inference
    return ds

print("\nBuilding tf.data pipelines...")
train_ds = make_dataset(df_train, training=True)
val_ds   = make_dataset(df_val,   training=False)

steps_per_epoch = len(df_train) // BATCH_SIZE
val_steps       = len(df_val)   // BATCH_SIZE

print(f"Steps/epoch: {steps_per_epoch}  |  Val steps: {val_steps}")

# ── Build MobileNetV2 ────────────────────────────────────────────
print("\nBuilding MobileNetV2 model...")
base_model = MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # frozen for Phase 1

x       = base_model.output
x       = GlobalAveragePooling2D()(x)
x       = BatchNormalization()(x)
x       = Dense(256, activation="relu")(x)
x       = Dropout(0.4)(x)
x       = Dense(128, activation="relu")(x)
x       = Dropout(0.3)(x)
outputs = Dense(num_classes, activation="softmax")(x)
model   = Model(base_model.input, outputs)

total_params = model.count_params()
print(f"Total params: {total_params:,}")

# ── Phase 1: Train classification head only ──────────────────────
print("\n" + "="*55)
print("PHASE 1: Training head  (backbone frozen)")
print("="*55)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

p1_callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
]

h1 = model.fit(
    train_ds,
    epochs=PHASE1_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=p1_callbacks
)

print(f"\nPhase 1 best val_accuracy: {max(h1.history['val_accuracy'])*100:.1f}%")

# ── Phase 2: Fine-tune top backbone layers ───────────────────────
print("\n" + "="*55)
print("PHASE 2: Fine-tuning backbone (last 40 layers)")
print("="*55)
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"Trainable layers: {trainable_count}")

model.compile(
    optimizer=Adam(learning_rate=5e-5),   # lower LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

p2_callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
    ModelCheckpoint(
        MODEL_SAVE, monitor="val_accuracy",
        save_best_only=True, verbose=1
    )
]

h2 = model.fit(
    train_ds,
    epochs=PHASE2_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=p2_callbacks
)

# ── Save & report ────────────────────────────────────────────────
model.save(MODEL_SAVE)
best_acc = max(h1.history["val_accuracy"] + h2.history["val_accuracy"])
print("\n" + "="*55)
print(f"✅ Model saved → {MODEL_SAVE}")
print(f"   Input shape     : {model.input_shape}")
print(f"   Best val acc    : {best_acc*100:.2f}%")
print(f"   Class mapping   :")
for i, c in enumerate(class_names):
    print(f"     {i}: {c}")
print("="*55)

# ── Plot training curves ─────────────────────────────────────────
all_acc  = h1.history["accuracy"]  + h2.history["accuracy"]
all_vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
epochs_x = range(1, len(all_acc) + 1)
split    = len(h1.history["accuracy"])

plt.figure(figsize=(10, 5))
plt.plot(epochs_x, all_acc,  "o-", label="Train Accuracy")
plt.plot(epochs_x, all_vacc, "s-", label="Val Accuracy")
plt.axvline(x=split + 0.5, color="gray", linestyle="--", label="Fine-tune start")
plt.title("MobileNetV2 — Rice Classification (Full Dataset)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r"D:\grain analysis\training_accuracy.png")
print("📊 Plot saved → training_accuracy.png")
plt.show()
