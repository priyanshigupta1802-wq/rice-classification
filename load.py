"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')

# =========================
# Dataset Path
# =========================
dataset_path = r"D:/grain analysis/Rice_Image_Dataset/Rice_Image_Dataset"

sns.set_theme(style="darkgrid")

# =========================
# Load Images & Labels
# =========================
images = []
labels = []

for subfolder in os.listdir(dataset_path):
    subfolder_path = os.path.join(dataset_path, subfolder)

    if not os.path.isdir(subfolder_path):
        continue

    for image_filename in os.listdir(subfolder_path):
        image_path = os.path.join(subfolder_path, image_filename)
        images.append(image_path)
        labels.append(subfolder)

# Create DataFrame
df = pd.DataFrame({'image': images, 'label': labels})

# =========================
# Visualize Class Distribution
# =========================
plt.figure(figsize=(8,5))
ax = sns.countplot(x=df['label'])

ax.set_xlabel("Class Name")
ax.set_ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.show()

# =========================
# Show Sample Images
# =========================
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(15, 15))
gs = GridSpec(5, 4, figure=fig)

for i, category in enumerate(df['label'].unique()):
    filepaths = df[df['label'] == category]['image'].values[:4]

    for j, filepath in enumerate(filepaths):
        ax = fig.add_subplot(gs[i, j])
        ax.imshow(plt.imread(filepath))
        ax.axis('off')

    ax.text(20, 20, category, fontsize=14, color='blue')

plt.show()

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df['image'], df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']   # better split
)

df_train = pd.DataFrame({'image': X_train, 'label': y_train})
df_test = pd.DataFrame({'image': X_test, 'label': y_test})

# =========================
# Image Generators
# =========================
image_size = (64, 64)   # slightly better than 50x50
batch_size = 32

# Training generator (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

# Test generator (NO augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    df_train,
    x_col='image',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    df_test,
    x_col='image',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# =========================
# Model
# =========================
input_shape = (64, 64, 3)
#num_classes = train_generator.num_classes
num_classes = len(train_generator.class_indices)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),   # prevents overfitting

    Dense(num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# Training
# =========================
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# =========================
# Plot Accuracy
# =========================
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], marker='o')
plt.plot(history.history['val_accuracy'], marker='h')
plt.title('Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# =========================
# Plot Loss
# =========================
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], marker='o')
plt.plot(history.history['val_loss'], marker='h')
plt.title('Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

# =========================
# Evaluation
# =========================
loss, accuracy = model.evaluate(test_generator)
print("Test Accuracy:", accuracy)

# =========================
# Save Model
# =========================
model.save("rice_classification_model.h5")
print("Model saved successfully!")
#=========================================================
"""
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("rice_classification_model.h5")

# Class labels (same order as training folders)
classes = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    # Resize to model input size (usually 224x224)
    # img = cv2.resize(frame, (224, 224))
    img = cv2.resize(frame, (128, 128))
    img = np.reshape(img, (1, 128, 128, 3))
    img = img / 255.0  # Normalize
   # img = np.reshape(img, (1, 224, 224, 3))

    # Prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    label = classes[class_index]
    confidence = np.max(prediction)

    # Display text on screen
    text = f"{label} ({confidence*100:.2f}%)"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    # Show video
    cv2.imshow("Rice Classification", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(model.input_shape)