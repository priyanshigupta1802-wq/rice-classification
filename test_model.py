"""Test the model on actual dataset images to diagnose prediction issues."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import sys
# Write output to file
outfile = open(r"D:\grain analysis\test_results.txt", "w", encoding="utf-8")
def log(msg=""):
    print(msg)
    outfile.write(msg + "\n")

import tensorflow as tf
import numpy as np
from PIL import Image
from collections import Counter

MODEL_PATH = r"D:\grain analysis\rice_mobilenet_model.h5"
DATASET_PATH = r"D:\grain analysis\Rice_Image_Dataset\Rice_Image_Dataset"
IMAGE_SIZE = (128, 128)
classes = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

log("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
log(f"Input shape: {model.input_shape}")
log(f"Output shape: {model.output_shape}")

# Test 1: with /255 normalization (what training used)
log("\n" + "="*60)
log("TEST 1: Using /255.0 normalization (training method)")
log("="*60)

correct = 0
total = 0

for true_class in classes:
    cls_dir = os.path.join(DATASET_PATH, true_class)
    files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    test_files = files[:10]
    
    predictions_for_class = []
    for fname in test_files:
        fpath = os.path.join(cls_dir, fname)
        img = Image.open(fpath).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        
        pred = model.predict(arr, verbose=0)[0]
        pred_class = classes[np.argmax(pred)]
        predictions_for_class.append(pred_class)
        
        if pred_class == true_class:
            correct += 1
        total += 1
    
    counts = Counter(predictions_for_class)
    log(f"\nTrue: {true_class} -> Predictions: {dict(counts)}")
    
    # Show scores for first image
    fpath = os.path.join(cls_dir, test_files[0])
    img = Image.open(fpath).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr, verbose=0)[0]
    scores = ", ".join([f"{c}:{pred[i]*100:.1f}%" for i,c in enumerate(classes)])
    log(f"  Scores: {scores}")

log(f"\nAccuracy (/255): {correct}/{total} = {correct/total*100:.1f}%")

# Test 2: with MobileNetV2 preprocess_input
log("\n" + "="*60)
log("TEST 2: Using preprocess_input [-1,1]")
log("="*60)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

correct2 = 0
total2 = 0

for true_class in classes:
    cls_dir = os.path.join(DATASET_PATH, true_class)
    files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    test_files = files[:10]
    
    predictions_for_class = []
    for fname in test_files:
        fpath = os.path.join(cls_dir, fname)
        img = Image.open(fpath).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        arr = np.array(img).astype('float32')
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, 0)
        
        pred = model.predict(arr, verbose=0)[0]
        pred_class = classes[np.argmax(pred)]
        predictions_for_class.append(pred_class)
        
        if pred_class == true_class:
            correct2 += 1
        total2 += 1
    
    counts = Counter(predictions_for_class)
    log(f"\nTrue: {true_class} -> Predictions: {dict(counts)}")

log(f"\nAccuracy (preprocess_input): {correct2}/{total2} = {correct2/total2*100:.1f}%")

outfile.close()
log("Done! Results written to test_results.txt")
