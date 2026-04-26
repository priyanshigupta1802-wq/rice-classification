# rice-classification
# 🌾 Rice Grain Classification System (Backend)

## Overview

This project is a **Deep Learning-based Rice Grain Classification System** that identifies different types of rice grains from images using a trained **MobileNetV2 model**.

The backend is built using **Flask** and provides an API to upload rice grain images and receive predictions.

---
## Dataset 

https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

## Features

* Classifies rice into 5 categories:

  * Arborio
  * Basmati
  * Ipsala
  * Jasmine
  * Karacadag
* REST API for prediction
* Deep Learning model using Transfer Learning
* Fast and lightweight inference
* Supports image upload via frontend

---

## Model Details

* Model: MobileNetV2 (Transfer Learning)
* Input Size: 96 × 96
* Training Data: Rice Image Dataset 
* Classes: 5
* Loss Function: Categorical Crossentropy (with label smoothing)
* Optimizer: Adam

---

## Tech Stack

* Python
* TensorFlow / Keras
* Flask
* NumPy
* Pillow
* Flask-CORS

---

## Project Structure

```
backend/
│
├── app.py                  # Flask API
├── predict.py              # Prediction logic
├── load.py                 # Model loading
├── train_mobilenet.py      # Training script
├── model_evaluate.py       # Evaluation script
├── rice_mobilenet_model.h5 # Trained model
├── rice_classification_model.h5
├── test_model.py
├── test_results.txt
├── confusion_matrix.png
├── training_accuracy.png
└── README.md
```

---

##  How to Run

### 1️⃣ Clone Repository

```
git clone https://github.com/YOUR_USERNAME/rice-classification-backend.git
cd rice-classification-backend
```

---

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Run the Server

```
python app.py
```

Server will start at:

```
http://127.0.0.1:5000
```

---

##  API Usage

### 🔹 Endpoint:

```
POST /predict
```

### 🔹 Request:

* Form-data
* Key: `file`
* Value: Image file

### 🔹 Response:

```
{
  "class": "Basmati",
  "confidence": 97.45
}
```

---

##  Results

* High accuracy using transfer learning
* Confusion matrix and training graphs included

---

## ⚠️ Note

* Dataset is not included due to size constraints
* You can use any rice grain dataset for training

---

##  Future Improvements

* Live camera detection
* Cloud deployment (Render / AWS)
* Add more grain categories
* User authentication & history tracking

---
#  Rice Grain Classification System (Frontend)

##  Overview

This is the **frontend interface** for the Rice Grain Classification System.
It allows users to upload rice grain images and get predictions from the backend model in real time.

Built using **React (Vite)**, the UI is simple, fast, and user-friendly.

---

##  Features

* Upload rice grain images 📷
* Get instant predictions ⚡
* Displays:

  * Predicted rice type
  * Confidence score
* Clean and responsive UI
* Connects to Flask backend API

---

## Tech Stack

* React (Vite)
* JavaScript (ES6+)
* HTML5 & CSS3
* Fetch API / Axios

---
##  Project Structure

```id="u7r9k2"
rice-app/
│
├── public/
├── src/
│   ├── App.jsx
│   ├── main.jsx
│   ├── App.css
│   └── index.css
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

---

##  How to Run

### 1️⃣ Navigate to project

```id="z3s91c"
cd rice-app
```

### 2️⃣ Install dependencies

```id="o2v4p8"
npm install
```

### 3️⃣ Start development server

```id="k1x9lm"
npm run dev
```

App will run at:

```id="r8c0y5"
http://localhost:5173
```

---

## Backend Connection

Make sure backend is running at:

```id="d9p2h1"
http://127.0.0.1:5000
```

### Example API call:

```id="v6t8qw"
POST /predict
```

If needed, update API URL inside your frontend code:

```id="c4n7ab"
http://127.0.0.1:5000/predict
```

---

## How It Works

1. User uploads an image
2. Image is sent to backend API
3. Backend processes using ML model
4. Prediction + confidence returned
5. UI displays result

---

## ⚠️ Note

* Backend must be running before using frontend
* Dataset is not required here
* Ensure CORS is enabled in backend

---

## Future Improvements

* Drag & drop upload
* Live camera detection 📷
* Better UI (animations, loaders)
* Deployment (Vercel)

---

## Author

**Priyanshi Gupta**
BCA Final Year Student

---

## ⭐ If you like this project

Give it a star on GitHub!
