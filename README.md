# 💡 AI-Based Streetlight Failure Detection & Mapping

An end-to-end Deep Learning project that detects potentially faulty streetlights from images and maps predicted failures using geographic coordinates.

## 🚀 Features

- Custom CNN-based streetlight fault classifier
- Handles severe class imbalance using class weights
- Early stopping for training
- Optimized decision threshold for fault detection
- Precision, Recall, F1-score and confusion matrix evaluation
- GPS-based streetlight failure mapping
- Interactive Streamlit web application
- Public deployment on Streamlit Community Cloud

## 🧠 Model

Input Image → CNN → Fault Probability → Decision Threshold → Normal / Faulty

The model uses 128×128 RGB images and a custom CNN architecture with three convolutional blocks.

## 📊 Results

Test accuracy: **96.69%**

Faulty-class performance after threshold optimization:

- Precision: **58.82%**
- Recall: **66.67%**
- F1-score: **62.50%**

The optimized threshold is **0.4288**.

## 🗺️ Mapping

Predicted faulty streetlights are associated with their latitude and longitude and visualized on an interactive map.

## 🌐 Live Demo

[Open the Streamlit App]((https://dl-streetlight-failure-mapping-fqqmzp2iu9nch8cwcxqcmb.streamlit.app/))

## 📂 Project Structure

```text
DL-streetlight-failure-mapping/
├── app/
│   └── app.py
├── models/
│   ├── streetlight_cnn.keras
│   └── best_threshold.txt
├── failure_map.csv
├── Streetlight_Failure_Mapping.ipynb
├── requirements.txt
└── README.md
