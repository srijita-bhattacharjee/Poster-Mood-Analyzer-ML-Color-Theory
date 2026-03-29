# Poster Mood Analyzer using Machine Learning and Color Theory

A machine learning project that analyzes the mood of posters based on color composition using RGB/HSV features and a Random Forest model.

## 🚀 Overview
This project predicts the emotional tone of posters by extracting color-based features and applying a trained classification model. It combines concepts from **color theory** and **machine learning** to map visual patterns to human-perceived moods.

## 🧠 Features
- Mood classification using Random Forest (85–90% accuracy)
- Feature extraction using RGB and HSV color spaces
- Confidence-based predictions
- Interpretable model based on color distributions

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** scikit-learn, OpenCV, NumPy, Pandas  
- **Model:** Random Forest Classifier  

## 📊 Dataset
- Custom dataset of labeled posters
- Features engineered using:
  - RGB values
  - HSV values
  - Color intensity and distribution

## ⚙️ How It Works
1. Input poster image  
2. Extract color features (RGB + HSV)  
3. Pass features into trained model  
4. Output predicted mood with confidence score  

## 📁 Project Structure
