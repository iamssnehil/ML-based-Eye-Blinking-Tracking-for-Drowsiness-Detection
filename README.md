# Fatigue-Detection-System

This project uses a Convolutional Neural Network (CNN) to detect drowsiness in real-time by classifying images of the user's face into two categories: Active and Sleepy. The model predicts whether a person is awake or drowsy based on eye movements and facial expressions. The model was trained on a dataset of images and tested to predict whether the person is actively alert or showing signs of drowsiness.

## Overview
This project aims to detect drowsiness in individuals, specifically focusing on facial features like eye closures. The model works in two primary modes:
1. Real-time Inference: Using a webcam feed to predict the user's drowsiness status (Active or Sleepy).
2. Offline Evaluation: Evaluate the model's performance on a labeled test dataset.
The model was trained using a CNN architecture, optimized to detect facial features from images of various conditions (lighting, angles, etc.).

## Requirements
### Software:
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib (for visualization)
- scikit-learn (for model evaluation)
- playsound (for alarm)
### Hardware:
- Webcam for real-time detection.
- GPU (optional but recommended for faster training).

## Model Architecture

The CNN model consists of the following layers:
- Input Layer: Accepts a 224x224 RGB image.
- Convolutional Layers: Multiple convolution layers with ReLU activation to detect facial features.
- Pooling Layers: Max pooling layers to reduce dimensionality.
- Fully Connected Layers: Dense layers to interpret the features.
- Output Layer: A sigmoid output to predict a binary classification (Active vs Sleepy).

## Training the Model
### Training Steps:
- "preprocessing.ipynb" will Load and preprocess the training dataset (resizing images to 224x224 pixels, normalizing).
- "Training.ipynb" will Train the model with an appropriate batch size and number of epochs.
- Save the trained model as "fatigue_model_cnn.h5".
### Testing the Model
- Load the trained model (fatigue_model_cnn.h5).
- Evaluate the model on a separate test set (preprocessed similarly as training data).
- Calculate precision, recall, F1 score, and generate an ROC curve.
## Real-Time Inference
"real_time.ipynb" uses OpenCV to capture frames from a webcam. The CNN model is used to predict the user's state (Active or Sleepy) based on eye movement and facial expressions.

### Features:
- Face and Eye Detection: Uses OpenCV's Haar cascades for face and eye detection.
- Real-Time Classification: Classifies the user as "Active" or "Sleepy" and displays the result on the webcam feed.
- Alarm: An alarm will sound if the model detects 5 consecutive "Sleepy" frames.
