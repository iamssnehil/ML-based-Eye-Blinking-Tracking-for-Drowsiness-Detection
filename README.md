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
