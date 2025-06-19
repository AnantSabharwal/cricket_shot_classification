# 🏏 Cricket Shot Classification using Deep Learning

This repository contains the **code and project report** for classifying cricket shots using various deep learning models. The goal is to recognize and differentiate between different types of cricket shots using both image and sequence-based inputs.

---

## 📌 About the Project

Cricket shot classification is a challenging computer vision task involving temporal and spatial pattern recognition. This project explores **deep learning-based approaches** to classify shots such as:

- Cover Drive  
- Pull Shot  
- Sweep  
- Straight Drive  
- Cut Shot  
*(Exact classes depend on the dataset used)*

The project includes both **image-based** and **video-based** models to capture spatial and temporal information.

---

## 🧠 Models Implemented

We experimented with and compared the following models:

| Model            | Type                | Description |
|------------------|---------------------|-------------|
| `CNN_GRU`        | Sequential           | A CNN for spatial feature extraction + GRU for temporal sequence modeling 
| `DCNN_GRU`       | Graph-Based + GRU    | Diffusion Convolutional Neural Network combined with GRU for capturing graph-based spatial relationships and sequence dynamics |
| `VGG16 (4 layers)` | Transfer Learning | VGG16 with 4 unfrozen layers for fine-tuning |
| `VGG16 (8 layers)` | Transfer Learning | VGG16 with 8 unfrozen layers for deeper fine-tuning |
| `LRCN`           | Video Model          | Long-term Recurrent Convolutional Network (CNN + LSTM) |
| `ConvLSTM`       | Spatio-temporal      | Convolutional LSTM directly on video frame sequences |
