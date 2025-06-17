# Face Detection and Recognition System using YOLOv8 and Dlib

This project combines **YOLOv8** for face detection and **Dlib** for face recognition by extracting 128-dimensional embeddings. It's a lightweight, accurate system suitable for custom datasets and real-time applications.

---

## 🚀 Features

- 🧠 **Custom-trained YOLOv8** face detector (on Wider Face or your dataset)
- 🧷 **Dlib embeddings** for accurate face recognition
- 📝 Store face encodings in **SQLite** or **Pickle**
- 🧪 Face matching using **Euclidean/Cosine distance**
- 💡 Minimal dependencies, simple setup

---
## Why YOLO not OpenCV?

- We aimed for better accuraccy so YOLO being a Deep Learning algo does that.
- Even works with multiple faces at multiple angles.
- This project was run on Raspberry Pi 5 board so better optimization and runtime.

---
## 📁 Step-by-Step Instructions

# 🧠 YOLOv8 Face Detection on WIDER FACE Dataset

This repository demonstrates how to train a YOLOv8 model for face detection using the [WIDER FACE dataset](https://www.kaggle.com/datasets/lylmsc/wider-face-for-yolo-training). It includes dataset preparation, training, and inference using the `ultralytics` YOLOv8 library.

---

## 📦 Dataset Download & Preparation

```python
import os
import kagglehub

# Set KaggleHub cache directory
os.environ["KAGGLEHUB_CACHE"] = "/content/Dataset"

# Download WIDER FACE dataset
path = kagglehub.dataset_download("lylmsc/wider-face-for-yolo-training")



## ⚙️ Requirements for training

Install the necessary libraries:
```bash
pip install ultralytics kagglehub scikit-learn

