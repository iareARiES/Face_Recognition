# 🎯 Face Detection and Recognition System using YOLOv8 and Dlib

This project combines **YOLOv8** for high-accuracy face detection and **Dlib** for face recognition by extracting 128-dimensional facial embeddings. It’s optimized for real-time use and edge devices like **Raspberry Pi 5**.

---

## 🚀 Features

- 🧠 **YOLOv8** for fast and accurate face detection
- 🧷 **Dlib 128D embeddings** for face recognition
- 🗃️ Store encodings using **SQLite** or **Pickle**
- 🧪 Face matching using **Euclidean/Cosine distance**
- 🧩 Works on custom datasets and real-time webcam feeds
- 🧑‍💻 Runs on **Raspberry Pi 5** and low-power edge devices

---

## ❓ Why YOLOv8 instead of OpenCV ?

- YOLOv8 is **deep learning-based** and more accurate than Haar cascades.
- Detects **multiple faces** in multiple orientations.
- Suitable for **edge deployment** due to `yolov8n` (nano version) optimization.

---

## 📁 Project Stages

1. ✅ Train YOLOv8 on the WIDER FACE dataset  
2. ✅ Extract and save trained face detector (`best.pt`)  
3. 🔜 Integrate Dlib for recognition using face embeddings  
4. 🔜 Add face encoding database & matching  
5. 🔜 Real-time demo with live camera input  

---

## 📦 Dataset Download & Preparation

We use the **WIDER FACE** dataset available via KaggleHub.

```python
import os
import kagglehub

# Set KaggleHub cache directory
os.environ["KAGGLEHUB_CACHE"] = "/content/Dataset"

# Download WIDER FACE dataset
path = kagglehub.dataset_download("lylmsc/wider-face-for-yolo-training") 
```
---
# Run this in terminal to export the code in ONNX fomrat 

```
  yolo export \
  model=best.pt \
  format=onnx \
  simplify=True \
  dynamic=False \
  nms=True \
  imgsz=640 \
  opset=12
```


