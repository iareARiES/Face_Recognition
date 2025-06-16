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
## Under Progress 

