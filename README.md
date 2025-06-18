# 🎯 Face Detection and Recognition System using YOLOv8 and Dlib

This project combines **YOLOv8** for high-accuracy face detection and face recognition by extracting 128-dimensional facial embeddings. It’s optimized for real-time use and edge devices like **Raspberry Pi 5**.

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
3. ✅ Compile the model to ONNX for running on FPGA or other SoC's 
4. 🔜 Integrate library recognition using face embeddings  
5. 🔜 Add face encoding database & matching  
6. 🔜 Real-time demo with live camera input  

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
# Run this in terminal to export the code in ONNX format 

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
or
```
yolo export model=best.pt format=onnx simplify=True dynamic=False nms=True imgsz=640 opset=12
```

# WHY?

This is configured to export a YOLOv8 model to ONNX with specific parameters that make it easier to use in real-time applications like webcam face detection. Here's the reason for each flag:

🔍 Breakdown of Each Parameter
Parameter	Why it matters
- **model=best.pt**	This is your trained PyTorch YOLOv8 model to be exported.
- **format=onnx**	You want to use the model with onnxruntime, so ONNX is the required format.
- **simplify=True**	Removes redundant ops from the graph to speed up inference and reduce file size.
- **dynamic=False**	Input size is fixed (640×640). Fixed-size models run faster on ONNXRuntime.
- **nms=True**	Exports the model with Non-Maximum Suppression (NMS) built-in to ONNX.
- **imgsz=640**	YOLO expects 640×640 inputs. This must match your runtime input size.
- **opset=12**	Ensures compatibility with ONNX opset version 12 (commonly supported by runtime engines).

## 🚀 Why This Combo?
nms=True: You don't need to manually run NMS in Python, simplifies post-processing.
simplify=True: Optimized for low-latency, edge devices, and webcam inference.
dynamic=False: Ensures fast execution as ONNX can optimize better for fixed shapes.

# ⚠️ Important Matching
If you use imgsz=640, your inference frame must also be resized to (640, 640) like:

```
img = cv2.resize(frame, (640, 640))
```

