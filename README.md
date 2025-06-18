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

### Why These Parameters?

| Parameter      | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| model=best.pt  | Your trained YOLOv8 model                                               |
| format=onnx    | ONNX format for cross-platform/edge deployment                          |
| simplify=True  | Removes redundant ops for faster inference                              |
| dynamic=False  | Fixed input size (640x640) for speed                                    |
| nms=True       | Built-in Non-Maximum Suppression in ONNX                                |
| imgsz=640      | Input image size (must match inference size)                            |
| opset=12       | ONNX opset 12 for best compatibility                                    |

---

## ⚠️ Input Size Matching
If you export with `imgsz=640`, resize your inference frames to (640, 640):

## ⚠️ Important Matching
If you use imgsz=640, your inference frame must also be resized to (640, 640) like:

```
img = cv2.resize(frame, (640, 640))
```
---

# 🧪 Step 3: Real-Time Inference using ONNX + OpenCV

This step runs the real-time face detection using a YOLOv8 model exported to ONNX. We use `onnxruntime` for inference and `OpenCV` for webcam access and visualization.

### 🔧 Installation

Install required Python packages:

```bash
pip install opencv-python onnxruntime numpy
```
---

**Ensure:**
- You have `best.onnx` exported.
- Python 3.7 or higher.

---

### 🧠 Pipeline Overview

**Preprocessing**
- Resize to 640x640
- Convert BGR → RGB
- Normalize to [0, 1]
- Transpose to (C, H, W), add batch dim

**Postprocessing**
- Read model output: shape (N, 6)
- Apply confidence threshold (`conf_thres=0.2`)
- Rescale box coordinates to original image
- Draw bounding boxes and labels

**Inference Loop**
- Read webcam frames
- Detect faces in real-time
- Press 'q' to quit

---

### ⚙️ Key Parameters

| Parameter     | Purpose                                      | Recommendation         |
|---------------|----------------------------------------------|------------------------|
| imgsz=640     | Input size for model and inference           | 320 (faster), 640 (default) |
| conf_thres    | Confidence threshold for detections          | 0.2 (adjust as needed) |
| nms=True      | Built-in NMS in ONNX                         | Keep True              |
| simplify=True | Optimized model graph                        | Required for speed     |
| dynamic=False | Fixed input size                             | Best runtime perf      |

---

### ✅ Example Output

- Webcam feed with bounding boxes labeled "Face: XX.X%"
- ONNX output shapes printed per frame
- Press 'q' to exit

---

## 🔗 Next Steps

- Integrate Dlib-based 128D face embedding extractor
- Create face database (SQLite or Pickle)
- Match embeddings (cosine/Euclidean distance)
- Display recognized names in real-time


> You need a working webcam and a Raspberry Pi 5 or similar edge device for deployment.



