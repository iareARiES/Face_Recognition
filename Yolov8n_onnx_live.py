import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
onnx_model_path = "best.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name

print("input_name:",input_name)

# Preprocessing
def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  #Transposes shape from (H, W, C) → (C, H, W). as YOLOV8 expects in that way (C, H, W) → Channels x Height x Width (RGB, float32)
    img = np.expand_dims(img, axis=0).copy()   # [Channels, Height, Width] ----> # [Batch, Channels, Height, Width]
    return img


# Postprocessing for SINGLE class model (Face)
# Purpose: Extracts boxes from model output and overlays them on the frame.
def postprocess(outputs, orig_frame, conf_thres=0.2):
    predictions = outputs[0][0]  # (300, 6)
    orig_h, orig_w = orig_frame.shape[:2]

    scale_x = orig_w / 640
    scale_y = orig_h / 640

    face_crops = []

    for pred in predictions:
        obj_conf = pred[4]
        if obj_conf < conf_thres:
            continue

        x1 = int(pred[0] * scale_x)
        y1 = int(pred[1] * scale_y)
        x2 = int(pred[2] * scale_x)
        y2 = int(pred[3] * scale_y)

        # Bounds check
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        # Crop face
        face_crop = orig_frame[y1:y2, x1:x2].copy()
        if face_crop.size > 0:
            face_crops.append(face_crop)

        # Draw box
        score_text = f"Face: {obj_conf * 100:.1f}%"
        cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(orig_frame, score_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return orig_frame, face_crops


# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

last_num_faces = 0  # Track number of face windows in the last frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  #mirror flip
    
    input_tensor = preprocess(frame)  #does all the preprocessing 
    '''Converts the image to:
    640x640,RGB,Float32, normalized,Transposed to [1, 3, 640, 640]
    This is the format your ONNX model expect'''
    
    outputs = session.run(None, {input_name: input_tensor})    #running the inferance , Feeds the preprocessed tensor into the model.
    
    '''print(f"ONNX Output shape: {outputs[0].shape}")
    print(outputs)'''

    annotated, face_crops = postprocess(outputs, frame)
    
    print("last_num_faces before: ",last_num_faces)
    # === 🆕 Close any leftover windows from previous frame ===
    if last_num_faces > len(face_crops):
        for i in range(len(face_crops), last_num_faces):
            print("last_num_faces after: ",last_num_faces)
            cv2.destroyWindow(f"Face {i+1}")
    print("last_num_faces after again: ",last_num_faces)
    last_num_faces = len(face_crops)  # update count
    

# For now, show the first detected face crop
    for i, face_crop in enumerate(face_crops):
    	cv2.imshow(f"Face {i+1}", face_crop)

    '''This line takes the raw model outputs (outputs) and the original image/frame (frame), and processes them to:
    - Filter detections (e.g., confidence > 0.2)
    - Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    - Draw bounding boxes and labels on the frame (usually with cv2.rectangle() and cv2.putText())'''
    
    #annotated is the same image frame, but with boxes and labels drawn on detected objects — ready for display.

    cv2.imshow("YOLOv8 ONNX Face Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
