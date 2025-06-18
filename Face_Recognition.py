import cv2
import numpy as np
import onnxruntime as ort
import pickle
import os
from pathlib import Path

# --------------------- CONFIG ----------------------
YOLO_ONNX_PATH = "yolov8n_int8.onnx"  # Quantized YOLO model
FACENET_ONNX_PATH = "facenet_int8.onnx"  # Quantized FaceNet model
EMBEDDINGS_FILE = "embeddings.pkl"
CONFIDENCE_THRESHOLD = 0.4
RECOGNITION_THRESHOLD = 1.0

# ------------------ LOAD MODELS --------------------
print("[INFO] Loading models...")
yolo_sess = ort.InferenceSession(YOLO_ONNX_PATH)
facenet_sess = ort.InferenceSession(FACENET_ONNX_PATH)

def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}, 0

known_faces, next_id = load_embeddings()

# --------------- YOLOv8 HELPERS --------------------
def preprocess_yolo(img):
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = img_resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img_transposed, axis=0)

def postprocess_yolo(output, original_shape):
    predictions = output[0]  # N x 6 (x1, y1, x2, y2, conf, class)
    results = []
    for det in predictions:
        if det[4] >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, det[:4])
            h, w = original_shape
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            results.append((x1, y1, x2, y2))
    return results

# -------------- FACENET HELPERS --------------------
def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = (face - 127.5) / 128.0
    face = face.transpose(2, 0, 1)
    return np.expand_dims(face.astype(np.float32), axis=0)

def get_embedding(face):
    input_name = facenet_sess.get_inputs()[0].name
    return facenet_sess.run(None, {input_name: preprocess_face(face)})[0][0]

def recognize(embedding):
    global next_id
    for label, db_embedding in known_faces.items():
        dist = np.linalg.norm(embedding - db_embedding)
        if dist < RECOGNITION_THRESHOLD:
            return label
    # New person
    label = f"Person {next_id}"
    known_faces[label] = embedding
    next_id += 1
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(known_faces, f)
    return label

# ------------------ MAIN LOOP ----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam not found.")
    exit()

print("[INFO] Running face detection and recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_input = preprocess_yolo(frame)
    input_name = yolo_sess.get_inputs()[0].name
    detections = yolo_sess.run(None, {input_name: img_input})
    boxes = postprocess_yolo(detections, frame.shape[:2])

    for (x1, y1, x2, y2) in boxes:
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
        embedding = get_embedding(face)
        label = recognize(embedding)

        # Annotate
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
