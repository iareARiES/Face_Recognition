import cv2
import numpy as np
import onnxruntime as ort
import os
import pickle
import time
from sklearn.metrics.pairwise import cosine_similarity

# === Load YOLOv8 face detection model ===
yolo_model_path = "best.onnx"
yolo_session = ort.InferenceSession(yolo_model_path)
yolo_input_name = yolo_session.get_inputs()[0].name

# === Load face recognition model ===
rec_model_path = "w600k_mbf.onnx"
rec_session = ort.InferenceSession(rec_model_path)
rec_input_name = rec_session.get_inputs()[0].name

# === Load or create face database ===
db_path = "face_db.pkl"
if os.path.exists(db_path):
    with open(db_path, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}
next_id = len(face_db) + 1

# === Face recognition parameters ===
pending_faces = {}  # id: {embeddings: [], last_seen: timestamp, box: (x1, y1, x2, y2)}
embedding_threshold = 0.5
samples_required = 15
face_timeout = 2.0  # seconds

# === Preprocessing ===
def preprocess_yolo(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).copy()
    return img

def preprocess_face(face_img):
    face = cv2.resize(face_img, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)
    return face

# === Recognition logic ===
def recognize_face(embedding):
    for name, known_emb in face_db.items():
        sim = cosine_similarity([embedding], [known_emb])[0][0]
        if sim > embedding_threshold:
            return name
    return None

# === Postprocessing ===
def postprocess(outputs, orig_frame, conf_thres=0.2):
    predictions = outputs[0][0]
    orig_h, orig_w = orig_frame.shape[:2]
    scale_x = orig_w / 640
    scale_y = orig_h / 640
    boxes = []

    for pred in predictions:
        obj_conf = pred[4]
        if obj_conf < conf_thres:
            continue

        x1 = int(pred[0] * scale_x)
        y1 = int(pred[1] * scale_y)
        x2 = int(pred[2] * scale_x)
        y2 = int(pred[3] * scale_y)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        boxes.append((x1, y1, x2, y2))
        cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return orig_frame, boxes

# === Webcam loop ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    input_tensor = preprocess_yolo(frame)
    outputs = yolo_session.run(None, {yolo_input_name: input_tensor})
    annotated, boxes = postprocess(outputs, frame)
    now = time.time()

    for (x1, y1, x2, y2) in boxes:
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
            continue

        embedding_input = preprocess_face(face_crop)
        output = rec_session.run(None, {rec_input_name: embedding_input})
        embedding = output[0][0]
        embedding = embedding / np.linalg.norm(embedding)

        name = recognize_face(embedding)
        if name:
            print(f"✅ Face recognized as {name}")
            cv2.putText(annotated, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            continue

        # Match to pending_faces
        matched = False
        for pid, data in list(pending_faces.items()):
            sim = cosine_similarity([embedding], [np.mean(data['embeddings'], axis=0)])[0][0]
            if sim > 0.8:
                data['embeddings'].append(embedding)
                data['last_seen'] = now
                matched = True
                print(f"📸 Collecting samples for new face {pid} ({len(data['embeddings'])})")
                if len(data['embeddings']) >= samples_required:
                    final_embedding = np.mean(data['embeddings'], axis=0)
                    name = f"Person{next_id}"
                    face_db[name] = final_embedding
                    next_id += 1
                    with open(db_path, "wb") as f:
                        pickle.dump(face_db, f)
                    print(f"✅ New face saved as {name}")
                    del pending_faces[pid]
                break

        if not matched:
            pid = f"pending_{len(pending_faces)+1}"
            pending_faces[pid] = {
                'embeddings': [embedding],
                'last_seen': now,
                'box': (x1, y1, x2, y2)
            }
            print("👀 New face seen. Starting snapshot collection...")

    # Cleanup old pending faces
    for pid in list(pending_faces):
        if now - pending_faces[pid]['last_seen'] > face_timeout:
            del pending_faces[pid]

    cv2.imshow("YOLOv8 Face Detection + Recognition", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
