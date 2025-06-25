import cv2
import numpy as np
import onnxruntime as ort
import insightface
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from insightface.utils.face_align import norm_crop

# === YOLOv8 Face Detection (ONNX) ===
yolo_model_path = "best.onnx"
yolo_session = ort.InferenceSession(yolo_model_path)
yolo_input_name = yolo_session.get_inputs()[0].name

def preprocess_yolo(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).copy()
    return img

def postprocess_yolo(outputs, orig_frame, conf_thres=0.3):
    predictions = outputs[0][0]
    orig_h, orig_w = orig_frame.shape[:2]
    scale_x = orig_w / 640
    scale_y = orig_h / 640

    boxes = []
    for pred in predictions:
        if pred[4] < conf_thres:
            continue
        x1 = int(pred[0] * scale_x)
        y1 = int(pred[1] * scale_y)
        x2 = int(pred[2] * scale_x)
        y2 = int(pred[3] * scale_y)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        boxes.append((x1, y1, x2, y2))
    return boxes

# === InsightFace Setup ===
model_dir = "E:/CSIR_2/buffalo_s"
recognizer = insightface.model_zoo.get_model(os.path.join(model_dir, 'w600k_mbf.onnx'))
recognizer.prepare(ctx_id=0)
landmarker = insightface.model_zoo.get_model(os.path.join(model_dir, '2d106det.onnx'))
landmarker.prepare(ctx_id=0)

# === Face DB ===
db_path = "face_db.pkl"
if os.path.exists(db_path):
    with open(db_path, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

next_id = len(face_db)

# === Embedding Helper ===
def get_embedding_aligned(face_img):
    face = cv2.resize(face_img, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = np.transpose(face, (2, 0, 1)).astype(np.float32)
    face /= 255.0
    face = np.expand_dims(face, axis=0)
    return recognizer.get_embedding(face).squeeze()

def recognize_or_add(embedding):
    global next_id
    if len(face_db) == 0:
        face_db[f"Person{next_id}"] = embedding
        next_id += 1
        return f"🆕 New face saved as Person{next_id - 1}"
    names = list(face_db.keys())
    embs = np.array(list(face_db.values()))
    sims = cosine_similarity([embedding], embs)[0]
    best_idx = np.argmax(sims)
    if sims[best_idx] > 0.6:
        return f"✅ Recognized as {names[best_idx]}"
    else:
        face_db[f"Person{next_id}"] = embedding
        next_id += 1
        return f"🆕 New face saved as Person{next_id - 1}"

# === Camera Loop ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera error.")
    exit()

print("🔍 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    input_tensor = preprocess_yolo(frame)
    outputs = yolo_session.run(None, {yolo_input_name: input_tensor})
    boxes = postprocess_yolo(outputs, frame)

    for (x1, y1, x2, y2) in boxes:
        face_crop = frame[y1:y2, x1:x2].copy()
        if face_crop.size == 0:
            continue

        # Align face using bounding box
        landmark = landmarker.get(frame, np.array([[x1, y1, x2, y2]]))

        if landmark is None or len(landmark) == 0:
            continue
        aligned = norm_crop(frame, landmark[0])

        embedding = get_embedding_aligned(aligned)
        result = recognize_or_add(embedding)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, result, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO + Aligned Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

with open(db_path, "wb") as f:
    pickle.dump(face_db, f)

cap.release()
cv2.destroyAllWindows()
