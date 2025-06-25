import cv2
import numpy as np
import onnxruntime as ort
import os
import pickle
import traceback
from sklearn.metrics.pairwise import cosine_similarity

# === Load YOLO Face Detection ONNX ===
yolo_sess = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
yolo_input_name = yolo_sess.get_inputs()[0].name

# === Load Landmark ONNX Model ===
landmark_sess = ort.InferenceSession("2d106det.onnx", providers=['CPUExecutionProvider'])
landmark_input_name = landmark_sess.get_inputs()[0].name

# === Load Face Recognition ONNX ===
rec_sess = ort.InferenceSession("w600k_mbf.onnx", providers=['CPUExecutionProvider'])
rec_input_name = rec_sess.get_inputs()[0].name

# === Face Database ===
db_path = "face_db.pkl"
if os.path.exists(db_path):
    with open(db_path, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}
next_id = len(face_db)

def preprocess_yolo(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
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
        boxes.append((max(0,x1), max(0,y1), min(orig_w,x2), min(orig_h,y2)))
    return boxes

def get_landmarks(face_crop):
    face = cv2.resize(face_crop, (192, 192))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))[np.newaxis, ...]
    out = landmark_sess.run(None, {landmark_input_name: face})[0]
    print("Landmark shape:", out.shape)
    if out.shape[-1] == 212:
        landmarks = out.reshape(-1, 2) * [face_crop.shape[1], face_crop.shape[0]]
    elif out.shape[-2] == 106:
        landmarks = out[0] * [face_crop.shape[1], face_crop.shape[0]]
    else:
        raise ValueError("Unsupported landmark shape")
    print("Landmarks dtype:", type(landmarks), "shape:", landmarks.shape)
    return landmarks

def align_face_by_crop(face_crop, landmarks):
    # Use 5-point eye/nose/mouth landmark-based similarity transform manually
    indices = [30, 36, 45, 48, 54]  # nose tip, eyes, mouth corners (standard in 106-point models)
    landmark_subset = landmarks[indices].astype(np.float32)
    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    M = cv2.estimateAffinePartial2D(landmark_subset, dst, method=cv2.LMEDS)[0]
    aligned = cv2.warpAffine(face_crop, M, (112, 112), borderValue=0.0)
    return aligned

def get_embedding(aligned):
    face = cv2.resize(aligned, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))[np.newaxis, ...]
    emb = rec_sess.run(None, {rec_input_name: face})[0][0]
    return emb / np.linalg.norm(emb)

def recognize_or_add(embedding):
    global next_id
    if len(face_db) == 0:
        face_db[f"Person{next_id}"] = embedding
        next_id += 1
        print(f"Added new Person{next_id - 1} because DB was empty")
        return f"🆕 Person{next_id - 1}"
    names = list(face_db.keys())
    embs = np.array(list(face_db.values()))
    sims = cosine_similarity([embedding], embs)[0]
    print(f"Similarity scores: {sims}")
    best_idx = np.argmax(sims)
    print(f"Best match: {names[best_idx]} with similarity {sims[best_idx]}")
    if sims[best_idx] > 0.6:
        return f"✅ {names[best_idx]}"
    else:
        face_db[f"Person{next_id}"] = embedding
        print(f"Added new Person{next_id} due to low similarity")
        next_id += 1
        return f"🆕 Person{next_id - 1}"


# === Start Camera ===
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
    outputs = yolo_sess.run(None, {yolo_input_name: input_tensor})
    boxes = postprocess_yolo(outputs, frame)

    for (x1, y1, x2, y2) in boxes:
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
            continue
        try:
            landmarks = get_landmarks(face_crop)
            aligned = align_face_by_crop(face_crop, landmarks)
            cv2.imshow("aligned", aligned)
            emb = get_embedding(aligned)
            label = recognize_or_add(emb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception as e:
            print("⚠️ Landmark/Embed fail:")
            traceback.print_exc()
            continue

    cv2.imshow("YOLO + ONNX Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

with open(db_path, "wb") as f:
    pickle.dump(face_db, f)
cap.release()
cv2.destroyAllWindows()
