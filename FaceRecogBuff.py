import cv2
import numpy as np
import os
import pickle
import time
from sklearn.metrics.pairwise import cosine_similarity
import insightface

# === Load InsightFace model (buffalo_s for Pi5 or buffalo_l for better accuracy) ===
face_model = insightface.app.FaceAnalysis(name="buffalo_s")  # use 'buffalo_l' for better accuracy on PC
face_model.prepare(ctx_id=0)

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
embedding_threshold = 0.6
samples_required = 25
face_timeout = 3.0  # seconds

# === Recognition logic ===
def recognize_face(embedding):
    for name, known_emb in face_db.items():
        sim = cosine_similarity([embedding], [known_emb])[0][0]
        if sim > embedding_threshold:
            return name
    return None

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
    faces = face_model.get(frame)
    now = time.time()

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        embedding = face.normed_embedding

        name = recognize_face(embedding)
        if name:
            print(f"✅ Face recognized as {name}")
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Cleanup old pending faces
    for pid in list(pending_faces):
        if now - pending_faces[pid]['last_seen'] > face_timeout:
            del pending_faces[pid]

    cv2.imshow("InsightFace Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
