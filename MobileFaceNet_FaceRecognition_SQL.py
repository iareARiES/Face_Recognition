#this files saves the data in the locally created PostgraceSQL 

import cv2
import numpy as np
import onnxruntime as ort
import psycopg2
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from picamera2 import Picamera2
import os

# Load local DB credentials
load_dotenv("face.env")

# PostgreSQL connection
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

# YOLOv8 model (face detector)
yolo_session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
yolo_input_name = yolo_session.get_inputs()[0].name

# MobileFaceNet model (face recognizer)
rec_session = ort.InferenceSession("w600k_mbf.onnx", providers=["CPUExecutionProvider"])
rec_input_name = rec_session.get_inputs()[0].name

# --- Preprocessing ---
def preprocess_yolo(frame):
    img = cv2.resize(frame, (416, 416))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0).copy()

def postprocess(outputs, orig_frame, conf_thres=0.2):
    predictions = outputs[0][0]
    h, w = orig_frame.shape[:2]
    scale_x, scale_y = w / 416, h / 416
    boxes = []

    for pred in predictions:
        if pred[4] < conf_thres:
            continue
        x1, y1 = int(pred[0] * scale_x), int(pred[1] * scale_y)
        x2, y2 = int(pred[2] * scale_x), int(pred[3] * scale_y)
        boxes.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))

    return boxes

def preprocess_face(face_img):
    face = cv2.resize(face_img, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    face = np.transpose(face, (2, 0, 1))
    return np.expand_dims(face, axis=0)

def get_embedding(face_img):
    input_tensor = preprocess_face(face_img)
    output = rec_session.run(None, {rec_input_name: input_tensor})
    embedding = output[0][0]
    return embedding / np.linalg.norm(embedding)

def load_face_db():
    with conn.cursor() as cur:
        cur.execute("SELECT name, embedding FROM face_embeddings")
        rows = cur.fetchall()
        return {name: np.array(embedding, dtype=np.float32) for name, embedding in rows}

def recognize(embedding, face_db, threshold=0.5):
    for name, known_emb in face_db.items():
        sim = cosine_similarity([embedding], [known_emb])[0][0]
        if sim > threshold:
            return name
    return None

def detect_and_recognize(frame, face_db):
    input_tensor = preprocess_yolo(frame)
    outputs = yolo_session.run(None, {yolo_input_name: input_tensor})
    boxes = postprocess(outputs, frame)
    results = []

    for (x1, y1, x2, y2) in boxes:
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        emb = get_embedding(face_crop)
        name = recognize(emb, face_db)
        results.append(((x1, y1, x2, y2), name or "Unknown"))

    return results

# Main camera loop
if __name__ == "__main__":
    cap = Picamera2()
    cap.preview_configuration.main.size = (640, 480)
    cap.preview_configuration.main.format = "RGB888"
    cap.configure("preview")
    cap.start()

    face_db = load_face_db()
    print("[INFO] Face recognition started. Press 'q' to exit.")

    while True:
        frame = cap.capture_array()
        frame = cv2.flip(frame, 1)
        results = detect_and_recognize(frame, face_db)

        for (x1, y1, x2, y2), name in results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.stop()
    cv2.destroyAllWindows()
