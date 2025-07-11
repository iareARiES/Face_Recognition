import cv2
import os
import time
import numpy as np
import onnxruntime as ort
import psycopg2
from dotenv import load_dotenv
from picamera2 import Picamera2
from MobileFaceNet_FaceRecognition_SQL import preprocess_yolo, postprocess, get_embedding

# Load DB env
load_dotenv("face.env")

# Connect to local PostgreSQL
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
     password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

def save_face_to_db(name, embedding):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO face_embeddings (name, embedding) VALUES (%s, %s)", (name, embedding.tolist()))
        conn.commit()
        print(f"[✅ DB] Face for '{name}' saved to local PostgreSQL.")

# Load YOLO
YOLO_MODEL_PATH = "best.onnx"
yolo_session = ort.InferenceSession(YOLO_MODEL_PATH, providers=["CPUExecutionProvider"])
yolo_input_name = yolo_session.get_inputs()[0].name

FACE_VIEWS = ["Frontal", "Right Profile", "Left Profile"]
SAMPLES_PER_VIEW = 40

def register_face():
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (416, 416)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    print("[INFO] Please face the camera. We'll capture your face from multiple angles.")
    print("[INFO] Registration will begin in 3 seconds...")
    time.sleep(3)

    collected_embeddings = []

    for view in FACE_VIEWS:
        print(f"[INFO] Now capturing: {view} view. Hold still and follow instructions.")
        print(f"[INFO] Collecting {SAMPLES_PER_VIEW} samples...")
        count = 0
        view_embeddings = []

        while count < SAMPLES_PER_VIEW:
            frame = picam2.capture_array()
            frame = cv2.flip(frame, 1)
            input_tensor = preprocess_yolo(frame)
            outputs = yolo_session.run(None, {yolo_input_name: input_tensor})
            boxes = postprocess(outputs, frame)

            for (x1, y1, x2, y2) in boxes:
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                emb = get_embedding(face_img)
                view_embeddings.append(emb)
                count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                break

            cv2.putText(frame, f"View: {view} | Captured: {count}/{SAMPLES_PER_VIEW}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("Registering Face - Follow View Instructions", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                picam2.stop()
                print("[INFO] Registration aborted by user.")
                return

        collected_embeddings.extend(view_embeddings)

    cv2.destroyAllWindows()
    picam2.stop()

    if collected_embeddings:
        avg_embedding = np.mean(collected_embeddings, axis=0)
        name = input("Enter your name for registration: ").strip()
        if name:
            save_face_to_db(name, avg_embedding)
        else:
            print("[ERROR] Invalid name. Registration aborted.")
    else:
        print("[ERROR] No face data was collected.")

if __name__ == "__main__":
    register_face()
