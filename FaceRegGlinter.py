import cv2
import time
import numpy as np
import pickle
import os
from FaceRecoGlinter import preprocess_face, get_embedding, preprocess_yolo, postprocess
import onnxruntime as ort
from picamera2 import Picamera2

DB_PATH = "face_db.pkl"
YOLO_MODEL_PATH = "best.onnx"
yolo_session = ort.InferenceSession(YOLO_MODEL_PATH, providers=["CPUExecutionProvider"])
yolo_input_name = yolo_session.get_inputs()[0].name

FACE_VIEWS = ["Frontal", "Right Profile", "Left Profile"]
SAMPLES_PER_VIEW = 40  

def load_face_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_face_db(face_db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(face_db, f)

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
            face_db = load_face_db()
            face_db[name] = avg_embedding
            save_face_db(face_db)
            print(f"[INFO] Face successfully registered as '{name}'")
        else:
            print("[ERROR] Invalid name. Registration aborted.")
    else:
        print("[ERROR] No face data was collected.")

if __name__ == "__main__":
    register_face()
