# MobileFaceNet_FaceRegister_JSON.py
import cv2
import time
import numpy as np
import onnxruntime as ort

from FaceRecognition_Video_Inferencing import preprocess_yolo, postprocess, get_embedding
from face_store_json import save_face_json

# Load YOLO (local to this script for detection)
YOLO_MODEL_PATH = "best.onnx"
yolo_session = ort.InferenceSession(YOLO_MODEL_PATH, providers=["CPUExecutionProvider"])
yolo_input_name = yolo_session.get_inputs()[0].name

FACE_VIEWS = ["Frontal", "Right Profile", "Left Profile"]
SAMPLES_PER_VIEW = 100

def save_face_to_json(name, embedding):
    # normalize before saving (consistent with recognition)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    save_face_json(name, embedding)
    print(f"[✅ JSON] Face for '{name}' saved to face_embeddings.jsonl")

def register_face():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)

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
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame from camera.")
                continue
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

            cv2.putText(frame, f"View: {view} | Captured: {count}/{SAMPLES_PER_VIEW}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("Registering Face - Follow View Instructions", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                cap.release()
                print("[INFO] Registration aborted by user.")
                return

        collected_embeddings.extend(view_embeddings)

    cv2.destroyAllWindows()
    cap.release()

    if collected_embeddings:
        avg_embedding = np.mean(collected_embeddings, axis=0)
        name = input("Enter your name for registration: ").strip()
        if name:
            save_face_to_json(name, avg_embedding)
        else:
            print("[ERROR] Invalid name. Registration aborted.")
    else:
        print("[ERROR] No face data was collected.")

if __name__ == "__main__":
    register_face()
