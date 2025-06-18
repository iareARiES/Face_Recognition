from ultralytics import YOLO
import dlib
import cv2
import numpy as np
import pickle
import os

# Load models
print("[INFO] Loading YOLOv8 face detector...")
model = YOLO("best.pt")

print("[INFO] Loading Dlib face detector and encoder...")
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load known embeddings
print("[INFO] Loading saved embeddings...")
if os.path.exists("embeddings.pkl"):
    with open("embeddings.pkl", "rb") as f:
        known_faces = pickle.load(f)
    print(f"[INFO] Loaded {len(known_faces)} known face(s).")
else:
    known_faces = {}  # label -> embedding
    print("[INFO] No embeddings found. Starting fresh.")

# Next person ID
next_id = len(known_faces)

# Function: Extract embedding from a face image
def get_face_embedding(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    dets = face_detector(gray, 1)
    if len(dets) == 0:
        return None
    shape = shape_predictor(gray, dets[0])
    embedding = face_encoder.compute_face_descriptor(face_image, shape)
    return np.array(embedding)

# Function: Match face embedding with existing ones
def recognize_face(new_embedding):
    threshold = 0.6
    for label, embedding in known_faces.items():
        dist = np.linalg.norm(new_embedding - embedding)
        if dist < threshold:
            print(f"[INFO] Existing person recognized: {label}")
            return label
    print("[INFO] New person detected.")
    return None

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam could not be opened.")
    exit()

print("[INFO] Starting live face recognition...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame read failed.")
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    results = model.predict(source=frame, imgsz=640, conf=0.4, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            print("[WARNING] Empty face region skipped.")
            continue

        embedding = get_face_embedding(face)
        if embedding is None:
            print("[WARNING] Dlib could not detect landmarks.")
            continue

        label = recognize_face(embedding)

        if label is None:
            label = f"Person {next_id}"
            known_faces[label] = embedding
            next_id += 1
            with open("embeddings.pkl", "wb") as f:
                pickle.dump(known_faces, f)
            print(f"[INFO] New embedding stored as {label}.")

        # Draw face box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
