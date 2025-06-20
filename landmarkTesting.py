import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
onnx_model_path = "best.onnx"
yolo_session = ort.InferenceSession(onnx_model_path)
yolo_input_name = yolo_session.get_inputs()[0].name

# Load 106-point landmark ONNX model
landmark_session = ort.InferenceSession("2d106det.onnx", providers=["CPUExecutionProvider"])
landmark_input_name = landmark_session.get_inputs()[0].name


def get_landmarks(face_crop):
    input_img = cv2.resize(face_crop, (192, 192))
    h_orig, w_orig = face_crop.shape[:2]

    img = input_img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None, ...]  # shape: (1,3,192,192)

    preds = landmark_session.run(None, {landmark_input_name: img})[0]
    landmarks = preds.reshape(106, 2)

    # These are in range [0,1] because model is trained on 192x192 normalized
    landmarks[:, 0] *= w_orig
    landmarks[:, 1] *= h_orig
    return landmarks


    
def align_face_with_affine(face_crop, landmarks):
    # Select 3 key landmarks
    left_eye = np.mean(landmarks[60:66], axis=0)
    right_eye = np.mean(landmarks[68:74], axis=0)
    nose_tip = landmarks[96]
    src = np.array([left_eye, right_eye, nose_tip], dtype=np.float32)

    # Canonical target points (for aligned 112x112 face)
    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366]
    ], dtype=np.float32)

    # Get affine transform and apply
    M = cv2.getAffineTransform(src, dst)
    aligned = cv2.warpAffine(face_crop, M, (112, 112), flags=cv2.INTER_LINEAR)

    return aligned


# Preprocessing
def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  #Transposes shape from (H, W, C) → (C, H, W). as YOLOV8 expects in that way (C, H, W) → Channels x Height x Width (RGB, float32)
    img = np.expand_dims(img, axis=0).copy()   # [Channels, Height, Width] ----> # [Batch, Channels, Height, Width]
    return img



def postprocess(outputs, orig_frame, conf_thres=0.2):
    predictions = outputs[0][0]  # (300, 6)
    orig_h, orig_w = orig_frame.shape[:2]

    scale_x = orig_w / 640
    scale_y = orig_h / 640

    face_crops = []

    for pred in predictions:
        obj_conf = pred[4]
        if obj_conf < conf_thres:
            continue

        x1 = int(pred[0] * scale_x)
        y1 = int(pred[1] * scale_y)
        x2 = int(pred[2] * scale_x)
        y2 = int(pred[3] * scale_y)

        # Bounds check
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        # Crop face
        face_crop = orig_frame[y1:y2, x1:x2].copy()
        if face_crop.size > 0:
            face_crops.append(face_crop)

        # Draw box
        score_text = f"Face: {obj_conf * 100:.1f}%"
        cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(orig_frame, score_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return orig_frame, face_crops


# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

last_num_faces = 0  # Track number of face windows in the last frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror flip
    input_tensor = preprocess(frame)  # Preprocess for YOLO
    outputs = yolo_session.run(None, {yolo_input_name: input_tensor})
    annotated, face_crops = postprocess(outputs, frame)

    # Handle extra closed windows
    if last_num_faces > len(face_crops):
        for i in range(len(face_crops), last_num_faces):
            print("last_num_faces after: ", last_num_faces)
            cv2.destroyWindow(f"Face {i+1}")
            cv2.destroyWindow(f"Aligned Face {i+1}")
            cv2.destroyWindow(f"Landmark Overlay {i+1}")

    print("last_num_faces after again: ", last_num_faces)
    last_num_faces = len(face_crops)

    for i, face_crop in enumerate(face_crops):
        landmarks = get_landmarks(face_crop)

        # 🔍 Landmark debugging
        debug_crop = face_crop.copy()
        for (x, y) in landmarks.astype(np.int32):
            cv2.circle(debug_crop, (x, y), 1, (0, 255, 0), -1)
        cv2.imshow(f"Landmark Overlay {i+1}", debug_crop)

        # ✨ Alignment
        aligned_face = align_face_with_affine(face_crop, landmarks)
        cv2.imshow(f"Aligned Face {i+1}", aligned_face)

    cv2.imshow("YOLOv8 ONNX Face Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()
