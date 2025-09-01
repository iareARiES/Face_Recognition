# recognition_client.py
import cv2, json, threading, time
import numpy as np
import onnxruntime as ort
import requests
from sklearn.metrics.pairwise import cosine_similarity
from websocket import WebSocketApp  # pip install websocket-client
from typing import Dict

HOST_IP = "192.168.252.10"   # <-- change to your host laptop's IP on the hotspot
BASE = f"http://{HOST_IP}:8000"
WS_URL = f"ws://{HOST_IP}:8000/updates"

# ---- Shared DB (thread-safe) ----
face_db_lock = threading.Lock()
face_db: Dict[str, np.ndarray] = {}  # {name: np.ndarray}

def set_full_db(db_json: dict):
    with face_db_lock:
        face_db.clear()
        for name, emb_list in db_json.items():
            face_db[name] = np.array(emb_list, dtype=np.float32)

def add_or_update_record(name: str, emb_list):
    with face_db_lock:
        face_db[name] = np.array(emb_list, dtype=np.float32)

# ---- Bootstrap once via HTTP ----
def bootstrap_db():
    try:
        r = requests.get(f"{BASE}/db", timeout=2.0)
        r.raise_for_status()
        set_full_db(r.json())
        print(f"[DB] Bootstrapped {len(face_db)} identities from host")
    except Exception as e:
        print(f"[DB] Bootstrap failed: {e}")

# ---- Live updates via WebSocket ----
def on_ws_message(ws, message):
    try:
        msg = json.loads(message)
        if msg.get("type") == "full":
            set_full_db(msg.get("db", {}))
            print(f"[WS] Full DB received ({len(face_db)} identities)")
        elif msg.get("type") == "add":
            rec = msg.get("record", {})
            name = rec.get("name")
            emb = rec.get("embedding")
            if name and isinstance(emb, list):
                add_or_update_record(name, emb)
                print(f"[WS] Updated identity: {name}")
    except Exception as e:
        print(f"[WS] Parse error: {e}")

def on_ws_close(ws, status, msg):
    print("[WS] Closed, retrying in 2s...")
    time.sleep(2)
    start_ws_thread()

def on_ws_error(ws, err):
    print(f"[WS] Error: {err}")

def ws_runner():
    ws = WebSocketApp(WS_URL, on_message=on_ws_message, on_close=on_ws_close, on_error=on_ws_error)
    ws.run_forever()

def start_ws_thread():
    t = threading.Thread(target=ws_runner, daemon=True)
    t.start()

# ---- Your unchanged CV pipeline ----
# YOLOv8 (face detector)
yolo_session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
yolo_input_name = yolo_session.get_inputs()[0].name

# MobileFaceNet (face recognizer)
rec_session = ort.InferenceSession("w600k_mbf.onnx", providers=["CPUExecutionProvider"])
rec_input_name = rec_session.get_inputs()[0].name

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

def recognize(embedding, threshold=0.5):
    # Snapshot the DB safely
    with face_db_lock:
        items = list(face_db.items())
    best_name, best_sim = None, -1.0
    for name, known_emb in items:
        sim = cosine_similarity([embedding], [known_emb])[0][0]
        if sim > best_sim:
            best_sim, best_name = sim, name
    return best_name if best_sim >= threshold else None

def detect_and_recognize(frame):
    input_tensor = preprocess_yolo(frame)
    outputs = yolo_session.run(None, {yolo_input_name: input_tensor})
    boxes = postprocess(outputs, frame)
    results = []
    for (x1, y1, x2, y2) in boxes:
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        emb = get_embedding(face_crop)
        name = recognize(emb)
        results.append(((x1, y1, x2, y2), name or "Unknown"))
    return results

if __name__ == "__main__":
    bootstrap_db()
    start_ws_thread()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Recognition client started. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame from camera.")
            continue
        frame = cv2.flip(frame, 1)
        results = detect_and_recognize(frame)

        for (x1, y1, x2, y2), name in results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition (Receiver)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
