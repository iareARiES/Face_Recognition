# MobileFaceNet_FaceRecognition_JSON.py
import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
import math
from face_store_json import load_face_db_json

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
    """Reads face_embeddings.jsonl and returns {name: embedding}."""
    return load_face_db_json()

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
    return results, len(results)

def clamp(centre,low,high):
    minPoint = max(low,min(centre,high))
    return minPoint


def minDistAndPoints(a,b): #focus on calulating the distance between 2 boxes input = box[i] and next box[i+1]
    """
    True shortest distance between two axis-aligned boxes (OpenCV coords)
    and the two points (one on each box) that realize that distance.
    Boxes: (x1,y1,x2,y2) with y downwards.
    """
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b

    centreAx,centreAy = ((ax1+ax2)/2.0), ((ay1+ay2)/2.0)
    centreBx,centreBy = ((bx1+bx2)/2.0), ((by1+by2)/2.0)

    ptABx = clamp(centreAx,bx1,bx2)  #A is fixed -> take centre of A
    ptABy = clamp(centreAy,by1,by2)

    ptBAx = clamp(centreBx,ax1,ax2) #B is fixed -> take centre of B
    ptBAy = clamp(centreBy,ay1,ay2)

    dx = ptABx - ptBAx
    dy = ptABy - ptBAy
    distance = math.hypot(dx,dy)

    return distance,ptABx,ptABy,ptBAx,ptBAy

def calculateDist(results): # loops over all the boxes taking 2 at a time
    boxes = [box for(box,_)in results]
    n = len(boxes)
    out = []
    for i in range(n):
        for j in range(i+1,n):
            dist, ptAx ,ptAy ,ptBx, ptBy = minDistAndPoints(boxes[i],boxes[j])
            out.append({"pair": (i,j),
                        "dist": dist, 
                        "ptAx": ptAx,"ptAy": ptAy, 
                        "ptBx": ptBx,"ptBy": ptBy})
    return out

if __name__ == "__main__":
    import os, csv, glob

    # --------- EDIT THIS PATH ----------
    IN_PATH = r"E:\CSIR\image"   # can be a single image OR a folder
    # -----------------------------------

    # valid extensions
    VALID_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

    def process_image(in_image):
        base_dir  = os.path.dirname(in_image)
        base_name, ext = os.path.splitext(os.path.basename(in_image))
        if ext.lower() not in VALID_EXTS:
            print(f"[SKIP] {in_image} (unsupported format)")
            return

        out_image = os.path.join(base_dir, f"{base_name}_annotated{ext}")
        out_csv   = os.path.join(base_dir, f"{base_name}_pairs.csv")

        # load image
        img = cv2.imread(in_image)
        if img is None:
            print(f"[ERROR] Could not read image: {in_image}")
            return

        # run inference
        results, counts = detect_and_recognize(img, face_db)
        print(f"[INFO] {in_image} → {counts} face(s)")

        # assign labels
        labels, unk_counter = [], 0
        for (_box, name) in results:
            if name == "Unknown":
                unk_counter += 1
                labels.append(f"Unknown{unk_counter}")
            else:
                labels.append(name)

        # draw boxes + labels
        for idx, ((x1, y1, x2, y2), _) in enumerate(results):
            color = (0, 255, 0) if not labels[idx].startswith("Unknown") else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, labels[idx], (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # pairwise distances
        pairs = calculateDist(results) if counts >= 2 else []

        # draw lines + distances
        for p in pairs:
            x1, y1 = int(p["ptAx"]), int(p["ptAy"])
            x2, y2 = int(p["ptBx"]), int(p["ptBy"])
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(img, f"{p['dist']:.1f}px", (mx, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # save CSV
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["FaceA", "FaceB", "Distance(px)"])
            for p in pairs:
                i, j = p["pair"]
                w.writerow([labels[i], labels[j], f"{p['dist']:.2f}"])
        print(f"   [OK] CSV saved → {out_csv}")

        # save annotated image
        if not cv2.imwrite(out_image, img):
            print(f"[ERROR] Failed to save image: {out_image}")
        else:
            print(f"   [OK] Image saved → {out_image}")

    # --- main driver ---
    face_db = load_face_db()

    if os.path.isfile(IN_PATH):
        # single image
        process_image(IN_PATH)
    elif os.path.isdir(IN_PATH):
        # all images in folder
        for ext in VALID_EXTS:
            for f in glob.glob(os.path.join(IN_PATH, f"*{ext}")):
                process_image(f)
    else:
        print(f"[ERROR] Path not found: {IN_PATH}")
