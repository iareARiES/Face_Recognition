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
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_db = load_face_db()
    print("[INFO] Face recognition started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame from camera.")
            continue
        frame = cv2.flip(frame, 1)
        results,counts = detect_and_recognize(frame, face_db)
        #results = sorted(results, key= lambda r: (r[0][0],r[0][1]))

        for (x1, y1, x2, y2), name in results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if(counts>=2):
            pairs = calculateDist(results)
            for p in pairs:
                x1,y1=int(p["ptAx"]),int(p["ptAy"])
                x2,y2=int(p["ptBx"]),int(p["ptBy"])

            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2) #drawing the line between the coordinates

            mx,my=(x1+x2)//2, (y1+y2)//2
            cv2.putText(frame,f"{p['dist']:.1f}px",(mx,my),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)

        print(f"(The number of faces:{counts})")
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
