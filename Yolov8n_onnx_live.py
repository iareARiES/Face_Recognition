import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
onnx_model_path = "best.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name

# Preprocessing
def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).copy()
    return img

# Postprocessing for SINGLE class model (Face)
def postprocess(outputs, orig_frame, conf_thres=0.3):
    predictions = outputs[0][0]  # (300, 6)
    orig_h, orig_w = orig_frame.shape[:2]

    scale_x = orig_w / 640
    scale_y = orig_h / 640

    for pred in predictions:
        obj_conf = pred[4]
        if obj_conf < conf_thres:
            continue

        x1 = int(pred[0] * scale_x)
        y1 = int(pred[1] * scale_y)
        x2 = int(pred[2] * scale_x)
        y2 = int(pred[3] * scale_y)

        cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig_frame, f"Face", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return orig_frame




# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    input_tensor = preprocess(frame)
    outputs = session.run(None, {input_name: input_tensor})
    print(f"ONNX Output shape: {outputs[0].shape}")

    annotated = postprocess(outputs, frame)

    cv2.imshow("YOLOv8 ONNX Face Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
