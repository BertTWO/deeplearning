import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "./model.keras"
CLASS_PATH = "./class_names.txt"
IMG_SIZE   = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_PATH) as f:
    class_names = [line.strip() for line in f]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    size = min(h, w)
    y1, x1 = (h - size) // 2, (w - size) // 2
    crop = frame[y1:y1+size, x1:x1+size]

    img = cv2.resize(crop, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    preds = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    label = class_names[int(np.argmax(preds))]

    cv2.rectangle(frame, (x1, y1), (x1+size, y1+size), (0, 200, 255), 2)
    cv2.putText(frame, label, (x1+10, y1+35), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 200, 255), 2)

    cv2.imshow("Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()