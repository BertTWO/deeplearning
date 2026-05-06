"""
camera.py - Real-time object classification using webcam or a single image.

Place this file in the same folder as train.py.
Run train.py first to generate model.keras and class_names.txt.

Usage:
    python camera.py                        # webcam mode
    python camera.py --image photo.jpg      # image mode (drag and drop path)

Controls (webcam mode):
    Q - quit
"""

import os
import sys
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf
import time


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

MODEL_PATH     = "./model.keras"
CLASS_PATH     = "./class_names.txt"
IMG_SIZE       = (224, 224)
CONFIDENCE_MIN = 0.60
SMOOTH_FRAMES  = 5

CLASS_COLORS = [
    (0,   200, 255),
    (80,  255,  80),
    (255, 100,  80),
    (255,  80, 200),
]


# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH) as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera.\n"
            "If you do not have a camera, use: python camera.py --image your_photo.jpg"
        )
    return cap


# ------------------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------------------

def preprocess_frame(frame):
    """Center-crop to square, resize, and normalize."""
    h, w = frame.shape[:2]
    size = min(h, w)
    y1   = (h - size) // 2
    x1   = (w - size) // 2
    crop = frame[y1:y1 + size, x1:x1 + size]
    img  = cv2.resize(crop, IMG_SIZE)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img  = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict(model, frame):
    preds      = model.predict(preprocess_frame(frame), verbose=0)[0]
    class_idx  = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    return class_idx, confidence


def smooth_predictions(buffer, new_pred, max_len):
    buffer.append(new_pred)
    if len(buffer) > max_len:
        buffer.pop(0)
    return np.mean(buffer, axis=0)


# ------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------

def draw_overlay(frame, label, confidence, color, fps=None):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX

    # Crop region box
    size = min(h, w)
    y1   = (h - size) // 2
    x1   = (w - size) // 2
    cv2.rectangle(frame, (x1, y1), (x1 + size, y1 + size), color, 2)

    # Prediction label
    display_label = label.upper() if confidence >= CONFIDENCE_MIN else "UNCERTAIN"
    label_color   = color if confidence >= CONFIDENCE_MIN else (140, 140, 140)
    (tw, th), _   = cv2.getTextSize(display_label, font, 1.2, 2)
    cv2.rectangle(frame, (10, 10), (tw + 30, th + 30), label_color, -1)
    cv2.putText(frame, display_label, (20, th + 18), font, 1.2, (0, 0, 0), 2)

    # Confidence bar
    bar_x, bar_y = 20, h - 55
    bar_w, bar_h = 280, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * confidence), bar_y + bar_h), color, -1)
    cv2.putText(frame, f"{confidence * 100:.0f}%",
                (bar_x + bar_w + 10, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # FPS (webcam mode only)
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.0f}",
                    (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    # Hint
    hint = "Q - quit" if fps is not None else "Any key - close"
    cv2.putText(frame, hint,
                (w - 130, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return frame


# ------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------

def run_camera(model, class_names):
    cap         = open_camera()
    pred_buffer = []
    fps_counter = 0
    fps_timer   = time.time()
    fps         = 0.0

    print("Camera running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        preds      = model.predict(preprocess_frame(frame), verbose=0)[0]
        avg_preds  = smooth_predictions(pred_buffer, preds, SMOOTH_FRAMES)
        class_idx  = int(np.argmax(avg_preds))
        confidence = float(avg_preds[class_idx])
        label      = class_names[class_idx]
        color      = CLASS_COLORS[class_idx % len(CLASS_COLORS)]

        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps         = fps_counter
            fps_counter = 0
            fps_timer   = time.time()

        frame = draw_overlay(frame, label, confidence, color, fps)
        cv2.imshow("Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


def run_image(model, class_names, image_path):
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: could not read image: {image_path}")
        sys.exit(1)

    class_idx, confidence = predict(model, frame)
    label = class_names[class_idx]
    color = CLASS_COLORS[class_idx % len(CLASS_COLORS)]

    print(f"\nResult     : {label}")
    print(f"Confidence : {confidence * 100:.1f}%")

    frame = draw_overlay(frame, label, confidence, color, fps=None)
    cv2.imshow(f"Classifier - {os.path.basename(image_path)}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Object classifier - webcam or image")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to an image file. Omit this flag to use the webcam.",
    )
    args = parser.parse_args()

    print("Loading model...")
    model, class_names = load_resources()
    print(f"Classes: {class_names}")

    if args.image:
        print(f"Image mode: {args.image}")
        run_image(model, class_names, args.image)
    else:
        print("Webcam mode")
        run_camera(model, class_names)


if __name__ == "__main__":
    main()
