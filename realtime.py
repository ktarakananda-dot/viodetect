"""Standalone webcam runner for the front-camera pose Transformer model."""

from pathlib import Path
import os

import cv2
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / "Website" / ".ultralytics"))

from Website.Website.person_tracker import PersonTracker


MODEL_PATH = BASE_DIR / "Website" / "Website" / "models" / "pose_transformer.keras"
YOLO_PATH = BASE_DIR / "yolov8n.pt"
DEEPSORT_PATH = BASE_DIR / "deep_sort" / "deep" / "checkpoint" / "ckpt.t7"


def main(camera_index=0):
    model = load_model(MODEL_PATH)
    print("Model loaded. Input shape:", model.input_shape)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera_index}. Connect a webcam or change the camera index."
        )

    tracker = PersonTracker(
        yolo_weights=str(YOLO_PATH),
        deepsort_ckpt=str(DEEPSORT_PATH),
        buffer_size=20,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("The camera stopped returning frames.")
                break

            output_frame, _label = tracker.update(frame, violence_model=model)
            cv2.imshow("Multi-Person Violence Detection", output_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
