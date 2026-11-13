"""
utils/detection.py
~~~~~~~~~~~~~~~~~~
Object detection and image gating utilities using YOLO.
Prevents processing non-vegetation images (e.g. cars, people, urban infrastructure).
"""

import os

# Classes that indicate the photo is NOT primarily of a tree or vegetation
INVALID_OBJECTS = {
    "person",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
    "train",
    "boat",
    "airplane",
    "traffic light",
    "stop sign",
}


def load_yolo_model(model_path="yolo11n.pt"):
    """
    Safely load the Ultralytics YOLO model.
    Returns (model, error_string_or_None).
    """
    try:
        from ultralytics import YOLO

        if not os.path.exists(model_path):
            return None, f"Model file '{model_path}' not found on disk."

        model = YOLO(model_path)
        return model, None
    except Exception as exc:
        return None, str(exc)


def detect_invalid_objects(image_cv2, yolo_model, conf_threshold=0.15, iou_threshold=0.45):
    """
    Run YOLO inference on the input image to check for prohibited objects.

    Returns:
        (detected_objects, is_successful)
        where detected_objects is a list of (class_name, confidence).
    """
    if yolo_model is None:
        return None, False

    try:
        results = yolo_model.predict(
            source=image_cv2,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            verbose=False,
        )

        detected_objects = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = yolo_model.names.get(class_id, "")

                if class_name in INVALID_OBJECTS:
                    detected_objects.append((class_name, confidence))

        return detected_objects, True
    except Exception as exc:
        print(f"YOLO detection exception: {exc}")
        return None, False
