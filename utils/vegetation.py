"""
utils/vegetation.py
~~~~~~~~~~~~~~~~~~~
Vegetation presence, tree trunk structure, and branch morphology detection.
Provides image pre-validation before calculating spectral vegetation indices.
"""

import cv2
import numpy as np
from utils.detection import detect_invalid_objects


def check_vegetation_presence(image_cv2):
    """
    Examines HSV channels, edge density, and vertical morphological kernels
    to confirm whether the image contains genuine trees or plant foliage.

    Detects:
      - Green leafy canopy
      - Autumn / dry brown leaves
      - Bare winter branches & vertical woody trunks

    Returns:
      (score: float, is_valid_vegetation: bool, details: dict)
    """
    small = cv2.resize(image_cv2, (500, 500))
    total_pixels = small.shape[0] * small.shape[1]

    # Convert to HSV
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # 1. Green foliage detection
    lower_green = np.array([25, 30, 25])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pct = (np.sum(green_mask > 0) / total_pixels) * 100.0

    # 2. Brown/woody structure detection (trunks, branches, dry leaves)
    lower_brown = np.array([0, 15, 20])
    upper_brown = np.array([35, 255, 240])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    kernel_3x3 = np.ones((3, 3), np.uint8)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel_3x3)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel_3x3)
    brown_pct = (np.sum(brown_mask > 0) / total_pixels) * 100.0

    # 3. Branch / edge complexity detection
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 120)
    edge_pct = (np.sum(edges > 0) / total_pixels) * 100.0

    # 4. Vertical trunk morphology (detects vertical tree stems)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    vertical_structure = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, vertical_kernel)
    vertical_pct = (np.sum(vertical_structure > 0) / total_pixels) * 100.0

    details = {
        "green_percentage": float(green_pct),
        "brown_percentage": float(brown_pct),
        "edge_percentage": float(edge_pct),
        "vertical_percentage": float(vertical_pct),
    }

    # Decision tree:
    # A. Sufficient green vegetation
    if green_pct >= 4.0:
        return green_pct, True, details

    # B. Leafless / dry tree (branches + brown wood)
    if brown_pct >= 3.0 and edge_pct >= 5.0:
        score = (brown_pct + edge_pct) / 2.0
        return score, True, details

    # C. Strong woody tree structure
    if brown_pct >= 6.0:
        return brown_pct, True, details

    # D. Strong vertical trunk structure
    if vertical_pct >= 1.0 and brown_pct >= 2.0:
        return vertical_pct, True, details

    return 0.0, False, details


def validate_image(image_array, yolo_model=None, yolo_available=False):
    """
    Complete gatekeeper validation for uploaded photos.
    Runs dimensional checks, YOLO non-target object filtering, and vegetation detection.

    Returns:
        (is_valid: bool, error_message: str, details: dict)
    """
    if image_array is None:
        return False, "Unable to read the image. Please upload a valid image file.", {}

    try:
        # Convert RGB to BGR for OpenCV processing
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_cv2 = image_array

        height, width = image_cv2.shape[:2]
        if height < 100 or width < 100:
            return False, "Image resolution is too low (< 100x100 px). Please upload a clearer photograph.", {}

        # YOLO object detection gating
        if yolo_available and yolo_model is not None:
            detected_objects, yolo_working = detect_invalid_objects(image_cv2, yolo_model)
            if yolo_working and detected_objects:
                object_names = sorted(set(name for name, _ in detected_objects))
                names_str = ", ".join(object_names)
                return False, f"Non-vegetation subject detected ({names_str}). Please upload an image focused on trees or foliage.", {}

        # Morphological & colour vegetation detection
        score, is_vegetation, details = check_vegetation_presence(image_cv2)
        if not is_vegetation:
            return False, (
                "No clear tree or vegetation structure detected. "
                "Ensure the photo shows foliage, branches, or tree trunks with adequate lighting."
            ), details

        return True, "", details

    except Exception as exc:
        return False, f"Validation error encountered: {exc}", {}
