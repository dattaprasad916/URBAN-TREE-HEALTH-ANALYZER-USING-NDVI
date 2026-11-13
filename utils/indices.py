"""
utils/indices.py
~~~~~~~~~~~~~~~~
Multi-spectral and Visible-band Vegetation Index calculations:
- Simulated NDVI (Normalized Difference Vegetation Index)
- VARI (Visible Atmospherically Resistant Index, Gitelson et al.)
- GLI (Green Leaf Index, Louhaichi et al.)

Provides peer-reviewed visible-spectrum remote-sensing alternatives
for RGB cameras without true near-infrared sensors.
"""

import cv2
import numpy as np
from utils.ndvi import (
    create_vegetation_mask,
    simulate_nir_advanced,
    compute_ndvi_advanced,
    classify_vegetation_health,
)

AVAILABLE_INDICES = {
    "NDVI (Simulated)": {
        "formula": "(NIR_sim - Red) / (NIR_sim + Red)",
        "description": "Simulates Near-Infrared from RGB channels to approximate classic NDVI.",
        "scientific_basis": "Classic remote sensing proxy for chlorophyll absorption & NIR scattering.",
        "healthy_threshold": 0.10,
        "moderate_threshold": -0.10,
    },
    "VARI (Visible Atmospherically Resistant Index)": {
        "formula": "(Green - Red) / (Green + Red - Blue)",
        "description": "Peer-reviewed RGB index that reduces atmospheric and illumination variations.",
        "scientific_basis": "Gitelson et al. (2002) — directly predicts green vegetation fraction from RGB.",
        "healthy_threshold": 0.12,
        "moderate_threshold": -0.05,
    },
    "GLI (Green Leaf Index)": {
        "formula": "(2*Green - Red - Blue) / (2*Green + Red + Blue)",
        "description": "Detects green leaf biomass while effectively suppressing bare soil and dead wood.",
        "scientific_basis": "Louhaichi et al. (2001) — peer-reviewed canopy cover estimation index.",
        "healthy_threshold": 0.10,
        "moderate_threshold": -0.05,
    },
}


def compute_vari(rgb_image):
    """
    Computes Visible Atmospherically Resistant Index (VARI).
    Formula: (G - R) / (G + R - B + eps)
    """
    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
        b = rgb_image[:, :, 0].astype(np.float32) / 255.0
        g = rgb_image[:, :, 1].astype(np.float32) / 255.0
        r = rgb_image[:, :, 2].astype(np.float32) / 255.0
    else:
        raise ValueError("VARI requires a 3-channel RGB/BGR image")

    eps = 1e-6
    denom = g + r - b
    denom = np.where(np.abs(denom) < eps, eps, denom)
    vari_raw = (g - r) / denom
    return np.clip(vari_raw, -1.0, 1.0)


def compute_gli(rgb_image):
    """
    Computes Green Leaf Index (GLI).
    Formula: (2*G - R - B) / (2*G + R + B + eps)
    """
    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
        b = rgb_image[:, :, 0].astype(np.float32) / 255.0
        g = rgb_image[:, :, 1].astype(np.float32) / 255.0
        r = rgb_image[:, :, 2].astype(np.float32) / 255.0
    else:
        raise ValueError("GLI requires a 3-channel RGB/BGR image")

    eps = 1e-6
    numerator = 2.0 * g - r - b
    denominator = 2.0 * g + r + b + eps
    gli_raw = numerator / denominator
    return np.clip(gli_raw, -1.0, 1.0)


def compute_vegetation_index(image_cv2, index_type="NDVI (Simulated)"):
    """
    Unified entry point for computing vegetation indices.

    Returns:
        (index_raw, index_norm, veg_mask, stats, classification)
    """
    veg_mask = create_vegetation_mask(image_cv2)
    meta = AVAILABLE_INDICES.get(index_type, AVAILABLE_INDICES["NDVI (Simulated)"])
    healthy_th = meta["healthy_threshold"]
    moderate_th = meta["moderate_threshold"]

    if "VARI" in index_type:
        raw_index = compute_vari(image_cv2)
        raw_index_masked = np.where(veg_mask, raw_index, -1.0)
    elif "GLI" in index_type:
        raw_index = compute_gli(image_cv2)
        raw_index_masked = np.where(veg_mask, raw_index, -1.0)
    else:  # Default to NDVI
        raw_index_masked, _, _ = compute_ndvi_advanced(image_cv2, return_raw=True)

    # Normalize [-1, 1] to [0, 1] for visualization
    norm_index = (raw_index_masked + 1.0) / 2.0

    veg_pixels = raw_index_masked[veg_mask]
    total_pixels = int(veg_mask.size)
    num_veg_pixels = int(np.sum(veg_mask))
    veg_pct = (num_veg_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    if len(veg_pixels) > 0:
        mean_val = float(np.mean(veg_pixels))
        std_val = float(np.std(veg_pixels))
        min_val = float(np.min(veg_pixels))
        max_val = float(np.max(veg_pixels))
    else:
        mean_val, std_val, min_val, max_val = -1.0, 0.0, -1.0, -1.0

    stats = {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "vegetation_pixels": num_veg_pixels,
        "total_pixels": total_pixels,
        "vegetation_percentage": veg_pct,
        "index_name": index_type,
    }

    # Classification logic
    if veg_pct < 5.0:
        classification = {
            "status": "No Vegetation",
            "icon": "🏜️",
            "color": "gray",
            "confidence": "High",
            "description": f"Insufficient vegetation identified ({veg_pct:.1f}% canopy cover).",
            "mean_index": mean_val,
            "vegetation_percentage": veg_pct,
        }
    elif mean_val > healthy_th:
        classification = {
            "status": "Healthy",
            "icon": "🌿",
            "color": "green",
            "confidence": "High" if std_val < 0.15 else "Medium",
            "description": f"Vigorous photosynthetic activity and dense canopy ({index_type} > {healthy_th:.2f}).",
            "mean_index": mean_val,
            "vegetation_percentage": veg_pct,
        }
    elif mean_val > moderate_th:
        classification = {
            "status": "Moderate",
            "icon": "🌾",
            "color": "orange",
            "confidence": "Medium",
            "description": f"Moderate foliage density or slight abiotic stress ({moderate_th:.2f} < {index_type} <= {healthy_th:.2f}).",
            "mean_index": mean_val,
            "vegetation_percentage": veg_pct,
        }
    else:
        classification = {
            "status": "Stressed",
            "icon": "🍂",
            "color": "red",
            "confidence": "High" if std_val < 0.20 else "Medium",
            "description": f"Signs of defoliation, senescence, drought, or disease ({index_type} <= {moderate_th:.2f}).",
            "mean_index": mean_val,
            "vegetation_percentage": veg_pct,
        }

    return raw_index_masked, norm_index, veg_mask, stats, classification
