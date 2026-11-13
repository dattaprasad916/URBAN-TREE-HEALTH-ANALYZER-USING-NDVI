"""
tests/test_new_features.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for detection, vegetation morphology, visible vegetation indices (VARI, GLI),
visualization overlays, and reporting exports.
"""

import os
import sys
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.indices import compute_vari, compute_gli, compute_vegetation_index
from utils.vegetation import check_vegetation_presence, validate_image
from utils.visualization import (
    create_heatmap,
    create_heatmap_overlay,
    calculate_health_percentages,
)
from utils.export import generate_json_report, format_report_as_text


@pytest.fixture
def synthetic_green_bgr():
    """120x120 synthetic green leaf image (OpenCV BGR)."""
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    img[:, :, 1] = 180  # Green high
    img[:, :, 2] = 50   # Red low
    img[:, :, 0] = 40   # Blue low
    return img


@pytest.fixture
def synthetic_stressed_bgr():
    """120x120 synthetic brown/yellow stressed leaf image."""
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    img[:, :, 2] = 160  # Red high
    img[:, :, 1] = 110  # Green moderate
    img[:, :, 0] = 30   # Blue low
    return img


class TestVisibleIndices:
    def test_vari_green_higher_than_stressed(self, synthetic_green_bgr, synthetic_stressed_bgr):
        vari_green = compute_vari(synthetic_green_bgr)
        vari_stressed = compute_vari(synthetic_stressed_bgr)
        assert np.mean(vari_green) > np.mean(vari_stressed)
        assert vari_green.min() >= -1.0
        assert vari_green.max() <= 1.0

    def test_gli_green_positive(self, synthetic_green_bgr):
        gli_green = compute_gli(synthetic_green_bgr)
        assert np.mean(gli_green) > 0.0
        assert gli_green.shape == (120, 120)

    def test_compute_vegetation_index_vari(self, synthetic_green_bgr):
        raw, norm, mask, stats, classification = compute_vegetation_index(
            synthetic_green_bgr, "VARI (Visible Atmospherically Resistant Index)"
        )
        assert raw.shape == (120, 120)
        assert norm.min() >= 0.0 and norm.max() <= 1.0
        assert classification["status"] == "Healthy"
        assert stats["vegetation_percentage"] > 50.0

    def test_compute_vegetation_index_gli(self, synthetic_green_bgr):
        raw, norm, mask, stats, classification = compute_vegetation_index(
            synthetic_green_bgr, "GLI (Green Leaf Index)"
        )
        assert classification["status"] == "Healthy"


class TestVegetationMorphology:
    def test_green_vegetation_detected(self, synthetic_green_bgr):
        score, is_veg, details = check_vegetation_presence(synthetic_green_bgr)
        assert is_veg is True
        assert details["green_percentage"] > 10.0

    def test_validate_image_tiny_rejected(self):
        tiny = np.zeros((50, 50, 3), dtype=np.uint8)
        is_valid, msg, _ = validate_image(tiny)
        assert is_valid is False
        assert "too low" in msg


class TestVisualizationAndOverlay:
    def test_create_heatmap(self, synthetic_green_bgr):
        norm = np.full((120, 120), 0.75, dtype=np.float32)
        heatmap = create_heatmap(norm, colormap_name="JET (Classic Rainbow)")
        assert heatmap.shape == (120, 120, 3)
        assert heatmap.dtype == np.uint8

    def test_create_heatmap_overlay(self, synthetic_green_bgr):
        heatmap = np.full((120, 120, 3), 200, dtype=np.uint8)
        blended = create_heatmap_overlay(synthetic_green_bgr, heatmap, alpha=0.5)
        assert blended.shape == (120, 120, 3)

    def test_calculate_health_percentages(self):
        norm = np.array([[0.8, 0.5], [0.2, 0.9]], dtype=np.float32)
        mask = np.ones((2, 2), dtype=bool)
        breakdown = calculate_health_percentages(norm, mask, healthy_th=0.6, moderate_th=0.4)
        assert breakdown["Healthy"] == 50.0
        assert breakdown["Moderate"] == 25.0
        assert breakdown["Stressed"] == 25.0


class TestExport:
    def test_generate_and_format_report(self):
        stats = {
            "mean": 0.25,
            "std": 0.04,
            "min": 0.10,
            "max": 0.40,
            "vegetation_pixels": 8000,
            "total_pixels": 10000,
            "vegetation_percentage": 80.0,
        }
        classification = {
            "status": "Healthy",
            "confidence": "High",
            "description": "Thriving canopy",
        }
        breakdown = {"Healthy": 85.0, "Moderate": 10.0, "Stressed": 5.0}

        report = generate_json_report(
            "sample_tree.jpg", (100, 100), "NDVI (Simulated)", stats, classification, breakdown
        )
        assert report["health_assessment"]["status"] == "Healthy"
        assert len(report["recommendations"]) > 0

        text = format_report_as_text(report)
        assert "URBAN TREE HEALTH ASSESSMENT REPORT" in text
        assert "sample_tree.jpg" in text
