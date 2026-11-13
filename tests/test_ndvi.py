"""
tests/test_ndvi.py
~~~~~~~~~~~~~~~~~~
Pytest test suite for utils/ndvi.py
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

# Allow importing utils from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.ndvi import (
    create_vegetation_mask,
    simulate_nir_advanced,
    compute_ndvi_advanced,
    classify_vegetation_health,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def green_image_bgr():
    """Synthetic 100x100 image that is clearly healthy green vegetation."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 1] = 180  # Green channel high (BGR: index 1 = G)
    img[:, :, 2] = 60   # Red channel low
    img[:, :, 0] = 40   # Blue channel low
    return img


@pytest.fixture
def brown_image_bgr():
    """Synthetic image that is brown/stressed vegetation."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 2] = 160  # Red high (BGR: index 2 = R)
    img[:, :, 1] = 100  # Green moderate
    img[:, :, 0] = 40   # Blue low
    return img


@pytest.fixture
def gray_image_bgr():
    """Uniform gray image — no vegetation."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    return img


# ---------------------------------------------------------------------------
# create_vegetation_mask
# ---------------------------------------------------------------------------

class TestCreateVegetationMask:

    def test_green_image_has_vegetation(self, green_image_bgr):
        mask = create_vegetation_mask(green_image_bgr)
        veg_fraction = np.sum(mask) / mask.size
        assert veg_fraction > 0.5, "Green image should be mostly detected as vegetation"

    def test_gray_image_has_little_vegetation(self, gray_image_bgr):
        mask = create_vegetation_mask(gray_image_bgr)
        veg_fraction = np.sum(mask) / mask.size
        assert veg_fraction < 0.5, "Gray (non-veg) image should not be mostly vegetation"

    def test_output_is_bool(self, green_image_bgr):
        mask = create_vegetation_mask(green_image_bgr)
        assert mask.dtype == bool

    def test_output_shape_matches_input(self, green_image_bgr):
        mask = create_vegetation_mask(green_image_bgr)
        assert mask.shape == green_image_bgr.shape[:2]


# ---------------------------------------------------------------------------
# simulate_nir_advanced
# ---------------------------------------------------------------------------

class TestSimulateNirAdvanced:

    def test_output_shape_matches_input(self, green_image_bgr):
        nir = simulate_nir_advanced(green_image_bgr)
        assert nir.shape == green_image_bgr.shape[:2]

    def test_output_is_uint8(self, green_image_bgr):
        nir = simulate_nir_advanced(green_image_bgr)
        assert nir.dtype == np.uint8

    def test_output_values_in_range(self, green_image_bgr):
        nir = simulate_nir_advanced(green_image_bgr)
        assert nir.min() >= 0
        assert nir.max() <= 255

    def test_green_nir_higher_than_brown(self, green_image_bgr, brown_image_bgr):
        """Healthy green vegetation should simulate higher NIR than brown/stressed."""
        nir_green = simulate_nir_advanced(green_image_bgr)
        nir_brown = simulate_nir_advanced(brown_image_bgr)
        assert np.mean(nir_green) > np.mean(nir_brown), \
            "Green vegetation should have higher simulated NIR than brown"

    def test_raises_on_none(self):
        with pytest.raises((ValueError, AttributeError, TypeError)):
            simulate_nir_advanced(None)


# ---------------------------------------------------------------------------
# compute_ndvi_advanced
# ---------------------------------------------------------------------------

class TestComputeNdviAdvanced:

    def test_output_shapes(self, green_image_bgr):
        ndvi, mask, stats = compute_ndvi_advanced(green_image_bgr)
        assert ndvi.shape == green_image_bgr.shape[:2]
        assert mask.shape == green_image_bgr.shape[:2]

    def test_stats_keys_present(self, green_image_bgr):
        _, _, stats = compute_ndvi_advanced(green_image_bgr)
        for key in ("mean", "std", "min", "max", "vegetation_pixels", "total_pixels"):
            assert key in stats, f"Missing key: {key}"

    def test_normalized_ndvi_in_zero_one(self, green_image_bgr):
        """Normalized NDVI (default) should be in [0, 1]."""
        ndvi, _, _ = compute_ndvi_advanced(green_image_bgr, return_raw=False)
        assert ndvi.min() >= 0.0
        assert ndvi.max() <= 1.0

    def test_raw_ndvi_in_minus_one_one(self, green_image_bgr):
        """Raw NDVI should be in [-1, 1]."""
        ndvi, _, _ = compute_ndvi_advanced(green_image_bgr, return_raw=True)
        assert ndvi.min() >= -1.0 - 1e-6
        assert ndvi.max() <= 1.0 + 1e-6

    def test_green_higher_ndvi_than_brown(self, green_image_bgr, brown_image_bgr):
        _, _, stats_green = compute_ndvi_advanced(green_image_bgr, return_raw=True)
        _, _, stats_brown = compute_ndvi_advanced(brown_image_bgr, return_raw=True)
        assert stats_green["mean"] > stats_brown["mean"], \
            "Green vegetation should have higher NDVI than brown"

    def test_total_pixels_correct(self, green_image_bgr):
        _, _, stats = compute_ndvi_advanced(green_image_bgr)
        assert stats["total_pixels"] == 100 * 100


# ---------------------------------------------------------------------------
# classify_vegetation_health
# ---------------------------------------------------------------------------

class TestClassifyVegetationHealth:

    def _make_stats(self, mean, std=0.05, veg_pixels=5000, total=10000):
        return {
            "mean": mean,
            "std": std,
            "min": mean - 0.1,
            "max": mean + 0.1,
            "vegetation_pixels": veg_pixels,
            "total_pixels": total,
        }

    def test_healthy_classification(self):
        stats = self._make_stats(mean=0.20)
        result = classify_vegetation_health(stats)
        assert result["status"] == "Healthy"

    def test_moderate_classification(self):
        stats = self._make_stats(mean=0.00)
        result = classify_vegetation_health(stats)
        assert result["status"] == "Moderate"

    def test_stressed_classification(self):
        stats = self._make_stats(mean=-0.30)
        result = classify_vegetation_health(stats)
        assert result["status"] == "Stressed"

    def test_no_vegetation(self):
        # Less than 5% vegetation coverage
        stats = self._make_stats(mean=0.20, veg_pixels=100, total=10000)
        result = classify_vegetation_health(stats)
        assert result["status"] == "No Vegetation"

    def test_result_has_required_keys(self):
        stats = self._make_stats(mean=0.20)
        result = classify_vegetation_health(stats)
        for key in ("status", "icon", "color", "confidence", "description",
                    "mean_ndvi", "std_ndvi", "vegetation_percentage"):
            assert key in result, f"Missing key: {key}"

    def test_high_confidence_low_std(self):
        stats = self._make_stats(mean=0.20, std=0.05)
        result = classify_vegetation_health(stats)
        assert result["confidence"] == "High"

    def test_low_confidence_high_std(self):
        stats = self._make_stats(mean=0.20, std=0.25)
        result = classify_vegetation_health(stats)
        assert result["confidence"] == "Low"
