# test_ndvi.py
from utils.ndvi import compute_ndvi_advanced
import numpy as np

# Create a dummy 3×3 RGB image to test function
dummy = np.array([
    [[100,120,140],[120,140,160],[140,160,180]],
    [[50,80,120],[80,110,140],[110,140,170]],
    [[30,60,90],[60,90,120],[90,120,150]]
], dtype=np.uint8)

ndvi_raw, veg_mask, stats = compute_ndvi_advanced(dummy, return_raw=True)

print("NDVI Stats:", stats)
print("Vegetation mask:\n", veg_mask.astype(int))
print("NDVI Raw:\n", ndvi_raw)
