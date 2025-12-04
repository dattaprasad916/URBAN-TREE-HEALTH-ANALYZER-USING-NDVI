#!/usr/bin/env python3
"""
Check actual NDVI values for autumn leaves
"""

import cv2
import numpy as np
from utils.ndvi import create_vegetation_mask, compute_ndvi_advanced, classify_vegetation_health, simulate_nir_advanced

# Test with autumn orange color
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
test_img[:, :] = [180, 120, 60]  # RGB: Orange autumn

img_bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)

# Get detailed analysis
veg_mask = create_vegetation_mask(img_bgr)
nir = simulate_nir_advanced(img_bgr)
ndvi_norm, _, ndvi_stats = compute_ndvi_advanced(img_bgr, return_raw=False)
ndvi_raw, _, _ = compute_ndvi_advanced(img_bgr, return_raw=True)

classification = classify_vegetation_health(ndvi_stats)

print("🍂 Autumn Orange Leaves RGB[180, 120, 60]:")
print(f"  Red channel: {img_bgr[0,0,2]}")
print(f"  Green channel: {img_bgr[0,0,1]}")
print(f"  Blue channel: {img_bgr[0,0,0]}")
print(f"\n  Simulated NIR: {nir[0,0]}")
print(f"  Raw NDVI: {ndvi_stats['mean']:.3f}")
print(f"  Normalized NDVI: {np.mean(ndvi_norm):.3f}")
print(f"\n  Vegetation detected: {(np.sum(veg_mask)/veg_mask.size)*100:.1f}%")
print(f"  Classification: {classification['status']} {classification['icon']}")
print(f"  NDVI Threshold: Healthy > 0.10, Moderate > -0.10")
print(f"\n  ❓ Why 'Healthy'? Raw NDVI {ndvi_stats['mean']:.3f} > 0.10")

print("\n" + "="*60)

# Test with healthy green
test_img2 = np.zeros((100, 100, 3), dtype=np.uint8)
test_img2[:, :] = [60, 150, 70]  # RGB: Green

img_bgr2 = cv2.cvtColor(test_img2, cv2.COLOR_RGB2BGR)
nir2 = simulate_nir_advanced(img_bgr2)
_, _, ndvi_stats2 = compute_ndvi_advanced(img_bgr2)
classification2 = classify_vegetation_health(ndvi_stats2)

print("🌿 Healthy Green Leaves RGB[60, 150, 70]:")
print(f"  Simulated NIR: {nir2[0,0]}")
print(f"  Raw NDVI: {ndvi_stats2['mean']:.3f}")
print(f"  Classification: {classification2['status']} {classification2['icon']}")
print(f"\n  ✅ Both are above 0.10 threshold, so both show as 'Healthy'")
print(f"  💡 Need to adjust NIR simulation for autumn leaves to get lower NDVI!")