#!/usr/bin/env python3
"""
Quick test for autumn/dried leaves detection
"""

import cv2
import numpy as np
from utils.ndvi import create_vegetation_mask, compute_ndvi_advanced, classify_vegetation_health

def test_autumn_colors():
    """Test detection of autumn/dried leaf colors"""
    print("🍂 Testing autumn/dried leaves detection...\n")
    
    # Test cases for autumn foliage colors
    test_cases = [
        {
            'name': 'Bright Orange Autumn Leaves',
            'rgb': [180, 120, 60],  # Orange autumn color
            'expected': 'Should detect as stressed vegetation'
        },
        {
            'name': 'Yellow Autumn Leaves',
            'rgb': [150, 140, 70],  # Yellow autumn color
            'expected': 'Should detect as stressed vegetation'
        },
        {
            'name': 'Brown Dried Leaves',
            'rgb': [120, 90, 60],   # Brown dried color
            'expected': 'Should detect as stressed vegetation'
        },
        {
            'name': 'Dark Brown Trunk',
            'rgb': [80, 70, 50],    # Dark trunk - should NOT detect
            'expected': 'Should NOT detect (dead wood)'
        },
        {
            'name': 'Healthy Green Leaves',
            'rgb': [60, 150, 70],   # Green - should detect as healthy
            'expected': 'Should detect as healthy vegetation'
        }
    ]
    
    for test_case in test_cases:
        # Create test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:, :] = test_case['rgb']
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
        
        # Test vegetation detection
        veg_mask = create_vegetation_mask(img_bgr)
        vegetation_percentage = (np.sum(veg_mask) / veg_mask.size) * 100
        
        # Test full analysis
        ndvi_norm, _, ndvi_stats = compute_ndvi_advanced(img_bgr)
        classification = classify_vegetation_health(ndvi_stats)
        
        is_detected = vegetation_percentage > 5.0
        
        print(f"📋 {test_case['name']} RGB{test_case['rgb']}")
        print(f"   Vegetation detected: {vegetation_percentage:.1f}%")
        print(f"   Status: {classification['status']} {classification['icon']}")
        print(f"   Expected: {test_case['expected']}")
        print(f"   Result: {'✅ CORRECT' if is_detected and 'Should detect' in test_case['expected'] or not is_detected and 'Should NOT' in test_case['expected'] else '❌ NEEDS FIX'}")
        print()

if __name__ == "__main__":
    print("🌳 Autumn/Dried Leaves Detection Test\n")
    print("=" * 60)
    test_autumn_colors()
    print("=" * 60)
    print("\n💡 Expected behavior:")
    print("   • Orange/yellow autumn leaves: Detect as STRESSED vegetation")
    print("   • Brown dried leaves: Detect as STRESSED vegetation")
    print("   • Dark trunks/branches: Should NOT be detected")
    print("   • Green leaves: Detect as HEALTHY vegetation")