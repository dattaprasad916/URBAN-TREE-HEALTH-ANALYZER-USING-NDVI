#!/usr/bin/env python3
"""
Quick test to validate the dead tree detection fix
"""

import cv2
import numpy as np
import os
from utils.ndvi import simulate_nir_advanced, compute_ndvi_advanced, classify_vegetation_health, create_vegetation_mask

def test_dead_tree_scenarios():
    """Test various dead tree scenarios"""
    print("🏺 Testing dead tree detection fix...")
    
    # Test scenarios that should NOT be detected as vegetation
    test_cases = [
        {
            'name': 'Dead Brown Trunk',
            'rgb': [120, 100, 80],  # Brown trunk color
            'expected_vegetation': False
        },
        {
            'name': 'Dry Branches', 
            'rgb': [140, 120, 90],  # Dry branch color
            'expected_vegetation': False
        },
        {
            'name': 'Dead Gray Wood',
            'rgb': [110, 105, 95],  # Gray dead wood
            'expected_vegetation': False
        },
        {
            'name': 'Healthy Green Leaves',
            'rgb': [60, 180, 80],   # Bright green - should be detected
            'expected_vegetation': True
        },
        {
            'name': 'Moderate Green Grass',
            'rgb': [90, 140, 70],   # Medium green - should be detected
            'expected_vegetation': True
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
        
        # Test full NDVI analysis
        ndvi_norm, _, ndvi_stats = compute_ndvi_advanced(img_bgr)
        classification = classify_vegetation_health(ndvi_stats)
        
        # Check if result matches expectation
        is_vegetation_detected = vegetation_percentage > 5.0
        result_correct = is_vegetation_detected == test_case['expected_vegetation']
        
        status_icon = "✅" if result_correct else "❌"
        expected_text = "should detect" if test_case['expected_vegetation'] else "should NOT detect"
        
        print(f"   {status_icon} {test_case['name']} RGB{test_case['rgb']}: "
              f"{vegetation_percentage:.1f}% vegetation detected ({expected_text})")
        print(f"      Status: {classification['status']} {classification['icon']}, "
              f"NDVI: {classification['mean_ndvi']:.3f}")
        
        if not result_correct:
            print(f"      ❌ FAILED: Expected vegetation={test_case['expected_vegetation']}, "
                  f"Got vegetation={is_vegetation_detected}")
    
    return True

def test_sample_image_fix():
    """Test the actual sample image that was problematic"""
    print("\n🖼️  Testing with actual sample images...")
    
    sample_images = [
        "static/raw_images/tree_sample.jpg",
        "static/raw_images/OIP.webp", 
        "static/raw_images/pexels-photo-1080401.jpeg"
    ]
    
    for img_path in sample_images:
        if os.path.exists(img_path):
            print(f"\n📸 Processing: {os.path.basename(img_path)}")
            
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Analyze with new algorithm
            ndvi_norm, veg_mask, ndvi_stats = compute_ndvi_advanced(img)
            classification = classify_vegetation_health(ndvi_stats)
            
            print(f"   🌿 Vegetation coverage: {classification['vegetation_percentage']:.1f}%")
            print(f"   📊 Mean NDVI: {classification['mean_ndvi']:.3f}")
            print(f"   🏥 Status: {classification['status']} {classification['icon']}")
            print(f"   🎯 Confidence: {classification['confidence']}")
            
            # Special check for the dry tree image (should have low vegetation percentage)
            if 'dry' in img_path.lower() or classification['vegetation_percentage'] < 5:
                if classification['status'] in ['No Vegetation', 'Stressed']:
                    print("   ✅ CORRECT: Dead/dry trees properly classified")
                else:
                    print("   ❌ ISSUE: Dead trees still classified as healthy/moderate")
        
    return True

if __name__ == "__main__":
    print("🌳 Dead Tree Detection Fix - Validation Test\n")
    
    print("Testing synthetic scenarios...")
    test_dead_tree_scenarios()
    
    print("\nTesting real images...")
    test_sample_image_fix()
    
    print("\n✅ Test complete. Check results above to verify dead tree detection is working correctly.")
    print("\n💡 Expected results:")
    print("   • Brown/gray dead wood: Should NOT be detected as vegetation")
    print("   • Green leaves/grass: Should be detected as vegetation")  
    print("   • Dead tree images: Should show 'No Vegetation' or 'Stressed' status")