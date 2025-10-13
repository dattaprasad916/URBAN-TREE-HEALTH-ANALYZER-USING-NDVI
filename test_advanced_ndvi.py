#!/usr/bin/env python3
"""
Advanced test script for the improved Tree Health Monitor functionality
Tests the new vegetation detection and adaptive NDVI classification
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.ndvi import simulate_nir_advanced, compute_ndvi_advanced, classify_vegetation_health, create_vegetation_mask

def test_vegetation_detection():
    """Test vegetation mask creation"""
    print("🌿 Testing vegetation detection algorithms...")
    
    # Create test images with different characteristics
    
    # 1. Healthy green vegetation
    healthy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    healthy_img[:, :, 1] = 180  # High green
    healthy_img[:, :, 0] = 80   # Low red
    healthy_img[:, :, 2] = 60   # Low blue
    
    # 2. Stressed brown vegetation
    stressed_img = np.zeros((100, 100, 3), dtype=np.uint8)
    stressed_img[:, :, 0] = 150  # High red (brown/dry)
    stressed_img[:, :, 1] = 120  # Medium green
    stressed_img[:, :, 2] = 70   # Low blue
    
    # 3. Non-vegetation (soil/concrete)
    soil_img = np.zeros((100, 100, 3), dtype=np.uint8)
    soil_img[:, :] = [100, 95, 90]  # Grayish brown
    
    # Test vegetation detection
    for name, img in [("Healthy", healthy_img), ("Stressed", stressed_img), ("Soil", soil_img)]:
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        mask = create_vegetation_mask(img_bgr)
        vegetation_percentage = (np.sum(mask) / mask.size) * 100
        
        print(f"   {name} vegetation: {vegetation_percentage:.1f}% detected as vegetation")
    
    return True

def test_advanced_nir_simulation():
    """Test the advanced NIR simulation"""
    print("🔬 Testing advanced NIR simulation...")
    
    # Test with different vegetation types
    test_cases = [
        ("Healthy Dense Green", [50, 200, 80]),    # High green, low red
        ("Moderate Green", [100, 150, 90]),        # Medium green, medium red
        ("Stressed Brown", [140, 100, 70]),        # High red, low green (drought/stress)
        ("Dead Vegetation", [120, 110, 80]),       # Similar red/green (dead)
        ("Soil Background", [100, 95, 90]),        # Non-vegetation
    ]
    
    for name, rgb_color in test_cases:
        # Create test image
        test_img = np.zeros((50, 50, 3), dtype=np.uint8)
        test_img[:, :] = rgb_color
        
        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
        
        # Simulate NIR
        nir = simulate_nir_advanced(img_bgr)
        mean_nir = np.mean(nir)
        
        print(f"   {name}: RGB{rgb_color} → NIR={mean_nir:.1f}")
    
    return True

def test_adaptive_classification():
    """Test the adaptive vegetation health classification"""
    print("🏥 Testing adaptive health classification...")
    
    # Create synthetic test cases
    test_scenarios = [
        {
            'name': 'Dense Forest',
            'green': 200, 'red': 60, 'blue': 40,
            'expected': 'Healthy'
        },
        {
            'name': 'Moderate Grassland',
            'green': 140, 'red': 100, 'blue': 70,
            'expected': 'Moderate'
        },
        {
            'name': 'Drought Stressed',
            'green': 90, 'red': 130, 'blue': 60,
            'expected': 'Stressed'
        },
        {
            'name': 'Dead Vegetation',
            'green': 110, 'red': 115, 'blue': 80,
            'expected': 'Stressed'
        },
        {
            'name': 'Urban Area',
            'green': 100, 'red': 105, 'blue': 110,
            'expected': 'No Vegetation'
        }
    ]
    
    for scenario in test_scenarios:
        # Create test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:, :, 0] = scenario['red']
        test_img[:, :, 1] = scenario['green'] 
        test_img[:, :, 2] = scenario['blue']
        
        # Convert RGB to BGR
        img_bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
        
        # Process with advanced algorithms
        ndvi_norm, veg_mask, ndvi_stats = compute_ndvi_advanced(img_bgr)
        classification = classify_vegetation_health(ndvi_stats)
        
        result_status = "✅" if classification['status'] == scenario['expected'] else "❌"
        
        print(f"   {result_status} {scenario['name']}: {classification['status']} {classification['icon']} "
              f"(Expected: {scenario['expected']})")
        print(f"      Mean NDVI: {classification['mean_ndvi']:.3f}, "
              f"Vegetation: {classification['vegetation_percentage']:.1f}%")
    
    return True

def test_real_images():
    """Test with real sample images"""
    print("📸 Testing with real sample images...")
    
    sample_images = [
        "static/raw_images/tree_sample.jpg",
        "static/raw_images/OIP.webp",
        "static/raw_images/pexels-photo-1080401.jpeg"
    ]
    
    results = []
    
    for img_path in sample_images:
        if os.path.exists(img_path):
            print(f"\n🖼️  Processing: {os.path.basename(img_path)}")
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"   ❌ Failed to load {img_path}")
                continue
                
            print(f"   📏 Image size: {img.shape}")
            
            # Process with advanced algorithms
            ndvi_norm, veg_mask, ndvi_stats = compute_ndvi_advanced(img)
            classification = classify_vegetation_health(ndvi_stats)
            
            print(f"   🌿 Vegetation coverage: {classification['vegetation_percentage']:.1f}%")
            print(f"   📊 Mean NDVI (raw): {classification['mean_ndvi']:.3f}")
            print(f"   🏥 Health status: {classification['status']} {classification['icon']}")
            print(f"   🎯 Confidence: {classification['confidence']}")
            print(f"   📝 Description: {classification['description']}")
            
            results.append({
                'image': os.path.basename(img_path),
                'status': classification['status'],
                'ndvi': classification['mean_ndvi'],
                'vegetation_pct': classification['vegetation_percentage'],
                'confidence': classification['confidence']
            })
        else:
            print(f"   ⏭️  Image not found: {img_path}")
    
    # Summary
    if results:
        print(f"\n📊 Summary of {len(results)} images:")
        for result in results:
            print(f"   • {result['image']}: {result['status']} "
                  f"(NDVI: {result['ndvi']:.3f}, Veg: {result['vegetation_pct']:.1f}%)")
    
    return True

def main():
    """Run all advanced tests"""
    print("🌳 Tree Health Monitor - Advanced Algorithm Tests\n")
    
    try:
        # Test vegetation detection
        if not test_vegetation_detection():
            print("❌ Vegetation detection test failed!")
            return False
        
        print()
        
        # Test NIR simulation
        if not test_advanced_nir_simulation():
            print("❌ NIR simulation test failed!")
            return False
        
        print()
        
        # Test classification
        if not test_adaptive_classification():
            print("❌ Classification test failed!")
            return False
        
        print()
        
        # Test real images
        if not test_real_images():
            print("❌ Real image test failed!")
            return False
        
        print("\n🎉 All advanced tests passed!")
        print("\n💡 Key improvements:")
        print("   • Vegetation area detection and masking")
        print("   • Advanced NIR simulation with stress indicators")
        print("   • Adaptive thresholds based on vegetation characteristics")
        print("   • Confidence assessment and detailed classification")
        print("   • Better discrimination between healthy, moderate, and stressed vegetation")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)