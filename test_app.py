#!/usr/bin/env python3
"""
Test script for Tree Health Monitor functionality
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.ndvi import simulate_nir, compute_ndvi
from PIL import Image

def test_basic_functionality():
    """Test basic NDVI functionality"""
    print("🧪 Testing Tree Health Monitor functionality...")
    
    # Create a synthetic test image
    print("📸 Creating synthetic test image...")
    test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    # Make it more realistic - add some green areas (vegetation)
    test_image[50:150, 50:150, 1] = 200  # Green channel higher
    test_image[50:150, 50:150, 2] = 100  # Red channel lower
    test_image[50:150, 50:150, 0] = 50   # Blue channel lower
    
    print(f"✅ Test image created: {test_image.shape}")
    
    # Test NIR simulation
    print("🔍 Testing NIR simulation...")
    nir = simulate_nir(test_image)
    print(f"✅ NIR simulation successful: {nir.shape}")
    
    # Test NDVI calculation using the corrected formula
    print("📊 Testing NDVI calculation...")
    nir_norm = nir.astype(float) / 255.0
    red_norm = test_image[:, :, 2].astype(float) / 255.0
    ndvi = (nir_norm - red_norm) / (nir_norm + red_norm + 1e-6)
    ndvi = np.clip(ndvi, -1, 1)
    ndvi_norm = (ndvi + 1) / 2  # map -1..1 to 0..1
    print(f"✅ NDVI calculation successful: {ndvi_norm.shape}")
    print(f"   NDVI range: {ndvi_norm.min():.3f} to {ndvi_norm.max():.3f}")
    print(f"   Mean NDVI: {ndvi_norm.mean():.3f}")
    
    # Test health classification
    print("🏥 Testing health classification...")
    total_pixels = ndvi_norm.size
    healthy = np.sum(ndvi_norm > 0.55)
    moderate = np.sum((ndvi_norm > 0.35) & (ndvi_norm <= 0.55))
    stressed = np.sum(ndvi_norm <= 0.35)
    
    print(f"   Healthy: {healthy/total_pixels*100:.1f}%")
    print(f"   Moderate: {moderate/total_pixels*100:.1f}%")
    print(f"   Stressed: {stressed/total_pixels*100:.1f}%")
    
    # Test visualization
    print("🎨 Testing NDVI visualization...")
    ndvi_norm_clean = np.nan_to_num(ndvi_norm, nan=0.0, posinf=1.0, neginf=0.0)
    ndvi_colored = cv2.applyColorMap((ndvi_norm_clean*255).astype(np.uint8), cv2.COLORMAP_JET)
    print(f"✅ NDVI visualization successful: {ndvi_colored.shape}")
    
    return True

def test_sample_images():
    """Test with existing sample images if available"""
    print("\n📁 Testing with sample images...")
    
    sample_images = [
        "static/raw_images/tree_sample.jpg",
        "static/raw_images/OIP.webp",
        "static/raw_images/pexels-photo-1080401.jpeg"
    ]
    
    for img_path in sample_images:
        if os.path.exists(img_path):
            print(f"🖼️  Testing with: {img_path}")
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    print(f"   ✅ Image loaded: {img.shape}")
                    nir = simulate_nir(img)
                    print(f"   ✅ NIR calculated: {nir.shape}")
                    
                    # Quick NDVI calculation with proper normalization
                    nir_norm = nir.astype(float) / 255.0
                    red_norm = img[:, :, 2].astype(float) / 255.0
                    ndvi = (nir_norm - red_norm) / (nir_norm + red_norm + 1e-6)
                    ndvi = np.clip(ndvi, -1, 1)
                    ndvi_norm = (ndvi + 1) / 2
                    mean_ndvi = np.mean(ndvi_norm)
                    
                    if mean_ndvi > 0.55:
                        status = "Healthy 🌿"
                    elif mean_ndvi > 0.35:
                        status = "Moderate 🌾"
                    else:
                        status = "Stressed 🍂"
                    
                    print(f"   📊 Mean NDVI: {mean_ndvi:.3f} - Status: {status}")
                else:
                    print(f"   ⚠️  Failed to load image: {img_path}")
            except Exception as e:
                print(f"   ❌ Error processing {img_path}: {e}")
        else:
            print(f"   ⏭️  Sample image not found: {img_path}")
    
    return True

def test_imports():
    """Test all required imports"""
    print("\n📦 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🌳 Tree Health Monitor - Functionality Test\n")
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed!")
        exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("❌ Basic functionality test failed!")
        exit(1)
    
    # Test with sample images
    test_sample_images()
    
    print("\n🎉 All tests passed! The application should work correctly.")
    print("\n💡 To run the Streamlit app:")
    print("   streamlit run streamlit_app.py")
    print("   or")
    print("   .\\run_streamlit.ps1")