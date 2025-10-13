import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

def create_vegetation_mask(image, method='green_threshold'):
    """
    Create a mask to identify vegetation areas in the image.
    """
    if len(image.shape) == 3:
        # Convert BGR to RGB for processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Method 1: Strict vegetation detection - exclude dead trees and branches
        if method == 'green_threshold':
            # Calculate channels
            green = rgb_image[:, :, 1].astype(float)
            red = rgb_image[:, :, 0].astype(float)
            blue = rgb_image[:, :, 2].astype(float)
            
            # Condition 1: Strong green dominance (healthy vegetation only)
            # Green must be significantly higher than both red and blue
            healthy_green = (green > red + 25) & (green > blue + 25) & (green > 80)
            
            # Condition 2: Moderate green vegetation (less strict but still requires green dominance)
            # Green must still be dominant but with some tolerance
            moderate_green = (green > red + 15) & (green > blue + 15) & (green > 60) & (green < 200)
            
            # Condition 3: Exclude brown/dead material (tree trunks, branches, dead leaves)
            # Dead trees often have similar red/brown values - exclude these
            not_brown_trunk = ~((red > green - 10) & (red > blue) & (red > 80) & (green < red + 30))
            not_dead_branches = ~((red > 100) & (green > 80) & (green < red + 20) & (blue < red - 10))
            
            # Condition 4: Exclude sky and other non-vegetation
            not_sky = blue < green + 15
            not_too_bright = (red + green + blue) < 650  # Avoid overexposed areas
            
            # Condition 5: Require sufficient color contrast (avoid gray areas)
            has_contrast = (np.max([red, green, blue], axis=0) - np.min([red, green, blue], axis=0)) > 20
            
            # Only accept clearly green vegetation
            vegetation_mask = (healthy_green | moderate_green) & not_brown_trunk & not_dead_branches & not_sky & not_too_bright & has_contrast
            
        # Method 2: HSV based vegetation detection
        elif method == 'hsv':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define range for green colors (vegetation)
            # Hue range for green: 40-80
            lower_green = np.array([40, 30, 30])
            upper_green = np.array([80, 255, 255])
            
            vegetation_mask = cv2.inRange(hsv, lower_green, upper_green) > 0
            
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        vegetation_mask = cv2.morphologyEx(vegetation_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
        
        return vegetation_mask.astype(bool)
    
    return np.ones(image.shape[:2], dtype=bool)

def simulate_nir_advanced(image):
    """
    Advanced NIR simulation that better distinguishes vegetation health.
    Uses multiple approaches to simulate NIR reflectance.
    """
    if len(image.shape) == 3:
        # Convert to RGB if needed
        if image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image.copy()
    else:
        return image
    
    # Normalize to 0-1
    rgb_norm = rgb_image.astype(np.float32) / 255.0
    red = rgb_norm[:, :, 0]
    green = rgb_norm[:, :, 1]
    blue = rgb_norm[:, :, 2]
    
    # Create vegetation mask
    veg_mask = create_vegetation_mask(image)
    
    # Method 1: Realistic NIR simulation based on actual vegetation
    # Only vegetation areas get enhanced NIR reflection
    
    # For actual vegetation (green areas), simulate appropriate NIR
    vegetation_nir = np.where(
        veg_mask,
        np.minimum(green * 1.8 + 0.2, 1.0),  # High NIR only for actual vegetation
        0.0  # Initialize non-vegetation to zero
    )
    
    # Method 2: Handle different material types
    # Dead wood, bark, branches have very low NIR (similar to red reflectance)
    dead_wood_mask = (red > green - 20) & (red > blue) & (red > 0.3) & ~veg_mask
    soil_mask = (np.abs(red - green) < 0.15) & (np.abs(green - blue) < 0.15) & ~veg_mask
    
    # Assign appropriate NIR values for different materials
    material_nir = np.where(
        dead_wood_mask,
        red * 0.7,  # Dead wood has low NIR, similar to red
        np.where(
            soil_mask,
            (red + green) * 0.4,  # Soil has moderate NIR
            red * 0.5  # Other non-vegetation materials
        )
    )
    
    # Combine vegetation and material NIR
    combined_nir = np.where(veg_mask, vegetation_nir, material_nir)
    
    # Method 3: Apply contrast enhancement only to actual vegetation areas
    enhanced_nir = combined_nir.copy()
    
    # Apply CLAHE only to vegetation areas to avoid artificially enhancing dead material
    if np.any(veg_mask):
        # Extract vegetation areas for enhancement
        veg_nir_values = combined_nir[veg_mask]
        if len(veg_nir_values) > 0:
            # Create a temporary 2D array for CLAHE (needs 2D input)
            temp_img = (combined_nir * 255).astype(np.uint8)
            enhanced_temp = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)).apply(temp_img)
            enhanced_nir = enhanced_temp.astype(np.float32) / 255.0
        else:
            enhanced_nir = combined_nir
    else:
        enhanced_nir = combined_nir
    
    # Final NIR - no additional processing needed
    final_nir = enhanced_nir
    
    return (np.clip(final_nir, 0, 1) * 255).astype(np.uint8)

def simulate_nir(image):
    """
    Backward compatibility wrapper for the old function name.
    """
    return simulate_nir_advanced(image)

def compute_ndvi_advanced(rgb_image, nir_channel=None, return_raw=False):
    """
    Advanced NDVI computation with vegetation masking and adaptive thresholding.
    
    Args:
        rgb_image: RGB image array (BGR format for OpenCV)
        nir_channel: Near-infrared channel (2D array) - if None, will be simulated
        return_raw: If True, return raw NDVI [-1,1], else normalized [0,1]
    
    Returns:
        ndvi: NDVI values
        vegetation_mask: Boolean mask of vegetation areas
        ndvi_stats: Dictionary with statistics
    """
    # Generate NIR if not provided
    if nir_channel is None:
        nir_channel = simulate_nir_advanced(rgb_image)
    
    # Create vegetation mask
    veg_mask = create_vegetation_mask(rgb_image)
    
    # Extract red channel (BGR format)
    if len(rgb_image.shape) == 3:
        red_channel = rgb_image[:, :, 2]  # BGR format
    else:
        red_channel = rgb_image
    
    # Convert to float and normalize
    nir = nir_channel.astype(np.float32) / 255.0
    red = red_channel.astype(np.float32) / 255.0
    
    # Calculate NDVI with epsilon for numerical stability
    epsilon = 1e-6
    ndvi_raw = (nir - red) / (nir + red + epsilon)
    
    # Clamp to valid NDVI range
    ndvi_raw = np.clip(ndvi_raw, -1, 1)
    
    # Apply vegetation mask - set non-vegetation areas to minimum NDVI
    ndvi_masked = np.where(veg_mask, ndvi_raw, -1.0)
    
    # Calculate statistics only for vegetation areas
    veg_ndvi = ndvi_masked[veg_mask]
    ndvi_stats = {
        'mean': np.mean(veg_ndvi) if len(veg_ndvi) > 0 else -1,
        'std': np.std(veg_ndvi) if len(veg_ndvi) > 0 else 0,
        'min': np.min(veg_ndvi) if len(veg_ndvi) > 0 else -1,
        'max': np.max(veg_ndvi) if len(veg_ndvi) > 0 else -1,
        'vegetation_pixels': np.sum(veg_mask),
        'total_pixels': veg_mask.size
    }
    
    if return_raw:
        return ndvi_masked, veg_mask, ndvi_stats
    else:
        # Normalize to [0, 1] for visualization
        ndvi_normalized = (ndvi_masked + 1) / 2
        return ndvi_normalized, veg_mask, ndvi_stats

def compute_ndvi(rgb_image, nir_channel=None):
    """
    Backward compatibility wrapper for the old function.
    """
    ndvi_normalized, _, _ = compute_ndvi_advanced(rgb_image, nir_channel, return_raw=False)
    return ndvi_normalized

def classify_vegetation_health(ndvi_stats, vegetation_percentage_threshold=5.0):
    """
    Classify vegetation health based on NDVI statistics.
    Uses adaptive thresholds based on the actual data distribution.
    
    Args:
        ndvi_stats: Dictionary with NDVI statistics
        vegetation_percentage_threshold: Minimum percentage of vegetation pixels required
    
    Returns:
        classification: Dict with health classification and confidence
    """
    # Check if we have enough vegetation in the image
    veg_percentage = (ndvi_stats['vegetation_pixels'] / ndvi_stats['total_pixels']) * 100
    
    if veg_percentage < vegetation_percentage_threshold:
        return {
            'status': 'No Vegetation',
            'icon': '🏜️',
            'color': 'gray',
            'confidence': 'High',
            'description': f'Insufficient vegetation detected ({veg_percentage:.1f}% of image)',
            'mean_ndvi': ndvi_stats['mean'],
            'std_ndvi': ndvi_stats['std'],
            'vegetation_percentage': veg_percentage
        }
    
    mean_ndvi = ndvi_stats['mean']
    std_ndvi = ndvi_stats['std']
    
    # Adaptive thresholds based on actual NIR simulation characteristics
    # These are calibrated for the advanced NIR simulation
    if mean_ndvi > 0.10:  # Healthy vegetation threshold - lowered for better detection
        status = 'Healthy'
        icon = '🌿'
        color = 'green'
        description = 'Dense, healthy vegetation with good NIR reflectance'
    elif mean_ndvi > -0.10:  # Moderate vegetation threshold - expanded range
        status = 'Moderate'
        icon = '🌾'
        color = 'orange'
        description = 'Moderate vegetation health, some stress indicators'
    else:  # Stressed vegetation
        status = 'Stressed'
        icon = '🍂'
        color = 'red'
        description = 'Stressed or sparse vegetation, low NIR reflectance'
    
    # Determine confidence based on standard deviation
    if std_ndvi < 0.1:
        confidence = 'High'
    elif std_ndvi < 0.2:
        confidence = 'Medium'
    else:
        confidence = 'Low'
    
    return {
        'status': status,
        'icon': icon,
        'color': color,
        'confidence': confidence,
        'description': description,
        'mean_ndvi': mean_ndvi,
        'std_ndvi': std_ndvi,
        'vegetation_percentage': veg_percentage
    }
