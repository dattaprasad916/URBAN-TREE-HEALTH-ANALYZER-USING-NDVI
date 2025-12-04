# utils/ndvi.py
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

        # Method 1: Enhanced vegetation detection - include dried/autumn leaves as stressed vegetation
        if method == 'green_threshold':
            # Calculate channels
            green = rgb_image[:, :, 1].astype(float)
            red = rgb_image[:, :, 0].astype(float)
            blue = rgb_image[:, :, 2].astype(float)

            # Condition 1: Strong green dominance (healthy vegetation)
            healthy_green = (green > red + 25) & (green > blue + 25) & (green > 80)

            # Condition 2: Moderate green vegetation
            moderate_green = (green > red + 15) & (green > blue + 15) & (green > 60) & (green < 200)

            # Condition 3: Dried/autumn leaves (yellow, orange, brown foliage - still vegetation)
            yellow_leaves = (red > 100) & (green > 80) & (red > green - 40) & (red < green + 60) & (blue < red - 20)
            brown_leaves = (red > 90) & (green > 60) & (green < red + 40) & (red > green - 30) & (blue < green - 10) & (red + green > 150)
            orange_leaves = (red > 120) & (green > 60) & (green < red) & (red - green < 80) & (blue < green)

            # Stressed/dried foliage detection
            stressed_foliage = yellow_leaves | brown_leaves | orange_leaves

            # Condition 4: Exclude ONLY dead woody material (trunks, bare branches)
            dead_wood_trunk = (red > 80) & (green > 60) & (abs(red - green) < 25) & (blue < green - 15) & (red + green + blue < 350)
            very_dark_branches = (red < 100) & (green < 90) & (blue < 80) & (red + green + blue < 250)

            # Condition 5: Exclude sky and bright non-vegetation
            not_sky = blue < green + 30
            not_too_bright = (red + green + blue) < 680

            # Condition 6: Require sufficient color variation
            has_contrast = (np.max([red, green, blue], axis=0) - np.min([red, green, blue], axis=0)) > 15

            # Accept healthy green OR stressed/dried foliage, but exclude dead wood
            vegetation_mask = ((healthy_green | moderate_green | stressed_foliage) &
                               ~dead_wood_trunk & ~very_dark_branches &
                               not_sky & not_too_bright & has_contrast)

        # Method 2: HSV based vegetation detection
        elif method == 'hsv':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 30, 30])
            upper_green = np.array([80, 255, 255])
            vegetation_mask = cv2.inRange(hsv, lower_green, upper_green) > 0

        else:
            # default fallback - everything True
            vegetation_mask = np.ones(image.shape[:2], dtype=bool)

        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        vegetation_mask = cv2.morphologyEx(vegetation_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)

        return vegetation_mask.astype(bool)

    # For grayscale or unknown shapes, mark all as non-vegetation True (fallback)
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

    # Detect stressed foliage (yellow, orange, brown leaves)
    stressed_foliage_mask = (
        ((red > green - 40) & (red < green + 60) & (red > 100) & (blue < red - 15)) |
        ((red > green) & (red > 120) & (green > 60) & (blue < green)) |
        ((red > 90) & (green > 60) & (green < red + 40) & (blue < green - 10))
    )

    # Detect healthy green vegetation
    healthy_green_mask = (green > red + 20) & (green > blue + 20) & (green > 70)

    # Assign NIR based on vegetation health
    vegetation_nir = np.where(
        veg_mask & healthy_green_mask,
        np.minimum(green * 1.8 + 0.2, 1.0),
        np.where(
            veg_mask & stressed_foliage_mask,
            np.minimum(red * 0.35 + green * 0.25, 0.55),
            np.where(
                veg_mask,
                np.minimum(green * 1.0 + 0.1, 0.85),
                0.0
            )
        )
    )

    # Handle different material types
    dead_wood_mask = (red > green - 20) & (red > blue) & (red > 0.3) & ~veg_mask
    soil_mask = (np.abs(red - green) < 0.15) & (np.abs(green - blue) < 0.15) & ~veg_mask

    material_nir = np.where(
        dead_wood_mask,
        red * 0.7,
        np.where(
            soil_mask,
            (red + green) * 0.4,
            red * 0.5
        )
    )

    combined_nir = np.where(veg_mask, vegetation_nir, material_nir)

    final_nir = combined_nir

    return (np.clip(final_nir, 0, 1) * 255).astype(np.uint8)


def simulate_nir(image):
    """Backward compatibility wrapper for the old function name."""
    return simulate_nir_advanced(image)


def compute_ndvi_advanced(rgb_image, nir_channel=None, return_raw=False):
    """
    Advanced NDVI computation with vegetation masking and adaptive thresholding.
    """
    # Generate NIR if not provided
    if nir_channel is None:
        nir_channel = simulate_nir_advanced(rgb_image)

    # Create vegetation mask
    veg_mask = create_vegetation_mask(rgb_image)

    # Extract red channel (assume BGR input)
    if len(rgb_image.shape) == 3:
        # BGR -> red is index 2
        red_channel = rgb_image[:, :, 2]
    else:
        red_channel = rgb_image

    # Convert to float and normalize
    nir = nir_channel.astype(np.float32) / 255.0
    red = red_channel.astype(np.float32) / 255.0

    # Calculate NDVI with epsilon for numerical stability
    epsilon = 1e-6
    ndvi_raw = (nir - red) / (nir + red + epsilon)
    ndvi_raw = np.clip(ndvi_raw, -1, 1)

    # Apply vegetation mask - set non-vegetation areas to minimum NDVI
    ndvi_masked = np.where(veg_mask, ndvi_raw, -1.0)

    # Calculate statistics only for vegetation areas
    veg_ndvi = ndvi_masked[veg_mask]
    ndvi_stats = {
        'mean': float(np.mean(veg_ndvi)) if len(veg_ndvi) > 0 else -1.0,
        'std': float(np.std(veg_ndvi)) if len(veg_ndvi) > 0 else 0.0,
        'min': float(np.min(veg_ndvi)) if len(veg_ndvi) > 0 else -1.0,
        'max': float(np.max(veg_ndvi)) if len(veg_ndvi) > 0 else -1.0,
        'vegetation_pixels': int(np.sum(veg_mask)),
        'total_pixels': int(veg_mask.size)
    }

    if return_raw:
        return ndvi_masked, veg_mask, ndvi_stats
    else:
        ndvi_normalized = (ndvi_masked + 1) / 2
        return ndvi_normalized, veg_mask, ndvi_stats


def compute_ndvi(rgb_image, nir_channel=None):
    """Backward compatibility wrapper for the old function."""
    ndvi_normalized, _, _ = compute_ndvi_advanced(rgb_image, nir_channel, return_raw=False)
    return ndvi_normalized


def classify_vegetation_health(ndvi_stats, vegetation_percentage_threshold=5.0):
    """
    Classify vegetation health based on NDVI statistics.
    """
    veg_percentage = (ndvi_stats['vegetation_pixels'] / ndvi_stats['total_pixels']) * 100 if ndvi_stats['total_pixels'] > 0 else 0.0

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

    if mean_ndvi > 0.10:
        status = 'Healthy'
        icon = '🌿'
        color = 'green'
        description = 'Dense, healthy vegetation with good NIR reflectance'
    elif mean_ndvi > -0.10:
        status = 'Moderate'
        icon = '🌾'
        color = 'orange'
        description = 'Moderate vegetation health, some stress indicators'
    else:
        status = 'Stressed'
        icon = '🍂'
        color = 'red'
        description = 'Stressed or sparse vegetation, low NIR reflectance'

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

# Optional: allow running this file directly for a quick sanity check (no GUI)
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        ndvi_masked, veg_mask, stats = compute_ndvi_advanced(img, return_raw=True)
        print("NDVI stats:", stats)
    else:
        print("utils.ndvi loaded successfully. No image provided for direct run.")
