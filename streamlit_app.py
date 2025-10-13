import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from PIL import Image
from utils.ndvi import simulate_nir_advanced, compute_ndvi_advanced, classify_vegetation_health
import io

# Page configuration
st.set_page_config(
    page_title="Urban Tree Health Monitor",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories if they don't exist
UPLOAD_FOLDER = "static/raw_images"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def process_image(image_array):
    """Process the uploaded image and calculate NDVI using advanced algorithms"""
    
    # Convert PIL image to cv2 format
    if len(image_array.shape) == 3:
        image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image_cv2 = image_array
    
    # Use advanced NDVI computation
    ndvi_norm, vegetation_mask, ndvi_stats = compute_ndvi_advanced(image_cv2)
    
    # Get raw NDVI for analysis
    ndvi_raw, _, _ = compute_ndvi_advanced(image_cv2, return_raw=True)
    
    # Classify vegetation health
    health_classification = classify_vegetation_health(ndvi_stats)
    
    return image_cv2, ndvi_raw, ndvi_norm, vegetation_mask, ndvi_stats, health_classification

def create_ndvi_visualization(ndvi_norm):
    """Create NDVI heatmap with legend"""
    
    # NDVI color map - handle NaN values
    ndvi_norm_clean = np.nan_to_num(ndvi_norm, nan=0.0, posinf=1.0, neginf=0.0)
    ndvi_colored = cv2.applyColorMap((ndvi_norm_clean*255).astype(np.uint8), cv2.COLORMAP_JET)
    ndvi_colored = cv2.cvtColor(ndvi_colored, cv2.COLOR_BGR2RGB)
    
    # Create legend
    legend_height = 256
    legend_width = 50
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    for i in range(legend_height):
        value = 1 - i / (legend_height - 1)
        color = cv2.applyColorMap(np.array([[int(value*255)]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0]
        legend[i, :, :] = color
    
    # Convert legend to RGB
    legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
    
    # Add text labels to legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(legend, 'Healthy', (5, 20), font, 0.3, (255,255,255), 1)
    cv2.putText(legend, 'Moderate', (5, legend_height//2 + 5), font, 0.3, (255,255,255), 1)
    cv2.putText(legend, 'Stressed', (5, legend_height - 10), font, 0.3, (255,255,255), 1)
    
    # Overlay legend on NDVI image
    if ndvi_colored.shape[0] >= legend_height and ndvi_colored.shape[1] >= legend_width:
        ndvi_colored[0:legend_height, 0:legend_width] = legend
    
    return ndvi_colored

def calculate_health_stats_advanced(ndvi_norm, vegetation_mask, ndvi_stats, health_classification):
    """Calculate advanced health statistics with vegetation masking"""
    
    # Check if this is a "No Vegetation" case
    if health_classification['status'] == 'No Vegetation':
        # For no vegetation, all percentages should be 0
        health_percent = {
            "Healthy": 0.0,
            "Moderate": 0.0,
            "Stressed": 0.0
        }
        return health_percent, health_classification['status'], health_classification['color'], health_classification['mean_ndvi']
    
    # Calculate percentages based on vegetation pixels only
    veg_ndvi = ndvi_norm[vegetation_mask]
    
    if len(veg_ndvi) > 0:
        # Use adaptive thresholds from the health classification
        healthy_threshold = 0.6  # Normalized equivalent of 0.10 raw NDVI
        moderate_threshold = 0.45  # Normalized equivalent of -0.10 raw NDVI
        
        healthy = np.sum(veg_ndvi > healthy_threshold)
        moderate = np.sum((veg_ndvi > moderate_threshold) & (veg_ndvi <= healthy_threshold))
        stressed = np.sum(veg_ndvi <= moderate_threshold)
        total_veg_pixels = len(veg_ndvi)
        
        health_percent = {
            "Healthy": round(healthy / total_veg_pixels * 100, 1),
            "Moderate": round(moderate / total_veg_pixels * 100, 1),
            "Stressed": round(stressed / total_veg_pixels * 100, 1)
        }
    else:
        # Fallback case - should not happen if vegetation detection works correctly
        health_percent = {
            "Healthy": 0.0,
            "Moderate": 0.0,
            "Stressed": 0.0
        }
    
    return health_percent, health_classification['status'], health_classification['color'], health_classification['mean_ndvi']

def create_plots(ndvi_norm, health_percent):
    """Create histogram and bar chart"""
    
    # NDVI histogram
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.hist(ndvi_norm.ravel(), bins=50, color='green', alpha=0.7, edgecolor='black')
    ax1.set_title('NDVI Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('NDVI Value (0-1)')
    ax1.set_ylabel('Pixel Count')
    ax1.grid(True, alpha=0.3)
    
    # Health bar chart
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    labels = ['Healthy', 'Moderate', 'Stressed']
    values = [health_percent['Healthy'], health_percent['Moderate'], health_percent['Stressed']]
    colors = ['green', 'orange', 'red']
    
    bars = ax2.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title('Tree Health Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Percentage (%)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value}%', ha='center', va='bottom', fontweight='bold')
    
    return fig1, fig2

# Main Streamlit UI
def main():
    st.title("🌳 Urban Tree Health Monitor")
    st.markdown("Upload an image of trees or vegetation to analyze their health using NDVI (Normalized Difference Vegetation Index)")
    
    # Sidebar for information
    with st.sidebar:
        st.header("ℹ️ About Advanced NDVI Analysis")
        st.markdown("""
        **Enhanced NDVI** with vegetation detection:
        - **Healthy** 🌿: Dense, healthy vegetation with good NIR reflectance
        - **Moderate** 🌾: Moderate vegetation health with some stress
        - **Stressed** 🍂: Sparse or stressed vegetation, low NIR
        - **No Vegetation** 🏜️: Insufficient vegetation detected
        
        **Key Improvements:**
        - Vegetation area masking
        - Adaptive thresholds
        - Stress indicator detection
        - Confidence assessment
        """)
        
        st.header("🔬 Technical Details")
        st.markdown("""
        **NIR Simulation:**
        - Green dominance detection
        - Stress indicator analysis  
        - Adaptive contrast enhancement
        
        **Classification:**
        - Raw NDVI > 0.10: Healthy
        - Raw NDVI > -0.10: Moderate
        - Raw NDVI ≤ -0.10: Stressed
        """)
        
        st.header("📊 Color Legend")
        st.markdown("""
        - 🟢 **Green/Blue**: Healthy vegetation areas
        - 🟡 **Yellow/Orange**: Moderate health areas
        - 🔴 **Red**: Stressed/sparse vegetation
        - ⚫ **Black**: Non-vegetation areas
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Upload an image of trees or vegetation for health analysis"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, caption="Uploaded Image", width="stretch")
        
        # Process image
        with st.spinner("Processing image and calculating NDVI..."):
            try:
                image_cv2, ndvi_raw, ndvi_norm, vegetation_mask, ndvi_stats, health_classification = process_image(image_array)
                ndvi_colored = create_ndvi_visualization(ndvi_norm)
                health_percent, status, status_color, mean_ndvi = calculate_health_stats_advanced(ndvi_norm, vegetation_mask, ndvi_stats, health_classification)
                
                with col2:
                    st.subheader("🌿 NDVI Heatmap")
                    st.image(ndvi_colored, caption="NDVI Analysis Result", width="stretch")
                
                # Health status
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Main status with enhanced information
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"### Overall Status")
                    st.markdown(f"<h2 style='color: {status_color};'>{health_classification['icon']} {status}</h2>", unsafe_allow_html=True)
                    st.markdown(f"*Confidence: {health_classification['confidence']}*")
                
                with col2:
                    st.markdown(f"### Mean NDVI")
                    st.markdown(f"<h2>{mean_ndvi:.3f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"*Raw NDVI: {health_classification['mean_ndvi']:.3f}*")
                
                with col3:
                    st.markdown(f"### Vegetation Coverage")
                    st.markdown(f"<h2>{health_classification['vegetation_percentage']:.1f}%</h2>", unsafe_allow_html=True)
                    st.markdown(f"*{ndvi_stats['vegetation_pixels']:,} pixels*")
                    
                with col4:
                    st.markdown(f"### NDVI Std Dev")
                    st.markdown(f"<h2>{ndvi_stats['std']:.3f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"*Uniformity indicator*")
                
                # Add description
                st.info(f"📝 **Analysis**: {health_classification['description']}")
                
                # Health percentages
                st.markdown("### 📈 Health Distribution")
                
                # Special handling for no vegetation
                if health_classification['status'] == 'No Vegetation':
                    st.info("🏜️ **No vegetation detected in this image.** Health percentages are not applicable for non-vegetation areas (dead trees, buildings, soil, etc.)")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🌿 Healthy", "0.0%", help="No vegetation detected")
                    with col2:
                        st.metric("🌾 Moderate", "0.0%", help="No vegetation detected")
                    with col3:
                        st.metric("🍂 Stressed", "0.0%", help="No vegetation detected")
                else:
                    # Normal health distribution for vegetation
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🌿 Healthy", f"{health_percent['Healthy']}%", 
                                 delta=f"{health_percent['Healthy'] - 33.3:.1f}% vs avg")
                    
                    with col2:
                        st.metric("🌾 Moderate", f"{health_percent['Moderate']}%",
                                 delta=f"{health_percent['Moderate'] - 33.3:.1f}% vs avg")
                    
                    with col3:
                        st.metric("🍂 Stressed", f"{health_percent['Stressed']}%",
                                 delta=f"{health_percent['Stressed'] - 33.3:.1f}% vs avg",
                                 delta_color="inverse")
                
                # Charts
                st.markdown("---")
                st.subheader("📊 Detailed Analysis")
                
                if health_classification['status'] == 'No Vegetation':
                    st.info("📉 **Charts not applicable:** No vegetation detected in this image. The image contains primarily non-vegetation elements such as dead trees, buildings, soil, or other materials.")
                    
                    # Show a simple message instead of charts
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**NDVI Distribution**")
                        st.write("No vegetation areas to analyze")
                    with col2:
                        st.markdown("**Health Distribution**")
                        st.write("No health data available")
                else:
                    # Normal charts for vegetation images
                    fig1, fig2 = create_plots(ndvi_norm, health_percent)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(fig1)
                    
                    with col2:
                        st.pyplot(fig2)
                
                # Technical details in expander
                with st.expander("🔬 Technical Details"):
                    if health_classification['status'] == 'No Vegetation':
                        st.markdown(f"""
                        **Image Analysis Results:**
                        - **Image Dimensions:** {image_array.shape[1]} x {image_array.shape[0]} pixels
                        - **Vegetation Pixels Detected:** {ndvi_stats['vegetation_pixels']:,} out of {ndvi_stats['total_pixels']:,}
                        - **Vegetation Coverage:** {health_classification['vegetation_percentage']:.1f}%
                        - **Primary Content:** Non-vegetation (dead trees, soil, buildings, etc.)
                        
                        **Detection Method:**
                        - Advanced vegetation masking with stress detection
                        - Brown/dead material exclusion algorithms
                        - Minimum green dominance requirements
                        - Classification confidence: {health_classification['confidence']}
                        
                        **Why No Vegetation?**
                        This image was classified as "No Vegetation" because it contains primarily:
                        - Dead or leafless trees (brown/gray woody material)
                        - Soil, rock, or ground surfaces
                        - Buildings or infrastructure
                        - Other non-photosynthetic materials
                        """)
                    else:
                        st.markdown(f"""
                        **Image Dimensions:** {image_array.shape[1]} x {image_array.shape[0]} pixels
                        
                        **Vegetation Analysis:**
                        - **Vegetation Coverage:** {health_classification['vegetation_percentage']:.1f}%
                        - **Vegetation Pixels:** {ndvi_stats['vegetation_pixels']:,} out of {ndvi_stats['total_pixels']:,}
                        - **Classification Confidence:** {health_classification['confidence']}
                        
                        **NDVI Statistics (Vegetation Areas Only):**
                        - Mean: {ndvi_stats['mean']:.3f}
                        - Standard Deviation: {ndvi_stats['std']:.3f}
                        - Min: {ndvi_stats['min']:.3f}
                        - Max: {ndvi_stats['max']:.3f}
                        
                        **Processing Method:**
                        - Advanced NIR simulation with vegetation masking
                        - Stress indicator detection
                        - NDVI = (NIR - Red) / (NIR + Red)
                        - Adaptive thresholds: Healthy > 0.10, Moderate > -0.10
                        """)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Please try with a different image or check if the image is valid.")

if __name__ == "__main__":
    main()