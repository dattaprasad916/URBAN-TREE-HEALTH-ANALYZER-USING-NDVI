"""
Urban Tree Health Monitor — Advanced Canopy Assessment Platform
================================================================
A computer-vision and spectral index application for evaluating urban forest
and vegetation vitality from high-resolution imagery.

Supported Indices:
- Simulated NDVI (Near-Infrared Approximation)
- VARI (Visible Atmospherically Resistant Index, Gitelson et al.)
- GLI (Green Leaf Index, Louhaichi et al.)
"""

import io
import json
import os
import sys
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Import modularized utilities
from utils.detection import load_yolo_model
from utils.vegetation import validate_image
from utils.indices import AVAILABLE_INDICES, compute_vegetation_index
from utils.visualization import (
    COLORMAP_CHOICES,
    create_heatmap,
    create_heatmap_overlay,
    calculate_health_percentages,
    create_plots,
)
from utils.export import generate_json_report, format_report_as_text

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="Urban Tree Health Monitor",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        padding: 1rem 1.2rem;
        border-radius: 0.6rem;
        background-color: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.2);
        margin-bottom: 0.5rem;
    }
    .status-banner {
        padding: 1.2rem;
        border-radius: 0.8rem;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# RESOURCE INITIALIZATION (CACHED)
# ============================================================================

@st.cache_resource
def get_cached_yolo():
    return load_yolo_model("yolo11n.pt")


yolo_model, yolo_err = get_cached_yolo()
yolo_available = yolo_model is not None


# ============================================================================
# CACHED ANALYSIS ENGINE
# ============================================================================

@st.cache_data(show_spinner=False)
def analyze_image_bytes(image_bytes, index_name, colormap_name, overlay_alpha):
    """
    Decodes image bytes, runs index calculation, heatmapping, and statistics.
    Cached by input hash for instant UI interactions.
    """
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb_arr = np.array(image_pil)
    # Convert RGB to OpenCV BGR
    bgr_arr = rgb_arr[:, :, ::-1]

    # Validate image
    is_valid, err_msg, details = validate_image(rgb_arr, yolo_model, yolo_available)
    if not is_valid:
        return None, err_msg

    # Compute selected vegetation index
    raw_masked, norm_masked, veg_mask, stats, classification = compute_vegetation_index(
        bgr_arr, index_type=index_name
    )

    # Health breakdown
    health_percent = calculate_health_percentages(norm_masked, veg_mask)

    # Heatmap & blended overlay
    heatmap_rgb = create_heatmap(norm_masked, colormap_name=colormap_name, add_legend=True)
    overlay_rgb = create_heatmap_overlay(rgb_arr, heatmap_rgb, alpha=overlay_alpha)

    return {
        "rgb_arr": rgb_arr,
        "raw_masked": raw_masked,
        "norm_masked": norm_masked,
        "veg_mask": veg_mask,
        "stats": stats,
        "classification": classification,
        "health_percent": health_percent,
        "heatmap_rgb": heatmap_rgb,
        "overlay_rgb": overlay_rgb,
    }, None


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.title("⚙️ Canopy Analytics")

    # Index selection
    selected_index = st.selectbox(
        "Spectral Vegetation Index",
        options=list(AVAILABLE_INDICES.keys()),
        index=0,
        help="Choose between simulated NIR NDVI and peer-reviewed visible-spectrum RGB indices (VARI, GLI).",
    )

    index_info = AVAILABLE_INDICES[selected_index]
    st.caption(f"**Formula:** `{index_info['formula']}`")
    st.caption(f"_{index_info['description']}_")

    st.markdown("---")

    # Colormap & visualization settings
    st.subheader("🎨 Visual Settings")
    selected_cmap = st.selectbox("Heatmap Palette", options=list(COLORMAP_CHOICES.keys()), index=0)
    overlay_alpha = st.slider(
        "Overlay Transparency (Alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
        help="0.0 shows only original photo; 1.0 shows only false-colour heatmap.",
    )

    st.markdown("---")

    # Diagnostics / YOLO status
    st.subheader("🛡️ AI Gatekeeper")
    if yolo_available:
        st.success("✅ YOLOv11-nano Active (Object Filtering Enabled)")
    else:
        st.warning("⚠️ YOLO Inactive — Falling back to morphology filter")
        if yolo_err:
            st.caption(f"Note: {yolo_err}")


# ============================================================================
# MAIN APPLICATION INTERFACE
# ============================================================================

def main():
    # Ensure working directories
    os.makedirs("static/raw_images", exist_ok=True)
    os.makedirs("static/processed", exist_ok=True)

    st.title("🌳 Urban Tree Health Monitor")
    st.markdown(
        "Automated canopy health diagnostic platform. Upload tree photographs to analyze "
        "foliar vigor, identify abiotic stress, and evaluate urban tree canopy health."
    )

    tab_single, tab_compare, tab_guide = st.tabs(
        ["📊 Single Tree Health Assessment", "🔄 Comparative / Seasonal Analysis", "📖 Scientific Methodology"]
    )

    # ------------------------------------------------------------------------
    # TAB 1: SINGLE TREE ASSESSMENT
    # ------------------------------------------------------------------------
    with tab_single:
        uploaded_file = st.file_uploader(
            "Upload Tree / Canopy Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="single_uploader",
            help="Upload clear photographs of trees, leaf canopies, or urban vegetation.",
        )

        if uploaded_file is not None:
            image_bytes = uploaded_file.read()

            with st.spinner("🔬 Computing spectral vegetation index & canopy metrics..."):
                res, err = analyze_image_bytes(image_bytes, selected_index, selected_cmap, overlay_alpha)

            if err:
                st.error(f"❌ {err}")
                st.info("Tip: Ensure the image contains clear vegetation, branches, or leaf canopy.")
                return

            # Display Visualizations in 3 Columns
            col_orig, col_heat, col_blend = st.columns(3)
            with col_orig:
                st.subheader("📸 Original Photo")
                st.image(res["rgb_arr"], use_container_width=True, caption=uploaded_file.name)

            with col_heat:
                st.subheader(f"🌿 {selected_index.split()[0]} Heatmap")
                st.image(res["heatmap_rgb"], use_container_width=True, caption="False-Color Index")

            with col_blend:
                st.subheader(f"🔍 Canopy Overlay ({int(overlay_alpha*100)}%)")
                st.image(res["overlay_rgb"], use_container_width=True, caption="Heatmap Blended on Real Photo")

            # Assessment Header
            cls = res["classification"]
            stats = res["stats"]
            hp = res["health_percent"]

            st.markdown(
                f"""
                <div class="status-banner" style="border: 2px solid {cls['color']}; background-color: rgba(120,120,120,0.08);">
                    <div style="font-size:1.1rem; font-weight:600; text-transform:uppercase; letter-spacing:1px;">
                        Canopy Health Diagnostic
                    </div>
                    <div style="font-size:2.4rem; font-weight:800; color:{cls['color']}; margin:0.3rem 0;">
                        {cls['icon']} {cls['status']}
                    </div>
                    <div style="font-size:1.05rem; opacity:0.9;">
                        {cls['description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Key Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Overall Status", f"{cls['icon']} {cls['status']}", delta=f"Confidence: {cls['confidence']}")
            with m2:
                st.metric(f"Mean {selected_index.split()[0]}", f"{stats['mean']:.3f}", delta=f"Min: {stats['min']:.2f} | Max: {stats['max']:.2f}")
            with m3:
                st.metric("Canopy Coverage", f"{stats['vegetation_percentage']:.1f}%", delta=f"{stats['vegetation_pixels']:,} px")
            with m4:
                st.metric("Index Std Dev", f"{stats['std']:.3f}", delta="Canopy Uniformity")

            # Health Composition Metrics
            st.markdown("### 🌿 Foliar Health Composition")
            h1, h2, h3 = st.columns(3)
            with h1:
                st.metric("Healthy Foliage", f"{hp['Healthy']}%")
            with h2:
                st.metric("Moderate Foliage", f"{hp['Moderate']}%")
            with h3:
                st.metric("Stressed / Defoliated", f"{hp['Stressed']}%")

            # Distribution and Composition Plots
            st.markdown("---")
            st.markdown("### 📈 Statistical & Canopy Composition Charts")
            fig1, fig2 = create_plots(res["norm_masked"], hp, index_title=selected_index.split()[0])
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.pyplot(fig1)
                plt.close(fig1)
            with p_col2:
                st.pyplot(fig2)
                plt.close(fig2)

            # Professional Report Export
            st.markdown("---")
            st.subheader("📥 Export Assessment Report")
            report_data = generate_json_report(
                uploaded_file.name,
                res["rgb_arr"].shape,
                selected_index,
                stats,
                cls,
                hp,
            )

            report_json_str = json.dumps(report_data, indent=2)
            report_text_str = format_report_as_text(report_data)

            exp1, exp2 = st.columns(2)
            with exp1:
                st.download_button(
                    label="📄 Download Full JSON Report (GIS / Database)",
                    data=report_json_str,
                    file_name=f"tree_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with exp2:
                st.download_button(
                    label="📋 Download Summary Text Report",
                    data=report_text_str,
                    file_name=f"tree_health_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            # Technical Inspection Expander
            with st.expander("🔬 Comprehensive Technical & Metadata Details"):
                st.json(report_data)

        else:
            st.info("👋 Upload a tree or foliage photograph above to begin analysis.")

    # ------------------------------------------------------------------------
    # TAB 2: COMPARATIVE / SEASONAL ANALYSIS
    # ------------------------------------------------------------------------
    with tab_compare:
        st.subheader("🔄 Multi-Temporal / Before-and-After Canopy Comparison")
        st.markdown(
            "Compare two photographs of the same tree or canopy across different seasons, "
            "before/after drought events, or before/after horticultural treatment."
        )

        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            st.markdown("#### Baseline / Image A (e.g. Prior Season)")
            file_a = st.file_uploader("Upload Baseline Image", type=["jpg", "jpeg", "png", "webp"], key="file_a")
        with cmp_col2:
            st.markdown("#### Current / Image B (e.g. Current Condition)")
            file_b = st.file_uploader("Upload Current Image", type=["jpg", "jpeg", "png", "webp"], key="file_b")

        if file_a and file_b:
            bytes_a = file_a.read()
            bytes_b = file_b.read()

            with st.spinner("Analyzing comparative canopy pairs..."):
                res_a, err_a = analyze_image_bytes(bytes_a, selected_index, selected_cmap, overlay_alpha)
                res_b, err_b = analyze_image_bytes(bytes_b, selected_index, selected_cmap, overlay_alpha)

            if err_a:
                st.error(f"Baseline Image Error: {err_a}")
            elif err_b:
                st.error(f"Current Image Error: {err_b}")
            else:
                # Comparative visuals
                c_vis1, c_vis2 = st.columns(2)
                with c_vis1:
                    st.image(res_a["overlay_rgb"], caption=f"Baseline: {res_a['classification']['status']}", use_container_width=True)
                with c_vis2:
                    st.image(res_b["overlay_rgb"], caption=f"Current: {res_b['classification']['status']}", use_container_width=True)

                # Delta statistics
                delta_mean = res_b["stats"]["mean"] - res_a["stats"]["mean"]
                delta_coverage = res_b["stats"]["vegetation_percentage"] - res_a["stats"]["vegetation_percentage"]
                delta_healthy = res_b["health_percent"]["Healthy"] - res_a["health_percent"]["Healthy"]

                st.markdown("### 📊 Delta Assessment (Current vs. Baseline)")
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.metric(
                        f"Mean {selected_index.split()[0]} Change",
                        f"{res_b['stats']['mean']:.3f}",
                        delta=f"{delta_mean:+.3f}",
                    )
                with d2:
                    st.metric(
                        "Canopy Coverage Delta",
                        f"{res_b['stats']['vegetation_percentage']:.1f}%",
                        delta=f"{delta_coverage:+.1f}%",
                    )
                with d3:
                    st.metric(
                        "Healthy Foliage Delta",
                        f"{res_b['health_percent']['Healthy']:.1f}%",
                        delta=f"{delta_healthy:+.1f}%",
                    )

                if delta_mean > 0.03:
                    st.success("📈 **Trend:** Canopy vigor has improved compared to baseline.")
                elif delta_mean < -0.03:
                    st.warning("📉 **Trend:** Canopy vigor has degraded. Investigation recommended.")
                else:
                    st.info("⚖️ **Trend:** Canopy vigor remains stable relative to baseline.")

    # ------------------------------------------------------------------------
    # TAB 3: SCIENTIFIC METHODOLOGY
    # ------------------------------------------------------------------------
    with tab_guide:
        st.subheader("📖 Vegetation Index Methodology & Scientific Guide")
        st.markdown(
            """
            ### Understanding RGB vs. Multispectral Remote Sensing

            Standard satellite and aerial forestry systems use dedicated **Near-Infrared (NIR)** sensors.
            Healthy chlorophyll strongly absorbs blue and red photons for photosynthesis while strongly scattering
            near-infrared energy to avoid leaf overheating.

            When using standard RGB cameras (smartphones, handheld cameras, standard drones), true NIR is not captured.
            To provide accurate assessments, this platform incorporates both **simulated NIR algorithms** and
            **peer-reviewed visible-spectrum vegetation indices**:

            ---

            #### 1. VARI — Visible Atmospherically Resistant Index
            $$\\text{VARI} = \\frac{G - R}{G + R - B}$$
            - **Peer Review:** Gitelson et al. (2002), *Remote Sensing of Environment*.
            - **Best For:** RGB cameras under varying atmospheric and illumination conditions. Directly correlates with green canopy fraction.

            #### 2. GLI — Green Leaf Index
            $$\\text{GLI} = \\frac{2G - R - B}{2G + R + B}$$
            - **Peer Review:** Louhaichi et al. (2001), *International Journal of Remote Sensing*.
            - **Best For:** Discriminating active photosynthetic leaf tissue from background soil, mulch, bark, and dead wood.

            #### 3. Simulated NDVI
            $$\\text{NDVI}_{\\text{sim}} = \\frac{\\text{NIR}_{\\text{sim}} - R}{\\text{NIR}_{\\text{sim}} + R}$$
            - **Approach:** Simulates NIR response using green dominance, organic matter stress masks, and morphological features.
            - **Note:** Correlates with relative canopy greenness; use VARI or GLI for rigorous visible-band comparative studies.
            """
        )


if __name__ == "__main__":
    main()
