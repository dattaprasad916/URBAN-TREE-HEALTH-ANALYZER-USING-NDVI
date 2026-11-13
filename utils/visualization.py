"""
utils/visualization.py
~~~~~~~~~~~~~~~~~~~~~~
Visualization, false-colour heatmapping, alpha blending, and metric plotting
for vegetation health analysis.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

COLORMAP_CHOICES = {
    "JET (Classic Rainbow)": cv2.COLORMAP_JET,
    "TURBO (High Dynamic Range)": cv2.COLORMAP_TURBO,
    "VIRIDIS (Perceptually Uniform)": cv2.COLORMAP_VIRIDIS,
    "SUMMER (Green / Yellow Foliage)": cv2.COLORMAP_SUMMER,
}


def create_heatmap(index_norm, colormap_name="JET (Classic Rainbow)", add_legend=True):
    """
    Renders a normalized [0, 1] vegetation index array as a false-colour heatmap.
    Returns an RGB image (uint8) with optional vertical legend sidebar.
    """
    clean_norm = np.nan_to_num(index_norm, nan=0.0, posinf=1.0, neginf=0.0)
    clean_norm = np.clip(clean_norm, 0.0, 1.0)

    cmap_code = COLORMAP_CHOICES.get(colormap_name, cv2.COLORMAP_JET)
    colored_bgr = cv2.applyColorMap((clean_norm * 255).astype(np.uint8), cmap_code)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    if not add_legend:
        return colored_rgb

    # Generate professional gradient legend
    legend_h = min(256, colored_rgb.shape[0])
    legend_w = 48

    if colored_rgb.shape[0] >= legend_h and colored_rgb.shape[1] >= legend_w + 20:
        legend = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
        for i in range(legend_h):
            val = 1.0 - (i / (legend_h - 1))
            val_byte = int(val * 255)
            color = cv2.applyColorMap(np.array([[val_byte]], dtype=np.uint8), cmap_code)[0, 0]
            legend[i, :] = color

        legend_rgb = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(legend_rgb, "High", (4, 18), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(legend_rgb, "Mid", (4, legend_h // 2 + 4), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(legend_rgb, "Low", (4, legend_h - 8), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Overlay legend in top-left
        colored_rgb[0:legend_h, 0:legend_w] = legend_rgb

    return colored_rgb


def create_heatmap_overlay(original_rgb, heatmap_rgb, alpha=0.55):
    """
    Blends the false-colour heatmap directly over the original photographic image.
    Alpha ranges from 0.0 (original photo only) to 1.0 (heatmap only).
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.01:
        return original_rgb
    if alpha >= 0.99:
        return heatmap_rgb

    # Ensure identical dimensions
    if original_rgb.shape[:2] != heatmap_rgb.shape[:2]:
        heatmap_resized = cv2.resize(heatmap_rgb, (original_rgb.shape[1], original_rgb.shape[0]))
    else:
        heatmap_resized = heatmap_rgb

    blended = cv2.addWeighted(
        heatmap_resized, alpha,
        original_rgb, 1.0 - alpha,
        0
    )
    return blended


def calculate_health_percentages(index_norm, veg_mask, healthy_th=0.55, moderate_th=0.45):
    """
    Computes pixel-level percentage distribution within the detected vegetation mask.
    """
    veg_values = index_norm[veg_mask]
    if len(veg_values) == 0:
        return {"Healthy": 0.0, "Moderate": 0.0, "Stressed": 0.0}

    total = len(veg_values)
    healthy_count = np.sum(veg_values > healthy_th)
    moderate_count = np.sum((veg_values > moderate_th) & (veg_values <= healthy_th))
    stressed_count = np.sum(veg_values <= moderate_th)

    return {
        "Healthy": round(float(healthy_count) / total * 100.0, 1),
        "Moderate": round(float(moderate_count) / total * 100.0, 1),
        "Stressed": round(float(stressed_count) / total * 100.0, 1),
    }


def create_plots(index_norm, health_percent, index_title="NDVI"):
    """
    Creates distribution histogram and health breakdown bar chart.
    Returns (fig1, fig2). Caller is responsible for closing figures after display.
    """
    # 1. Histogram
    fig1, ax1 = plt.subplots(figsize=(7, 3.8), dpi=100)
    ax1.hist(
        index_norm.ravel(),
        bins=45,
        color="#2b8a3e",
        edgecolor="#1864ab",
        alpha=0.75,
    )
    ax1.set_title(f"{index_title} Pixel Distribution", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Normalized Value (0.0 = Stressed/No-Veg, 1.0 = Dense Canopy)")
    ax1.set_ylabel("Pixel Frequency")
    ax1.grid(True, linestyle="--", alpha=0.35)
    fig1.tight_layout()

    # 2. Health Distribution Bar Chart
    fig2, ax2 = plt.subplots(figsize=(7, 3.8), dpi=100)
    categories = ["Healthy", "Moderate", "Stressed"]
    values = [
        health_percent.get("Healthy", 0.0),
        health_percent.get("Moderate", 0.0),
        health_percent.get("Stressed", 0.0),
    ]
    colors = ["#2f9e44", "#f59f00", "#e03131"]

    bars = ax2.bar(categories, values, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    ax2.set_title("Canopy Health Composition (%)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Vegetation Area (%)")
    ax2.set_ylim(0, max(max(values) + 12, 100))
    ax2.grid(True, linestyle="--", alpha=0.35, axis="y")

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    fig2.tight_layout()
    return fig1, fig2
