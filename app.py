from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.ndvi import simulate_nir

app = Flask(__name__)

UPLOAD_FOLDER = "static/raw_images"
PROCESSED_FOLDER = "static/processed"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Load image and compute NDVI
    image = cv2.imread(filepath)
    nir = simulate_nir(image)
    ndvi = (nir.astype(float) - image[:, :, 2].astype(float)) / (nir + image[:, :, 2] + 1e-6)
    ndvi_norm = (ndvi + 1) / 2  # map -1..1 to 0..1

    # NDVI color map
    ndvi_colored = cv2.applyColorMap((ndvi_norm*255).astype(np.uint8), cv2.COLORMAP_JET)

    # Create full gradient legend with labels
    legend_height = 256
    legend_width = 50
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    for i in range(legend_height):
        value = 1 - i / (legend_height - 1)
        color = cv2.applyColorMap(np.array([[int(value*255)]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0]
        legend[i, :, :] = color

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(legend, 'Healthy', (5, 20), font, 0.5, (255,255,255), 1)
    cv2.putText(legend, 'Moderate', (5, legend_height//2 + 5), font, 0.5, (255,255,255), 1)
    cv2.putText(legend, 'Stressed', (5, legend_height - 5), font, 0.5, (255,255,255), 1)

    # Overlay legend on NDVI image (top-left corner)
    ndvi_colored[0:legend_height, 0:legend_width] = legend

    # Save NDVI result
    output_path = os.path.join(PROCESSED_FOLDER, "ndvi_result.jpg")
    cv2.imwrite(output_path, ndvi_colored)

    # Pixel-based health percentages
    total_pixels = ndvi_norm.size
    healthy = np.sum(ndvi_norm > 0.55)
    moderate = np.sum((ndvi_norm > 0.35) & (ndvi_norm <= 0.55))
    stressed = np.sum(ndvi_norm <= 0.35)

    health_percent = {
        "Healthy": round(healthy / total_pixels * 100, 1),
        "Moderate": round(moderate / total_pixels * 100, 1),
        "Stressed": round(stressed / total_pixels * 100, 1)
    }

    # NDVI classification
    mean_ndvi = np.mean(ndvi_norm)
    if mean_ndvi > 0.55:
        status = "Healthy 🌿"
    elif mean_ndvi > 0.35:
        status = "Moderate 🌾"
    else:
        status = "Stressed 🍂"

    # Create NDVI histogram
    plt.figure(figsize=(5,3))
    plt.hist(ndvi_norm.ravel(), bins=50, color='green', alpha=0.7)
    plt.title('NDVI Distribution')
    plt.xlabel('NDVI Value (0-1)')
    plt.ylabel('Pixel Count')
    hist_path = os.path.join(PROCESSED_FOLDER, "ndvi_histogram.png")
    plt.savefig(hist_path)
    plt.close()

    # Create health bar chart
    labels = ['Healthy', 'Moderate', 'Stressed']
    values = [health_percent['Healthy'], health_percent['Moderate'], health_percent['Stressed']]
    colors = ['green', 'yellow', 'red']

    plt.figure(figsize=(5,3))
    plt.bar(labels, values, color=colors)
    plt.title('Tree Health (%)')
    plt.ylabel('Percentage')
    bar_path = os.path.join(PROCESSED_FOLDER, "health_bar.png")
    plt.savefig(bar_path)
    plt.close()

    return render_template("result.html",
                           ndvi_image="data/processed/ndvi_result.jpg",
                           status=status,
                           health_percent=health_percent,
                           hist_image="data/processed/ndvi_histogram.png",
                           bar_image="data/processed/health_bar.png")

if __name__ == "__main__":
    app.run(debug=True)
