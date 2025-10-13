from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
from utils.ndvi import compute_ndvi

app = Flask(__name__)

UPLOAD_FOLDER = 'data/raw_images'
PROCESSED_FOLDER = 'data/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400
    file = request.files['image']
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Load and process image
    image = cv2.imread(file_path)
    simulated_nir = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    simulated_nir = cv2.equalizeHist(simulated_nir)
    ndvi = compute_ndvi(image, simulated_nir)

    result_path = os.path.join(PROCESSED_FOLDER, f"ndvi_{filename}")
    cv2.imwrite(result_path, (ndvi * 255).astype(np.uint8))

    return render_template('result.html', image_file=filename, result_file=f"ndvi_{filename}")

@app.route('/data/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
