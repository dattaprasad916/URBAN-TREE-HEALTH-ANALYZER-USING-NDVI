# 🌳 Urban Tree Health Monitor

A Streamlit-based web application for analyzing tree and vegetation health using NDVI (Normalized Difference Vegetation Index) calculations from RGB images.

## 🎯 Features

- **Interactive Web Interface**: Easy-to-use Streamlit interface for image uploads
- **NDVI Analysis**: Simulates near-infrared data from RGB images to calculate vegetation health
- **Health Classification**: Categorizes vegetation into Healthy, Moderate, and Stressed categories
- **Visual Analysis**: 
  - NDVI heatmaps with color-coded legends
  - Distribution histograms
  - Health percentage bar charts
- **Detailed Statistics**: Comprehensive NDVI statistics and technical details
- **Real-time Processing**: Instant analysis and visualization of uploaded images

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "C:\Users\acer\Desktop\tree health"
   ```

2. **Activate the virtual environment (if not already active):**
   ```bash
   .\venv\Scripts\Activate.ps1
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Streamlit application:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your web browser** and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

3. **Upload an image** of trees or vegetation using the file uploader

4. **View the analysis results** including:
   - NDVI heatmap visualization
   - Overall health status
   - Detailed health percentages
   - Statistical analysis and charts

## 📊 Understanding NDVI

**NDVI (Normalized Difference Vegetation Index)** is a measure of vegetation health:

- **Healthy** (NDVI > 0.55): Dense, healthy vegetation 🌿
- **Moderate** (0.35 < NDVI ≤ 0.55): Moderate vegetation health 🌾
- **Stressed** (NDVI ≤ 0.35): Sparse or stressed vegetation 🍂

### How it Works

1. **NIR Simulation**: The application simulates near-infrared data using: `NIR = 0.6 × Red + 0.4 × Green`
2. **NDVI Calculation**: Computes NDVI using: `NDVI = (NIR - Red) / (NIR + Red)`
3. **Visualization**: Maps NDVI values to colors for easy interpretation

## 📁 Project Structure

```
tree health/
├── streamlit_app.py      # Main Streamlit application
├── app.py               # Original Flask application (legacy)
├── web_app.py          # Alternative Flask implementation (legacy)
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── utils/
│   └── ndvi.py        # NDVI calculation utilities
├── templates/         # HTML templates (for Flask - not used in Streamlit)
│   ├── index.html
│   └── result.html
├── static/           # Static files and processed images
│   ├── raw_images/   # Uploaded images
│   └── processed/    # Generated outputs
├── models/          # Directory for ML models (future use)
└── venv/           # Python virtual environment
```

## 🔧 Technical Details

### Dependencies

- **Streamlit**: Web application framework
- **OpenCV**: Image processing and computer vision
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **Pillow**: Image handling
- **Flask**: Web framework (kept for compatibility)

### Key Functions

- `simulate_nir()`: Simulates NIR channel from RGB image
- `compute_ndvi()`: Calculates NDVI from RGB and NIR data
- `create_ndvi_visualization()`: Generates color-coded NDVI heatmaps
- `calculate_health_stats()`: Computes health percentages and classifications

## 🎨 Features Overview

### Main Interface
- Clean, intuitive file upload interface
- Side panel with NDVI information and color legend
- Responsive design that works on different screen sizes

### Analysis Results
- **Original vs NDVI**: Side-by-side comparison of original and analyzed images
- **Health Status**: Color-coded overall health assessment
- **Statistics**: Mean NDVI, pixel counts, and distribution metrics
- **Interactive Charts**: Histogram and bar chart visualizations
- **Technical Details**: Expandable section with detailed statistics

### Image Processing
- Supports PNG, JPG, JPEG, and WEBP formats
- Automatic color space conversion
- Error handling for invalid images
- Real-time processing with progress indicators

## 🚫 Migration from Flask

This project has been migrated from Flask to Streamlit for better user experience:

- **Before**: Flask with HTML templates and static file serving
- **After**: Streamlit with interactive widgets and real-time updates

The original Flask files (`app.py` and `web_app.py`) are preserved for reference.

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
2. **File Upload Issues**: Ensure uploaded images are valid and in supported formats
3. **Display Issues**: Try refreshing the browser or restarting the Streamlit server

### Performance Tips

- For large images, processing may take longer
- Close unused browser tabs to free up memory
- Use images with reasonable resolution (< 2MB recommended)

## 📈 Future Enhancements

- Integration of actual multispectral image data
- Machine learning models for disease detection
- Batch processing capabilities
- Historical analysis and comparison features
- Export functionality for reports

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for improvements.

## 📄 License

This project is open source and available under the MIT License.