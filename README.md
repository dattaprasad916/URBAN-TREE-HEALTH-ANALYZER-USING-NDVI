# 🌳 Urban Tree Health Monitor

An advanced computer-vision and spectral index platform for evaluating urban forest vitality, canopy density, and foliar stress from standard high-resolution photography.

---

## 🎯 Key Features

- **Multi-Index Spectral Analysis**:
  - **Simulated NDVI**: Near-Infrared simulation from RGB color spaces.
  - **VARI (Visible Atmospherically Resistant Index)**: Peer-reviewed index (*Gitelson et al., 2002*) designed specifically for RGB sensors to estimate green vegetation fraction while reducing atmospheric variability.
  - **GLI (Green Leaf Index)**: Peer-reviewed index (*Louhaichi et al., 2001*) that separates green foliage from dead wood, soil, and urban backgrounds.
- **AI Gatekeeper (YOLOv11-nano)**:
  - Deep learning object detector automatically identifies and rejects non-target photos (vehicles, pedestrians, urban infrastructure).
  - Gracefully falls back to computer-vision morphological analysis if deep-learning weights are offline.
- **Multi-Stage Morphological Vegetation Detector**:
  - Detects green leaves, autumn foliage, bare winter branches, and vertical trunk structures.
- **Interactive Heatmap Overlay**:
  - Adjustable alpha transparency slider (0% to 100%) blends false-colour spectral heatmaps directly over the real tree photograph, allowing arborists to inspect exactly which branches or canopy regions are stressed.
- **Comparative & Seasonal Analysis**:
  - Upload baseline and current images side-by-side to track canopy recovery, seasonal senescence, or post-treatment vigor deltas ($\Delta$).
- **Report & Data Export**:
  - One-click export of structured **JSON Assessment Reports** (for GIS / arborist inventories) and formatted **Text Summaries**.
- **Automated Pytest Suite**:
  - Comprehensive unit tests covering vegetation masking, index calculations, and report generation.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository and navigate into the project directory
git clone https://github.com/your-username/tree-health.git
cd tree-health

# Create and activate a Python virtual environment (Python 3.10+)
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Application

```bash
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501`.

### 3. Running Unit Tests

```bash
pytest tests/ -v
```

---

## 📊 Scientific Remote Sensing Indices

| Index | Formula | Primary Purpose | Reference |
|---|---|---|---|
| **VARI** | $\frac{G - R}{G + R - B}$ | Foliar greenness & canopy fraction from standard RGB cameras | *Gitelson et al. (2002)* |
| **GLI** | $\frac{2G - R - B}{2G + R + B}$ | Discriminating active leaves from bare ground, bark, and dead branches | *Louhaichi et al. (2001)* |
| **Simulated NDVI** | $\frac{\text{NIR}_{\text{sim}} - R}{\text{NIR}_{\text{sim}} + R}$ | Approximation of classic multispectral NIR absorption/scattering | Empirical approximation |

---

## 📁 Modular Architecture

```
tree health/
├── streamlit_app.py          # Modern, responsive multi-tab Streamlit dashboard
├── utils/
│   ├── detection.py          # YOLOv11 object detection & image gating
│   ├── vegetation.py         # Morphological vegetation & tree structure filters
│   ├── indices.py            # Spectral index engine (NDVI, VARI, GLI)
│   ├── ndvi.py               # Core simulated NIR & NDVI math
│   ├── visualization.py      # Heatmaps, custom palettes, alpha overlays, and charts
│   └── export.py             # JSON and text arborist assessment reports
├── tests/
│   ├── __init__.py
│   ├── test_ndvi.py          # Core NDVI calculation & classification tests
│   └── test_new_features.py  # Tests for VARI, GLI, overlays, and export routines
├── requirements.txt          # Pinned, tested dependencies
├── .gitignore                # Excludes runtime outputs, caches, and weight files
└── README.md
```

---

## 🤝 License

This project is open-source under the MIT License.
