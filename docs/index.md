# DeepGEE

<div align="center">

![DeepGEE Logo](assets/logo.png){ width="200" }

**Earth Observation with Google Earth Engine and Deep Learning**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/pulakeshpradhan/deepgee/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/pulakeshpradhan/deepgee)](https://github.com/pulakeshpradhan/deepgee)

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/pulakeshpradhan/deepgee){ .md-button }

</div>

---

## 🌟 Overview

**DeepGEE** is a comprehensive Python package that seamlessly integrates **Google Earth Engine (GEE)** with **Deep Learning** for advanced Earth observation analysis. It provides an easy-to-use interface for downloading satellite data, training deep learning models, and performing sophisticated geospatial analysis.

## ✨ Key Features

<div class="grid cards" markdown>

- :material-shield-check:{ .lg .middle } **Easy GEE Authentication**

    ---

    Simple authentication and initialization with Google Earth Engine using conventional methods.

    [:octicons-arrow-right-24: Learn more](user-guide/authentication.md)

- :material-download:{ .lg .middle } **Direct Data Download**

    ---

    Download satellite imagery directly using geemap integration with automatic cloud masking.

    [:octicons-arrow-right-24: Data Download Guide](user-guide/data-download.md)

- :material-brain:{ .lg .middle } **Deep Learning Models**

    ---

    Pre-built TensorFlow/Keras models for land cover classification and change detection.

    [:octicons-arrow-right-24: Model Documentation](user-guide/deep-learning.md)

- :material-chart-line:{ .lg .middle } **Spectral Indices**

    ---

    Automatic calculation of 7+ spectral indices including NDVI, EVI, NDWI, and more.

    [:octicons-arrow-right-24: See all indices](user-guide/data-download.md#spectral-indices)

</div>

## 🚀 Quick Start

### Installation

```bash
pip install git+https://github.com/pulakeshpradhan/deepgee.git
```

### Basic Usage

```python
import deepgee

# Initialize GEE
deepgee.initialize_gee(project='your-project-id')

# Download satellite data
from deepgee import GEEDataDownloader
downloader = GEEDataDownloader()

roi = [85.0, 20.0, 87.0, 22.0]  # [lon_min, lat_min, lon_max, lat_max]
composite = downloader.create_composite(
    roi=roi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    sensor='landsat8'
)

downloader.download_image(composite, 'output.tif', roi=roi, scale=30)
```

[See full quick start guide →](getting-started/quick-start.md){ .md-button }

## 📚 Use Cases

DeepGEE supports multiple Earth observation applications:

### 🌍 Land Cover Classification

Classify satellite imagery into multiple land cover types using deep learning.

```python
from deepgee import LandCoverClassifier

classifier = LandCoverClassifier(n_classes=9, architecture='dense')
classifier.build_model(input_shape=(14,))
classifier.train(X_train, y_train, epochs=50)
```

[View example →](examples/land-cover.md)

### 🔄 Change Detection

Detect temporal changes in land cover and vegetation.

```python
from deepgee import ChangeDetector

detector = ChangeDetector(method='difference')
changes = detector.detect_changes(image1, image2, threshold=0.1)
stats = detector.calculate_change_statistics(changes)
```

[View example →](examples/change-detection.md)

### 🌾 Crop Monitoring

Monitor crop health and detect stress areas using time series analysis.

```python
# Download monthly composites
for month in crop_season:
    composite = downloader.create_composite(roi, start, end)
    # Analyze NDVI trends
```

[View example →](examples/crop-monitoring.md)

## 🛠️ Core Components

### Authentication Module

Simple GEE authentication with multiple methods:

- Notebook authentication
- gcloud authentication  
- Service account support

[Learn more →](user-guide/authentication.md)

### Data Download Module

Comprehensive data handling:

- Cloud masking (Landsat 8/9, Sentinel-2)
- Spectral indices calculation
- Elevation data integration
- Training sample extraction

[Learn more →](user-guide/data-download.md)

### Deep Learning Module

Pre-built models and utilities:

- Land cover classifier (Dense, CNN, Simple)
- Change detector
- Automatic preprocessing
- Model save/load

[Learn more →](user-guide/deep-learning.md)

### Utilities Module

Helper functions for:

- GeoTIFF I/O
- Area statistics
- Professional visualization
- Plotting and mapping

[Learn more →](user-guide/utilities.md)

## 📊 Supported Data

### Sensors

- **Landsat 8** (Collection 2, Level 2)
- **Landsat 9** (Collection 2, Level 2)
- **Sentinel-2** (Harmonized)

### Spectral Indices

- NDVI (Normalized Difference Vegetation Index)
- EVI (Enhanced Vegetation Index)
- NDWI (Normalized Difference Water Index)
- NDBI (Normalized Difference Built-up Index)
- NBR (Normalized Burn Ratio)
- NDMI (Normalized Difference Moisture Index)
- NDBaI (Normalized Difference Bareness Index)

## 🎯 Why DeepGEE?

!!! success "Advantages"
    - **Easy to Use**: Simple API with minimal code required
    - **Comprehensive**: Complete workflow from data download to analysis
    - **Flexible**: Multiple sensors, indices, and model architectures
    - **Well-Documented**: Extensive documentation and examples
    - **Production-Ready**: Tested and ready for real-world applications

## 📖 Documentation

<div class="grid cards" markdown>

- [**Installation Guide**](getting-started/installation.md)

    Step-by-step installation instructions

- [**Quick Start**](getting-started/quick-start.md)

    Get up and running in 5 minutes

- [**User Guide**](user-guide/overview.md)

    Comprehensive usage documentation

- [**API Reference**](api/auth.md)

    Detailed API documentation

- [**Examples**](examples/land-cover.md)

    Complete workflow examples

- [**Contributing**](about/contributing.md)

    How to contribute to DeepGEE

</div>

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](about/contributing.md) for details.

## 📄 License

DeepGEE is licensed under the [MIT License](about/license.md).

## 📧 Contact

- **Author**: Pulakesh Pradhan
- **Email**: <pulakesh.mid@gmail.com>
- **GitHub**: [pulakeshpradhan/deepgee](https://github.com/pulakeshpradhan/deepgee)
- **Issues**: [GitHub Issues](https://github.com/pulakeshpradhan/deepgee/issues)

---

<div align="center">

**Made with ❤️ for the Earth Observation community**

🛰️ 🧠 🌍

</div>
