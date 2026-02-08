# DeepGEE: Earth Observation with Google Earth Engine and Deep Learning

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive Python package for integrating **Google Earth Engine (GEE)** with **Deep Learning** for advanced Earth observation analysis.

## 🌟 Features

- **🔐 Easy GEE Authentication**: Simple authentication and initialization
- **📥 Data Download**: Direct download using geemap
- **📊 Spectral Indices**: Calculate NDVI, EVI, NDWI, NDBI, NBR, and more
- **🧠 Deep Learning Models**: Pre-built classifiers for land cover classification
- **🗺️ Change Detection**: Temporal analysis and change detection
- **📈 Visualization**: Built-in plotting and mapping functions
- **🎯 Multiple Use Cases**: Land cover, crop monitoring, disaster response

## 📦 Installation

### Basic Installation

```bash
pip install deepgee
```

### With TensorFlow (for deep learning)

```bash
pip install deepgee[tensorflow]
```

### Development Installation

```bash
git clone https://github.com/your-repo/deepgee.git
cd deepgee
pip install -e .[dev]
```

## 🚀 Quick Start

### 1. Authenticate and Initialize GEE

```python
import deepgee

# Authenticate (first time only)
deepgee.authenticate_gee()

# Initialize with your project
deepgee.initialize_gee(project='your-project-id')
```

### 2. Download Satellite Data

```python
from deepgee import GEEDataDownloader

# Create downloader
downloader = GEEDataDownloader()

# Define region of interest
roi = [85.0, 20.0, 87.0, 22.0]  # [lon_min, lat_min, lon_max, lat_max]

# Create composite
composite = downloader.create_composite(
    roi=roi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    sensor='landsat8',
    add_indices=True,
    add_elevation=True
)

# Download to local file
downloader.download_image(
    composite,
    output_path='composite.tif',
    roi=roi,
    scale=30
)
```

### 3. Train Deep Learning Model

```python
from deepgee import LandCoverClassifier
import pandas as pd

# Load training data
samples = pd.read_csv('training_samples.csv')
X = samples[feature_columns].values
y = samples['class'].values

# Create classifier
classifier = LandCoverClassifier(n_classes=9, architecture='dense')

# Build model
classifier.build_model(input_shape=(14,))

# Prepare data
X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)

# Train
history = classifier.train(X_train, y_train, epochs=50)

# Evaluate
results = classifier.evaluate(X_test, y_test, class_names=class_names)
print(f"Accuracy: {results['accuracy']:.4f}")
```

### 4. Apply to Full Image

```python
from deepgee import load_geotiff, save_geotiff

# Load image
image, meta = load_geotiff('composite.tif')

# Reshape for prediction
n_bands, height, width = image.shape
image_reshaped = image.reshape(n_bands, -1).T

# Predict
predictions = classifier.predict(image_reshaped)
classified = predictions.reshape(height, width)

# Save result
save_geotiff(classified, 'classified.tif', meta, nodata=255)
```

## 📚 Examples

See the `examples/` directory for complete workflows:

- **Land Cover Classification**: `examples/land_cover_classification.py`
- **Change Detection**: `examples/change_detection.py`
- **Crop Monitoring**: `examples/crop_monitoring.py`
- **Disaster Assessment**: `examples/disaster_assessment.py`

## 🛠️ Main Components

### Authentication (`deepgee.auth`)

```python
import deepgee

# Authenticate
deepgee.authenticate_gee()

# Initialize
deepgee.initialize_gee(project='your-project-id')

# Check status
status = deepgee.auth.check_gee_status()
```

### Data Download (`deepgee.data`)

```python
from deepgee import GEEDataDownloader, SpectralIndices

downloader = GEEDataDownloader()

# Create composite
composite = downloader.create_composite(roi, '2023-01-01', '2023-12-31')

# Add spectral indices
composite = SpectralIndices.add_all_indices(composite, sensor='landsat8')

# Download
downloader.download_image(composite, 'output.tif', roi=roi)
```

### Deep Learning Models (`deepgee.models`)

```python
from deepgee import LandCoverClassifier, ChangeDetector

# Land cover classification
classifier = LandCoverClassifier(n_classes=9)
classifier.build_model(input_shape=(14,))
classifier.train(X_train, y_train)

# Change detection
detector = ChangeDetector(method='difference')
changes = detector.detect_changes(image1, image2, threshold=0.2)
```

### Utilities (`deepgee.utils`)

```python
from deepgee import (
    load_geotiff, save_geotiff, calculate_area_stats,
    plot_confusion_matrix, plot_classification_map
)

# Load/save data
image, meta = load_geotiff('input.tif')
save_geotiff(output, 'output.tif', meta)

# Calculate statistics
stats = calculate_area_stats(classified, class_names, pixel_size=30)

# Visualize
plot_classification_map(classified, class_names, class_colors)
```

## 📖 Documentation

Full documentation available at: [https://deepgee.readthedocs.io/](https://deepgee.readthedocs.io/)

## 🎯 Use Cases

### 1. Land Cover Classification

Classify satellite imagery into multiple land cover types using deep learning.

### 2. Crop Monitoring

Monitor crop health and predict yields using time series analysis.

### 3. Change Detection

Detect changes in land cover over time for deforestation, urbanization, etc.

### 4. Disaster Response

Rapid assessment of flood extent, fire damage, or other disasters.

### 5. Urban Planning

Extract building footprints and monitor urban growth.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Earth Engine team for the amazing platform
- geemap developers for the excellent Python package
- TensorFlow/Keras teams for deep learning frameworks

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/your-repo/deepgee/issues)
- **Email**: <deepgee@example.com>
- **Documentation**: [ReadTheDocs](https://deepgee.readthedocs.io/)

## 🌟 Citation

If you use DeepGEE in your research, please cite:

```bibtex
@software{deepgee2024,
  title={DeepGEE: Earth Observation with Google Earth Engine and Deep Learning},
  author={DeepGEE Team},
  year={2024},
  url={https://github.com/your-repo/deepgee}
}
```

---

**Made with ❤️ for the Earth Observation community**
