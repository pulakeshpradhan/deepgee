# DeepGEE Package Installation and Usage Guide

## 📦 Installation

### Option 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
cd deepgee_package

# Install in development mode
pip install -e .

# Or install with TensorFlow support
pip install -e .[tensorflow]
```

### Option 2: Install Dependencies Only

```bash
cd deepgee_package
pip install -r requirements.txt

# For deep learning, also install TensorFlow
pip install tensorflow
```

## 🔐 Google Earth Engine Setup

### 1. Sign Up for GEE

1. Visit [Google Earth Engine](https://earthengine.google.com/)
2. Sign up with your Google account
3. Wait for approval (usually 24-48 hours)

### 2. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Note your project ID (e.g., `my-gee-project-123456`)

### 3. Authenticate (First Time Only)

```python
import deepgee

# This will open a browser for authentication
deepgee.authenticate_gee()
```

### 4. Initialize in Your Scripts

```python
import deepgee

# Use your project ID
deepgee.initialize_gee(project='your-project-id')
```

## 🚀 Quick Start Examples

### Example 1: Download Satellite Data

```python
import deepgee

# Initialize
deepgee.initialize_gee(project='your-project-id')

# Create downloader
from deepgee import GEEDataDownloader
downloader = GEEDataDownloader()

# Define region
roi = [85.0, 20.0, 87.0, 22.0]  # [lon_min, lat_min, lon_max, lat_max]

# Create composite
composite = downloader.create_composite(
    roi=roi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    sensor='landsat8',
    add_indices=True
)

# Download
downloader.download_image(composite, 'output.tif', roi=roi, scale=30)
```

### Example 2: Land Cover Classification

```python
from deepgee import LandCoverClassifier
import pandas as pd

# Load training data
samples = pd.read_csv('samples.csv')
X = samples[feature_cols].values
y = samples['class'].values

# Create and train classifier
classifier = LandCoverClassifier(n_classes=9, architecture='dense')
classifier.build_model(input_shape=(14,))

# Prepare data
X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)

# Train
history = classifier.train(X_train, y_train, epochs=50)

# Evaluate
results = classifier.evaluate(X_test, y_test, class_names=class_names)
print(f"Accuracy: {results['accuracy']:.4f}")

# Save model
classifier.save('model.h5', 'scaler.pkl')
```

### Example 3: Change Detection

```python
from deepgee import ChangeDetector, load_geotiff

# Load images
image1, meta1 = load_geotiff('2020.tif')
image2, meta2 = load_geotiff('2023.tif')

# Detect changes
detector = ChangeDetector(method='difference')
changes = detector.detect_changes(image1[0], image2[0], threshold=0.1)

# Calculate statistics
stats = detector.calculate_change_statistics(changes, pixel_area=900)
print(f"Changed area: {stats['changed_area_km2']:.2f} km²")
```

## 📚 Complete Workflows

See the `examples/` directory for complete workflows:

### 1. Land Cover Classification

**File:** `examples/land_cover_classification.py`

Complete workflow including:

- GEE data download
- Model training
- Full image classification
- Area statistics
- Visualization

**Run:**

```bash
python examples/land_cover_classification.py
```

### 2. Change Detection

**File:** `examples/change_detection.py`

Temporal analysis including:

- Multi-temporal data download
- NDVI change calculation
- Change statistics
- Visualization

**Run:**

```bash
python examples/change_detection.py
```

### 3. Crop Monitoring

**File:** `examples/crop_monitoring.py`

Time series analysis including:

- Monthly composite download
- NDVI time series
- Crop stress detection
- Temporal profiles

**Run:**

```bash
python examples/crop_monitoring.py
```

### 4. Quick Start

**File:** `examples/quick_start.py`

Minimal example for beginners.

**Run:**

```bash
python examples/quick_start.py
```

## 🛠️ Package Structure

```
deepgee/
├── __init__.py          # Main package initialization
├── auth.py              # GEE authentication
├── data.py              # Data download and processing
├── models.py            # Deep learning models
└── utils.py             # Utility functions

examples/
├── land_cover_classification.py
├── change_detection.py
├── crop_monitoring.py
└── quick_start.py
```

## 📖 API Reference

### Authentication

```python
# Authenticate (first time)
deepgee.authenticate_gee(auth_mode='notebook')

# Initialize
deepgee.initialize_gee(project='your-project-id')

# Check status
status = deepgee.auth.check_gee_status()
```

### Data Download

```python
from deepgee import GEEDataDownloader, SpectralIndices

downloader = GEEDataDownloader()

# Create composite
composite = downloader.create_composite(
    roi=[lon_min, lat_min, lon_max, lat_max],
    start_date='YYYY-MM-DD',
    end_date='YYYY-MM-DD',
    sensor='landsat8',  # or 'landsat9', 'sentinel2'
    add_indices=True,
    add_elevation=True
)

# Download
downloader.download_image(image, 'output.tif', roi=roi, scale=30)

# Add spectral indices
composite = SpectralIndices.add_all_indices(composite, sensor='landsat8')
```

### Deep Learning

```python
from deepgee import LandCoverClassifier

# Create classifier
classifier = LandCoverClassifier(
    n_classes=9,
    architecture='dense'  # or 'cnn1d', 'simple'
)

# Build model
classifier.build_model(input_shape=(n_features,))

# Prepare data
X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)

# Train
history = classifier.train(X_train, y_train, epochs=100)

# Evaluate
results = classifier.evaluate(X_test, y_test, class_names=class_names)

# Predict
predictions = classifier.predict(X_new)

# Save/Load
classifier.save('model.h5', 'scaler.pkl')
classifier.load('model.h5', 'scaler.pkl')
```

### Change Detection

```python
from deepgee import ChangeDetector

detector = ChangeDetector(method='difference')  # or 'ratio'

# Detect changes
changes = detector.detect_changes(image1, image2, threshold=0.1)

# Statistics
stats = detector.calculate_change_statistics(changes, pixel_area=900)
```

### Utilities

```python
from deepgee import (
    load_geotiff, save_geotiff, calculate_area_stats,
    plot_training_history, plot_confusion_matrix,
    plot_classification_map, plot_area_distribution
)

# Load/Save
image, meta = load_geotiff('input.tif')
save_geotiff(output, 'output.tif', meta, nodata=255)

# Statistics
stats = calculate_area_stats(classified, class_names, pixel_size=30)

# Visualization
plot_training_history(history, save_path='history.png')
plot_confusion_matrix(cm, class_names, save_path='cm.png')
plot_classification_map(classified, class_names, colors, save_path='map.png')
plot_area_distribution(stats, colors, save_path='dist.png')
```

## 🔧 Troubleshooting

### GEE Authentication Issues

```python
# Try re-authenticating
deepgee.authenticate_gee()

# Check status
status = deepgee.auth.check_gee_status()
print(status)
```

### Memory Issues

For large images, use batch processing:

```python
# Predict in batches
predictions = classifier.predict(X, batch_size=5000)
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Install TensorFlow if needed
pip install tensorflow
```

## 📝 Notes

- **Project ID**: Always required for GEE initialization
- **Scale**: Use 30m for Landsat, 10m for Sentinel-2
- **ROI Format**: [lon_min, lat_min, lon_max, lat_max]
- **Date Format**: 'YYYY-MM-DD'
- **Class Labels**: Should be 0-indexed (0, 1, 2, ...)

## 🎯 Best Practices

1. **Always authenticate before initializing**
2. **Use appropriate scale for your sensor**
3. **Apply cloud masking for optical data**
4. **Normalize features before training**
5. **Save models and scalers for reproducibility**
6. **Use validation data to prevent overfitting**
7. **Calculate area statistics for analysis**

## 📧 Support

For issues and questions:

- GitHub Issues: [github.com/your-repo/deepgee/issues](https://github.com/your-repo/deepgee/issues)
- Email: <deepgee@example.com>

---

**Happy Earth Observing! 🛰️🌍**
