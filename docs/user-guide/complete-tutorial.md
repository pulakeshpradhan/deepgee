# Complete Tutorial: Beginner to Advanced

This comprehensive tutorial will guide you from basic concepts to advanced workflows in Earth observation with DeepGEE.

## 📚 Tutorial Structure

- **Level 1: Beginner** - Basic concepts and simple workflows
- **Level 2: Intermediate** - Custom workflows and analysis
- **Level 3: Advanced** - Large-scale processing and optimization

---

## Level 1: Beginner 🌱

### Tutorial 1.1: Your First Data Download (10 minutes)

#### Objective

Download a satellite image for a small area.

#### Prerequisites

- DeepGEE installed
- Google Earth Engine account
- GEE project ID

#### Step-by-Step

**1. Import and Initialize**

```python
import deepgee

# Initialize GEE (replace with your project ID)
deepgee.initialize_gee(project='your-project-id')
```

**2. Create Downloader**

```python
from deepgee import GEEDataDownloader

downloader = GEEDataDownloader()
```

**3. Define Study Area**

```python
# Small area around a city (example: Bhubaneswar, India)
roi = [85.8, 20.2, 85.9, 20.3]  # [lon_min, lat_min, lon_max, lat_max]
```

**4. Create Composite**

```python
composite = downloader.create_composite(
    roi=roi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    sensor='landsat8',
    add_indices=True,
    add_elevation=True
)
```

**5. Download**

```python
downloader.download_image(
    composite,
    output_path='my_first_image.tif',
    roi=roi,
    scale=30  # 30 meters for Landsat
)

print("✓ Download complete!")
```

#### What You Learned

- ✅ GEE initialization
- ✅ Creating composites
- ✅ Downloading images
- ✅ Understanding ROI format

---

### Tutorial 1.2: Visualize Your Data (15 minutes)

#### Objective

Load and visualize the downloaded image.

#### Code

```python
from deepgee.utils import load_geotiff
import matplotlib.pyplot as plt

# Load image
image, meta = load_geotiff('my_first_image.tif')

print(f"Image shape: {image.shape}")
print(f"Bands: {meta['count']}")

# Visualize RGB (bands 4, 3, 2 for Landsat)
rgb = image[[3, 2, 1], :, :]  # Red, Green, Blue

# Normalize for display
rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min())

# Plot
plt.figure(figsize=(10, 10))
plt.imshow(rgb_normalized.transpose(1, 2, 0))
plt.title('True Color Composite')
plt.axis('off')
plt.savefig('rgb_image.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualization complete!")
```

#### What You Learned

- ✅ Loading GeoTIFF files
- ✅ Understanding band order
- ✅ Creating RGB composites
- ✅ Saving figures

---

### Tutorial 1.3: Simple Classification (30 minutes)

#### Objective

Classify land cover using a simple model.

#### Complete Workflow

```python
import deepgee
import pandas as pd
import numpy as np
from deepgee import GEEDataDownloader, LandCoverClassifier

# 1. Initialize
deepgee.initialize_gee(project='your-project-id')
downloader = GEEDataDownloader()

# 2. Define area and classes
roi = [85.8, 20.2, 85.9, 20.3]

class_names = ['Water', 'Vegetation', 'Urban', 'Bare Soil']
class_values = [0, 1, 2, 3]

# 3. Download data
composite = downloader.create_composite(
    roi, '2023-01-01', '2023-12-31', sensor='landsat8'
)

downloader.download_image(composite, 'data.tif', roi=roi, scale=30)

# 4. Generate training samples
training_points = downloader.generate_training_samples(
    roi=roi,
    class_values=class_values,
    class_names=class_names,
    samples_per_class=200  # Small number for quick testing
)

# 5. Extract features
training_data = downloader.extract_training_samples(
    composite, training_points, scale=30, output_path='samples.csv'
)

# 6. Prepare data
samples = pd.read_csv('samples.csv')
feature_cols = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
                'NDVI', 'EVI', 'NDWI', 'NDBI', 'NBR', 'NDMI', 'NDBaI']

X = samples[feature_cols].values
y = samples['class'].values

# 7. Build and train model
classifier = LandCoverClassifier(n_classes=4, architecture='simple')
classifier.build_model(input_shape=(len(feature_cols),))

X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)

print("Training model...")
history = classifier.train(X_train, y_train, epochs=50, verbose=1)

# 8. Evaluate
results = classifier.evaluate(X_test, y_test, class_names=class_names)
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Kappa: {results['kappa']:.3f}")

print("✓ Classification complete!")
```

#### What You Learned

- ✅ Complete ML workflow
- ✅ Training sample generation
- ✅ Model training
- ✅ Accuracy evaluation

---

## Level 2: Intermediate 🌿

### Tutorial 2.1: Custom Training Data (45 minutes)

#### Objective

Use existing land cover maps for training.

#### Advanced Sampling

```python
import ee

# Use MODIS land cover as reference
modis_lc = ee.Image('MODIS/006/MCD12Q1/2020_01_01').select('LC_Type1')

# Extract stratified samples
training_samples = downloader.create_stratified_samples_from_classification(
    classified_image=modis_lc,
    roi=roi,
    class_band='LC_Type1',
    samples_per_class=500,
    scale=500  # MODIS resolution
)

# Extract features from Landsat
training_data = downloader.extract_training_samples(
    composite,
    samples=training_samples,
    scale=30,
    output_path='modis_training.csv'
)

print("✓ Training data from MODIS extracted!")
```

#### What You Learned

- ✅ Using existing land cover maps
- ✅ Stratified sampling
- ✅ Multi-scale analysis

---

### Tutorial 2.2: Model Comparison (60 minutes)

#### Objective

Compare different model architectures.

#### Code

```python
from deepgee import LandCoverClassifier

architectures = ['simple', 'dense', 'cnn1d']
results_comparison = {}

for arch in architectures:
    print(f"\n{'='*50}")
    print(f"Training {arch.upper()} model...")
    print('='*50)
    
    # Create classifier
    classifier = LandCoverClassifier(n_classes=9, architecture=arch)
    
    # Build model
    if arch == 'cnn1d':
        classifier.build_model(input_shape=(14, 1))
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        history = classifier.train(X_train_cnn, y_train, epochs=100)
        results = classifier.evaluate(X_test_cnn, y_test, class_names)
    else:
        classifier.build_model(input_shape=(14,))
        history = classifier.train(X_train, y_train, epochs=100)
        results = classifier.evaluate(X_test, y_test, class_names)
    
    results_comparison[arch] = results
    
    print(f"{arch.upper()} - Accuracy: {results['accuracy']:.2%}, Kappa: {results['kappa']:.3f}")

# Compare results
print("\n" + "="*50)
print("COMPARISON SUMMARY")
print("="*50)
for arch, res in results_comparison.items():
    print(f"{arch.upper():10s} - Accuracy: {res['accuracy']:.2%}, Kappa: {res['kappa']:.3f}")
```

#### What You Learned

- ✅ Different architectures
- ✅ Model comparison
- ✅ Performance metrics

---

### Tutorial 2.3: Visualization & Analysis (45 minutes)

#### Objective

Create professional visualizations.

#### Complete Visualization Workflow

```python
from deepgee.utils import (
    plot_training_history,
    plot_confusion_matrix,
    plot_classification_map,
    plot_area_distribution,
    calculate_area_stats
)

# 1. Training history
plot_training_history(
    history,
    save_path='outputs/training_history.png'
)

# 2. Confusion matrix
plot_confusion_matrix(
    results['confusion_matrix'],
    class_names,
    save_path='outputs/confusion_matrix.png'
)

# 3. Apply to image
from deepgee.utils import load_geotiff, save_geotiff

image, meta = load_geotiff('data.tif')
n_bands, height, width = image.shape
image_reshaped = image.reshape(n_bands, -1).T

predictions = classifier.predict(image_reshaped)
classified = predictions.reshape(height, width)

save_geotiff(classified, 'outputs/classified.tif', meta)

# 4. Classification map
class_colors = ['#0000FF', '#006400', '#FF0000', '#D2B48C', 
                '#90EE90', '#FFD700', '#00CED1', '#808000', '#FFFFFF']

plot_classification_map(
    classified,
    class_names,
    class_colors,
    save_path='outputs/classification_map.png',
    title='Land Cover Classification 2023'
)

# 5. Area statistics
stats = calculate_area_stats(classified, class_names, pixel_size=30.0)
print("\nArea Statistics:")
print(stats)

stats.to_csv('outputs/area_stats.csv', index=False)

# 6. Area distribution
plot_area_distribution(
    stats,
    class_colors=class_colors,
    save_path='outputs/area_distribution.png'
)

print("✓ All visualizations created!")
```

#### What You Learned

- ✅ Professional plotting
- ✅ Area calculations
- ✅ Results presentation

---

## Level 3: Advanced 🌳

### Tutorial 3.1: Large Area Processing (90 minutes)

#### Objective

Process a large region using tiled download.

#### Workflow

```python
import deepgee
from deepgee import GEEDataDownloader, LandCoverClassifier

# Initialize
deepgee.initialize_gee(project='your-project-id')
downloader = GEEDataDownloader()

# Large area (3° x 3°)
large_roi = [85.0, 20.0, 88.0, 23.0]

# Calculate area
area_deg = (88-85) * (23-20)
print(f"Area: {area_deg} square degrees")

# Create composite
print("Creating composite...")
composite = downloader.create_composite(
    large_roi,
    '2023-01-01',
    '2023-12-31',
    sensor='landsat8',
    add_indices=True,
    add_elevation=True
)

# Download using tiled method
print("Downloading using tiled method...")
downloader.download_image_tiled(
    composite,
    output_path='large_area.tif',
    roi=large_roi,
    scale=30,
    tile_size=0.5,  # 0.5 degree tiles
    temp_dir='./temp_tiles'
)

print("✓ Large area downloaded!")
```

#### What You Learned

- ✅ Tiled download
- ✅ Large area handling
- ✅ Memory management

---

### Tutorial 3.2: Batch Processing (120 minutes)

#### Objective

Process multiple time periods.

#### Time Series Analysis

```python
import os
from datetime import datetime, timedelta

# Define time periods
periods = [
    ('2020-01-01', '2020-12-31'),
    ('2021-01-01', '2021-12-31'),
    ('2022-01-01', '2022-12-31'),
    ('2023-01-01', '2023-12-31')
]

# Create output directory
os.makedirs('time_series', exist_ok=True)

# Process each period
for start, end in periods:
    year = start.split('-')[0]
    print(f"\nProcessing {year}...")
    
    # Create composite
    composite = downloader.create_composite(
        roi, start, end, sensor='landsat8'
    )
    
    # Download
    output_file = f'time_series/composite_{year}.tif'
    downloader.download_image(composite, output_file, roi=roi, scale=30)
    
    # Classify
    image, meta = load_geotiff(output_file)
    n_bands, height, width = image.shape
    image_reshaped = image.reshape(n_bands, -1).T
    
    predictions = classifier.predict(image_reshaped)
    classified = predictions.reshape(height, width)
    
    # Save classification
    save_geotiff(
        classified,
        f'time_series/classified_{year}.tif',
        meta
    )
    
    print(f"✓ {year} complete!")

print("\n✓ Time series processing complete!")
```

#### What You Learned

- ✅ Batch processing
- ✅ Time series analysis
- ✅ Automated workflows

---

### Tutorial 3.3: Custom Model & Optimization (150 minutes)

#### Objective

Build custom model with advanced techniques.

#### Advanced Model

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Custom model architecture
def create_custom_model(input_shape, n_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Feature extraction
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Output
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Create and compile
model = create_custom_model(input_shape=(14,), n_classes=9)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Advanced callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=20,
        restore_best_weights=True,
        monitor='val_loss'
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=10,
        min_lr=1e-7
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    ),
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

print("✓ Custom model trained!")
```

#### What You Learned

- ✅ Custom architectures
- ✅ Advanced callbacks
- ✅ TensorBoard integration
- ✅ Hyperparameter tuning

---

## 🎓 Summary

### Beginner Level

- ✅ Basic data download
- ✅ Simple visualization
- ✅ Basic classification

### Intermediate Level

- ✅ Custom training data
- ✅ Model comparison
- ✅ Professional visualization

### Advanced Level

- ✅ Large area processing
- ✅ Batch processing
- ✅ Custom models

---

## 📚 Next Steps

1. **Practice** - Try each tutorial with your own data
2. **Experiment** - Modify parameters and see results
3. **Combine** - Mix techniques for custom workflows
4. **Share** - Contribute your workflows to the community

---

## 🔗 Resources

- [API Reference](../api/auth.md)
- [Examples](../examples/land-cover.md)
- [Scientific Validation](https://github.com/pulakeshpradhan/deepgee/blob/main/SCIENTIFIC_VALIDATION.md)
- [GitHub](https://github.com/pulakeshpradhan/deepgee)

---

**Happy Earth Observing! 🛰️🌍**
