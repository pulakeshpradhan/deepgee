"""
Example: Using Public Training Datasets for Deep Learning Classification

This example demonstrates how to use publicly available land cover datasets
(ESA WorldCover, Dynamic World, MODIS, Copernicus) as training data for
deep learning classification.

Author: Pulakesh Pradhan
Copyright: 2026
Email: pulakesh.mid@gmail.com
"""

import deepgee
from deepgee import GEEDataDownloader, LandCoverClassifier
from deepgee.utils import load_geotiff, save_geotiff
import pandas as pd
import numpy as np

# Initialize GEE
deepgee.initialize_gee(project='your-project-id')

# Create downloader
downloader = GEEDataDownloader()

# Define region of interest
roi = [85.0, 20.0, 87.0, 22.0]  # Example: Eastern India

print("="*60)
print("USING PUBLIC DATASETS FOR TRAINING")
print("="*60)

# ============================================================================
# METHOD 1: ESA WorldCover (10m resolution, high quality)
# ============================================================================
print("\n1. ESA WorldCover Training Data")
print("-" * 60)

# Get training samples from ESA WorldCover 2021
worldcover_samples = downloader.get_training_from_esa_worldcover(
    roi=roi,
    year=2021,
    samples_per_class=500,
    scale=10
)

# Download Sentinel-2 composite for classification
print("\nDownloading Sentinel-2 composite...")
s2_composite = downloader.create_composite(
    roi=roi,
    start_date='2021-01-01',
    end_date='2021-12-31',
    sensor='sentinel2',
    add_indices=True,
    add_elevation=True
)

downloader.download_image(
    s2_composite,
    'sentinel2_composite.tif',
    roi=roi,
    scale=10
)

# Extract features from Sentinel-2 using WorldCover samples
print("\nExtracting features...")
worldcover_training = downloader.extract_training_samples(
    s2_composite,
    worldcover_samples,
    scale=10,
    output_path='worldcover_training.csv'
)

# ============================================================================
# METHOD 2: Dynamic World (10m, near real-time)
# ============================================================================
print("\n2. Dynamic World Training Data")
print("-" * 60)

# Get training samples from Dynamic World
dw_samples = downloader.get_training_from_dynamic_world(
    roi=roi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    samples_per_class=500,
    scale=10
)

# Extract features
dw_training = downloader.extract_training_samples(
    s2_composite,
    dw_samples,
    scale=10,
    output_path='dynamic_world_training.csv'
)

# ============================================================================
# METHOD 3: MODIS Land Cover (500m, long time series)
# ============================================================================
print("\n3. MODIS Land Cover Training Data")
print("-" * 60)

# Get training samples from MODIS
modis_samples = downloader.get_training_from_modis_lc(
    roi=roi,
    year=2020,
    samples_per_class=300,
    scale=500,
    lc_type=1  # IGBP classification
)

# Download Landsat composite for MODIS scale
print("\nDownloading Landsat composite...")
landsat_composite = downloader.create_composite(
    roi=roi,
    start_date='2020-01-01',
    end_date='2020-12-31',
    sensor='landsat8',
    add_indices=True,
    add_elevation=True
)

downloader.download_image(
    landsat_composite,
    'landsat_composite.tif',
    roi=roi,
    scale=30
)

# Extract features
modis_training = downloader.extract_training_samples(
    landsat_composite,
    modis_samples,
    scale=30,
    output_path='modis_training.csv'
)

# ============================================================================
# METHOD 4: Copernicus Land Cover (100m)
# ============================================================================
print("\n4. Copernicus Land Cover Training Data")
print("-" * 60)

# Get training samples from Copernicus
copernicus_samples = downloader.get_training_from_copernicus_lc(
    roi=roi,
    year=2019,
    samples_per_class=300,
    scale=100
)

# Extract features
copernicus_training = downloader.extract_training_samples(
    landsat_composite,
    copernicus_samples,
    scale=30,
    output_path='copernicus_training.csv'
)

# ============================================================================
# TRAIN DEEP LEARNING MODEL (Using ESA WorldCover as example)
# ============================================================================
print("\n" + "="*60)
print("TRAINING DEEP LEARNING MODEL")
print("="*60)

# Load training data
samples = pd.read_csv('worldcover_training.csv')

# Define features
feature_cols = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12',  # Sentinel-2 bands
                'NDVI', 'EVI', 'NDWI', 'NDBI', 'NBR', 'NDMI', 'NDBaI', 'elevation']

X = samples[feature_cols].values
y = samples['class'].values

# ESA WorldCover class names
class_names = [
    'Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up',
    'Bare/sparse', 'Snow/ice', 'Water', 'Wetland', 'Mangroves', 'Moss/lichen'
]

# Get unique classes in data
unique_classes = np.unique(y)
n_classes = len(unique_classes)

print(f"\nTraining data:")
print(f"  Samples: {len(X)}")
print(f"  Features: {len(feature_cols)}")
print(f"  Classes: {n_classes}")
print(f"  Class distribution:")
for cls in unique_classes:
    count = np.sum(y == cls)
    print(f"    Class {cls}: {count} samples")

# Create classifier
classifier = LandCoverClassifier(n_classes=n_classes, architecture='dense')

# Build model
classifier.build_model(input_shape=(len(feature_cols),))

# Prepare data
X_train, X_test, y_train, y_test = classifier.prepare_data(X, y, test_size=0.2)

# Train
print("\nTraining model...")
history = classifier.train(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    verbose=1
)

# Evaluate
print("\nEvaluating model...")
results = classifier.evaluate(X_test, y_test, class_names=class_names)

print(f"\nResults:")
print(f"  Overall Accuracy: {results['accuracy']:.2%}")
print(f"  Kappa: {results['kappa']:.3f}")

# ============================================================================
# APPLY TO FULL IMAGE
# ============================================================================
print("\n" + "="*60)
print("APPLYING TO FULL IMAGE")
print("="*60)

# Load image
image, meta = load_geotiff('sentinel2_composite.tif')

# Reshape for prediction
n_bands, height, width = image.shape
image_reshaped = image.reshape(n_bands, -1).T

print(f"\nImage shape: {image.shape}")
print(f"Predicting {height * width} pixels...")

# Predict
predictions = classifier.predict(image_reshaped)
classified = predictions.reshape(height, width)

# Save result
save_geotiff(classified, 'classified_worldcover.tif', meta, nodata=255)

print("✓ Classification complete!")
print(f"  Output: classified_worldcover.tif")

# ============================================================================
# VISUALIZE RESULTS
# ============================================================================
print("\n" + "="*60)
print("VISUALIZING RESULTS")
print("="*60)

from deepgee.utils import (
    plot_confusion_matrix,
    plot_classification_map,
    plot_training_history,
    calculate_area_stats,
    plot_area_distribution
)

# Create output directory
import os
os.makedirs('outputs', exist_ok=True)

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

# 3. Classification map
class_colors = [
    '#006400',  # Tree cover - dark green
    '#FFD700',  # Shrubland - gold
    '#90EE90',  # Grassland - light green
    '#FF8C00',  # Cropland - orange
    '#FF0000',  # Built-up - red
    '#D2B48C',  # Bare/sparse - tan
    '#FFFFFF',  # Snow/ice - white
    '#0000FF',  # Water - blue
    '#00CED1',  # Wetland - cyan
    '#008080',  # Mangroves - teal
    '#808000'   # Moss/lichen - olive
]

plot_classification_map(
    classified,
    class_names,
    class_colors,
    save_path='outputs/classification_map.png',
    title='Land Cover Classification (ESA WorldCover Training)'
)

# 4. Area statistics
stats = calculate_area_stats(classified, class_names, pixel_size=10.0)
print("\nArea Statistics:")
print(stats)

stats.to_csv('outputs/area_stats.csv', index=False)

# 5. Area distribution
plot_area_distribution(
    stats,
    class_colors=class_colors,
    save_path='outputs/area_distribution.png'
)

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print("\nOutputs:")
print("  - classified_worldcover.tif")
print("  - outputs/training_history.png")
print("  - outputs/confusion_matrix.png")
print("  - outputs/classification_map.png")
print("  - outputs/area_stats.csv")
print("  - outputs/area_distribution.png")

# ============================================================================
# COMPARISON: Which dataset to use?
# ============================================================================
print("\n" + "="*60)
print("DATASET COMPARISON GUIDE")
print("="*60)

comparison = """
1. ESA WorldCover (10m)
   ✓ Best for: High-resolution Sentinel-2 classification
   ✓ Pros: High quality, recent (2020-2021), 11 classes
   ✓ Cons: Limited temporal coverage
   ✓ Use with: Sentinel-2 data

2. Dynamic World (10m)
   ✓ Best for: Near real-time classification, time series
   ✓ Pros: Updated every 2-5 days, 9 classes, flexible dates
   ✓ Cons: May have some noise
   ✓ Use with: Sentinel-2 data

3. MODIS Land Cover (500m)
   ✓ Best for: Large areas, long time series (2001-present)
   ✓ Pros: Long record, multiple classification schemes
   ✓ Cons: Coarse resolution
   ✓ Use with: Landsat or coarser data

4. Copernicus Land Cover (100m)
   ✓ Best for: Medium resolution, detailed forest classes
   ✓ Pros: 23 classes, good for forest mapping
   ✓ Cons: Limited years (2015-2019)
   ✓ Use with: Landsat data

Recommendation:
- For Sentinel-2: Use ESA WorldCover or Dynamic World
- For Landsat: Use MODIS or Copernicus
- For time series: Use Dynamic World or MODIS
- For best quality: Use ESA WorldCover
"""

print(comparison)

print("\n✓ Example complete!")
print("Copyright © 2026 Pulakesh Pradhan")
