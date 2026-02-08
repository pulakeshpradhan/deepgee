"""
Example: Land Cover Classification with DeepGEE

This example demonstrates a complete workflow for land cover classification:
1. Authenticate and initialize GEE
2. Download satellite data
3. Train a deep learning model
4. Apply to full image
5. Visualize results
"""

import deepgee
import numpy as np
import pandas as pd

# =============================================================================
# STEP 1: AUTHENTICATE AND INITIALIZE GEE
# =============================================================================

print("Step 1: Authenticating with Google Earth Engine...")

# Authenticate (first time only - comment out after first run)
# deepgee.authenticate_gee()

# Initialize with your project ID
deepgee.initialize_gee(project='your-project-id')

# Check status
status = deepgee.auth.check_gee_status()
print(f"GEE Status: {status['status']}")

# =============================================================================
# STEP 2: DOWNLOAD SATELLITE DATA
# =============================================================================

print("\nStep 2: Downloading satellite data...")

from deepgee import GEEDataDownloader

# Create downloader
downloader = GEEDataDownloader()

# Define region of interest (example: part of India)
roi = [85.0, 20.0, 87.0, 22.0]  # [lon_min, lat_min, lon_max, lat_max]

# Create composite with spectral indices and elevation
composite = downloader.create_composite(
    roi=roi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    sensor='landsat8',
    add_indices=True,
    add_elevation=True
)

# Download to local file
print("Downloading composite image...")
downloader.download_image(
    composite,
    output_path='data/landsat_composite_2023.tif',
    roi=roi,
    scale=30,
    crs='EPSG:4326'
)

# =============================================================================
# STEP 3: PREPARE TRAINING DATA
# =============================================================================

print("\nStep 3: Preparing training data...")

# Load training samples (CSV exported from GEE)
# The CSV should have columns for each band/index and a 'class' column
samples = pd.read_csv('data/training_samples.csv')

# Define feature columns
feature_cols = [
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',  # Landsat bands
    'NDVI', 'EVI', 'NDWI', 'NDBI', 'NBR', 'NDMI', 'NDBaI',  # Indices
    'elevation'  # Topography
]

# Define class names
class_names = [
    'Water', 'Forest', 'Grassland', 'Cropland',
    'Urban', 'Bareland', 'Wetland', 'Shrubland', 'Snow/Ice'
]

class_colors = [
    '#0000FF',  # Water: Blue
    '#006400',  # Forest: Dark green
    '#90EE90',  # Grassland: Light green
    '#FFD700',  # Cropland: Gold
    '#FF0000',  # Urban: Red
    '#D2B48C',  # Bareland: Tan
    '#00CED1',  # Wetland: Dark turquoise
    '#808000',  # Shrubland: Olive
    '#FFFFFF'   # Snow/Ice: White
]

# Extract features and labels
X = samples[feature_cols].values
y = samples['class'].values  # Should be 0-8 for 9 classes

print(f"Training samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(class_names)}")

# =============================================================================
# STEP 4: BUILD AND TRAIN MODEL
# =============================================================================

print("\nStep 4: Building and training model...")

from deepgee import LandCoverClassifier

# Create classifier
classifier = LandCoverClassifier(
    n_classes=len(class_names),
    architecture='dense'  # Options: 'dense', 'cnn1d', 'simple'
)

# Build model
classifier.build_model(input_shape=(len(feature_cols),))

# Print model summary
classifier.model.summary()

# Prepare data (split and normalize)
X_train, X_test, y_train, y_test = classifier.prepare_data(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train model
print("\nTraining model...")
history = classifier.train(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    verbose=1
)

# =============================================================================
# STEP 5: EVALUATE MODEL
# =============================================================================

print("\nStep 5: Evaluating model...")

# Evaluate on test set
results = classifier.evaluate(X_test, y_test, class_names=class_names)

# Print results
from deepgee.utils import print_model_summary
print_model_summary(results)

# Plot training history
from deepgee.utils import plot_training_history
plot_training_history(history, save_path='outputs/training_history.png')

# Plot confusion matrix
from deepgee.utils import plot_confusion_matrix
plot_confusion_matrix(
    results['confusion_matrix'],
    class_names,
    save_path='outputs/confusion_matrix.png'
)

# Save model
classifier.save('models/land_cover_model.h5', 'models/scaler.pkl')

# =============================================================================
# STEP 6: APPLY TO FULL IMAGE
# =============================================================================

print("\nStep 6: Applying model to full image...")

from deepgee import load_geotiff, save_geotiff

# Load the composite image
image, meta = load_geotiff('data/landsat_composite_2023.tif')

print(f"Image shape: {image.shape}")

# Reshape for prediction
n_bands, height, width = image.shape
image_reshaped = image.reshape(n_bands, -1).T

print(f"Reshaped to: {image_reshaped.shape}")

# Predict (in batches to avoid memory issues)
print("Predicting...")
predictions = classifier.predict(image_reshaped, batch_size=10000)

# Reshape back to image
classified = predictions.reshape(height, width)

print(f"Classification complete! Shape: {classified.shape}")

# Save classified image
save_geotiff(
    classified,
    'outputs/classified_landcover.tif',
    meta,
    nodata=255
)

# =============================================================================
# STEP 7: CALCULATE STATISTICS
# =============================================================================

print("\nStep 7: Calculating area statistics...")

from deepgee import calculate_area_stats

# Calculate area statistics
stats = calculate_area_stats(
    classified,
    class_names,
    pixel_size=30.0  # Landsat pixel size in meters
)

print("\nArea Statistics:")
print(stats.to_string(index=False))

# Save statistics
stats.to_csv('outputs/area_statistics.csv', index=False)

# =============================================================================
# STEP 8: VISUALIZE RESULTS
# =============================================================================

print("\nStep 8: Visualizing results...")

from deepgee.utils import plot_classification_map, plot_area_distribution

# Plot classification map
plot_classification_map(
    classified,
    class_names,
    class_colors,
    save_path='outputs/classification_map.png',
    title='Land Cover Classification 2023'
)

# Plot area distribution
plot_area_distribution(
    stats,
    class_colors=class_colors,
    save_path='outputs/area_distribution.png'
)

print("\n" + "="*60)
print("WORKFLOW COMPLETE!")
print("="*60)
print("\nOutputs saved to 'outputs/' directory:")
print("  - training_history.png")
print("  - confusion_matrix.png")
print("  - classified_landcover.tif")
print("  - area_statistics.csv")
print("  - classification_map.png")
print("  - area_distribution.png")
print("\nModel saved to 'models/' directory:")
print("  - land_cover_model.h5")
print("  - scaler.pkl")
print("="*60)
