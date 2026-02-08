"""
Example: Land Cover Classification with DeepGEE (Enhanced with Tiled Download)

This example demonstrates a complete workflow for land cover classification:
1. Authenticate and initialize GEE
2. Download satellite data using TILED download for large areas
3. Generate proper training samples
4. Train a deep learning model
5. Apply to full image
6. Visualize results
"""

import deepgee
import numpy as np
import pandas as pd
import os

# Create output directories
os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

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
# STEP 2: DOWNLOAD SATELLITE DATA USING TILED DOWNLOAD
# =============================================================================

print("\nStep 2: Downloading satellite data using tiled download...")

from deepgee import GEEDataDownloader

# Create downloader
downloader = GEEDataDownloader()

# Define region of interest (example: smaller area for testing)
# For large areas, use tiled download
roi = [85.0, 20.0, 85.5, 20.5]  # Small area for testing
# roi = [85.0, 20.0, 88.0, 23.0]  # Large area - use tiled download

# Create composite with spectral indices and elevation
composite = downloader.create_composite(
    roi=roi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    sensor='landsat8',
    add_indices=True,
    add_elevation=True
)

# Check area size to decide download method
lon_min, lat_min, lon_max, lat_max = roi
area_deg = (lon_max - lon_min) * (lat_max - lat_min)

print(f"Area size: {area_deg:.2f} square degrees")

if area_deg > 1.0:  # Large area - use tiled download
    print("Using TILED download for large area...")
    downloader.download_image_tiled(
        composite,
        output_path='data/landsat_composite_2023.tif',
        roi=roi,
        scale=30,
        tile_size=0.5,  # 0.5 degree tiles
        crs='EPSG:4326'
    )
else:  # Small area - regular download
    print("Using regular download...")
    downloader.download_image(
        composite,
        output_path='data/landsat_composite_2023.tif',
        roi=roi,
        scale=30,
        crs='EPSG:4326'
    )

# =============================================================================
# STEP 3: GENERATE PROPER TRAINING SAMPLES
# =============================================================================

print("\nStep 3: Generating training samples...")

# Define class names and values
class_names = [
    'Water', 'Forest', 'Grassland', 'Cropland',
    'Urban', 'Bareland', 'Wetland', 'Shrubland', 'Snow/Ice'
]

class_values = list(range(len(class_names)))

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

# Option 1: Generate random training samples (for demonstration)
print("Generating random training samples...")
training_points = downloader.generate_training_samples(
    roi=roi,
    class_values=class_values,
    class_names=class_names,
    samples_per_class=300,  # 300 samples per class
    scale=30,
    seed=42
)

# Option 2: Use existing land cover map for training (more realistic)
# Uncomment to use MODIS land cover as reference
"""
print("Generating training samples from MODIS land cover...")
modis_lc = ee.Image('MODIS/006/MCD12Q1/2020_01_01').select('LC_Type1')
training_points = downloader.create_stratified_samples_from_classification(
    classified_image=modis_lc,
    roi=roi,
    class_band='LC_Type1',
    samples_per_class=300,
    scale=500,
    seed=42
)
"""

# Extract training data from composite
print("Extracting features from training points...")
training_data = downloader.extract_training_samples(
    image=composite,
    samples=training_points,
    scale=30,
    output_path='data/training_samples.csv'
)

# Load training samples
samples = pd.read_csv('data/training_samples.csv')

# Define feature columns
feature_cols = [
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',  # Landsat bands
    'NDVI', 'EVI', 'NDWI', 'NDBI', 'NBR', 'NDMI', 'NDBaI',  # Indices
    'elevation'  # Topography
]

# Extract features and labels
X = samples[feature_cols].values
y = samples['class'].values

print(f"Training samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(class_names)}")
print(f"Class distribution:")
for class_val, class_name in enumerate(class_names):
    count = np.sum(y == class_val)
    print(f"  {class_name}: {count} samples")

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
print("\nModel Architecture:")
classifier.model.summary()

# Prepare data (split and normalize)
X_train, X_test, y_train, y_test = classifier.prepare_data(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
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
print("\nData saved to 'data/' directory:")
print("  - landsat_composite_2023.tif")
print("  - training_samples.csv")
print("="*60)
print("\nNOTE: For large areas (>1 square degree), the tiled download")
print("method is automatically used to avoid GEE memory limits.")
print("Tiles are downloaded to temp_tiles/ and merged automatically.")
print("="*60)
