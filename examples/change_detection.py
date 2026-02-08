"""
Example: Change Detection with DeepGEE

This example demonstrates temporal change detection:
1. Download images from two time periods
2. Calculate spectral indices
3. Detect changes
4. Analyze change statistics
"""

import deepgee
import numpy as np

# =============================================================================
# STEP 1: INITIALIZE GEE
# =============================================================================

print("Initializing Google Earth Engine...")
deepgee.initialize_gee(project='your-project-id')

# =============================================================================
# STEP 2: DOWNLOAD IMAGES FROM TWO TIME PERIODS
# =============================================================================

print("\nDownloading images from two time periods...")

from deepgee import GEEDataDownloader

downloader = GEEDataDownloader()

# Define region of interest
roi = [85.0, 20.0, 87.0, 22.0]

# Create composite for Time 1 (e.g., 2020)
print("Creating composite for 2020...")
composite_2020 = downloader.create_composite(
    roi=roi,
    start_date='2020-01-01',
    end_date='2020-12-31',
    sensor='landsat8',
    add_indices=True,
    add_elevation=False
)

# Download Time 1
downloader.download_image(
    composite_2020,
    output_path='data/composite_2020.tif',
    roi=roi,
    scale=30
)

# Create composite for Time 2 (e.g., 2023)
print("Creating composite for 2023...")
composite_2023 = downloader.create_composite(
    roi=roi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    sensor='landsat8',
    add_indices=True,
    add_elevation=False
)

# Download Time 2
downloader.download_image(
    composite_2023,
    output_path='data/composite_2023.tif',
    roi=roi,
    scale=30
)

# =============================================================================
# STEP 3: LOAD IMAGES
# =============================================================================

print("\nLoading images...")

from deepgee import load_geotiff

image_2020, meta_2020 = load_geotiff('data/composite_2020.tif')
image_2023, meta_2023 = load_geotiff('data/composite_2023.tif')

print(f"2020 image shape: {image_2020.shape}")
print(f"2023 image shape: {image_2023.shape}")

# =============================================================================
# STEP 4: CALCULATE NDVI FOR BOTH IMAGES
# =============================================================================

print("\nCalculating NDVI...")

# Extract NDVI band (assuming it's band index 7 after B1-B7)
ndvi_2020 = image_2020[7]  # Adjust index based on your data
ndvi_2023 = image_2023[7]

print(f"NDVI 2020 range: {ndvi_2020.min():.3f} to {ndvi_2020.max():.3f}")
print(f"NDVI 2023 range: {ndvi_2023.min():.3f} to {ndvi_2023.max():.3f}")

# =============================================================================
# STEP 5: DETECT CHANGES
# =============================================================================

print("\nDetecting changes...")

from deepgee import ChangeDetector

# Create change detector
detector = ChangeDetector(method='difference')

# Detect changes in NDVI
ndvi_change = detector.detect_changes(ndvi_2020, ndvi_2023)

# Apply threshold to create binary change map
# Positive values = vegetation increase, negative = decrease
threshold = 0.1  # Adjust based on your data

vegetation_gain = (ndvi_change > threshold).astype(np.uint8)
vegetation_loss = (ndvi_change < -threshold).astype(np.uint8)
no_change = (np.abs(ndvi_change) <= threshold).astype(np.uint8)

print(f"Change range: {ndvi_change.min():.3f} to {ndvi_change.max():.3f}")

# =============================================================================
# STEP 6: CALCULATE CHANGE STATISTICS
# =============================================================================

print("\nCalculating change statistics...")

# Vegetation gain statistics
gain_stats = detector.calculate_change_statistics(
    vegetation_gain,
    pixel_area=900.0  # 30m x 30m
)

# Vegetation loss statistics
loss_stats = detector.calculate_change_statistics(
    vegetation_loss,
    pixel_area=900.0
)

print("\nVegetation Gain:")
print(f"  Area: {gain_stats['changed_area_km2']:.2f} km²")
print(f"  Percentage: {gain_stats['change_percentage']:.2f}%")

print("\nVegetation Loss:")
print(f"  Area: {loss_stats['changed_area_km2']:.2f} km²")
print(f"  Percentage: {loss_stats['change_percentage']:.2f}%")

# =============================================================================
# STEP 7: SAVE RESULTS
# =============================================================================

print("\nSaving results...")

from deepgee import save_geotiff

# Save NDVI change map
save_geotiff(
    ndvi_change,
    'outputs/ndvi_change_2020_2023.tif',
    meta_2020
)

# Create combined change map (0=no change, 1=gain, 2=loss)
change_map = no_change * 0 + vegetation_gain * 1 + vegetation_loss * 2

save_geotiff(
    change_map,
    'outputs/change_map_2020_2023.tif',
    meta_2020,
    nodata=255
)

# =============================================================================
# STEP 8: VISUALIZE RESULTS
# =============================================================================

print("\nVisualizing results...")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot NDVI 2020
im1 = axes[0, 0].imshow(ndvi_2020, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
axes[0, 0].set_title('NDVI 2020', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

# Plot NDVI 2023
im2 = axes[0, 1].imshow(ndvi_2023, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
axes[0, 1].set_title('NDVI 2023', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

# Plot NDVI change
im3 = axes[1, 0].imshow(ndvi_change, cmap='RdBu', vmin=-0.3, vmax=0.3)
axes[1, 0].set_title('NDVI Change (2023 - 2020)', fontsize=14, fontweight='bold')
axes[1, 0].axis('off')
cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
cbar3.set_label('NDVI Difference', fontsize=11)

# Plot change map
cmap_change = ListedColormap(['gray', 'green', 'red'])
im4 = axes[1, 1].imshow(change_map, cmap=cmap_change, vmin=0, vmax=2)
axes[1, 1].set_title('Change Classification', fontsize=14, fontweight='bold')
axes[1, 1].axis('off')
cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04, ticks=[0, 1, 2])
cbar4.set_ticklabels(['No Change', 'Vegetation Gain', 'Vegetation Loss'])

plt.tight_layout()
plt.savefig('outputs/change_detection_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Create summary statistics plot
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Vegetation Gain', 'Vegetation Loss', 'No Change']
areas = [
    gain_stats['changed_area_km2'],
    loss_stats['changed_area_km2'],
    (gain_stats['total_pixels'] - gain_stats['changed_pixels'] - loss_stats['changed_pixels']) * 900 / 1e6
]
colors = ['green', 'red', 'gray']

bars = ax.bar(categories, areas, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Area (km²)', fontsize=12, fontweight='bold')
ax.set_title('Change Detection Summary (2020-2023)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, area in zip(bars, areas):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{area:.2f} km²',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/change_statistics.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("CHANGE DETECTION COMPLETE!")
print("="*60)
print("\nOutputs saved to 'outputs/' directory:")
print("  - ndvi_change_2020_2023.tif")
print("  - change_map_2020_2023.tif")
print("  - change_detection_results.png")
print("  - change_statistics.png")
print("="*60)
