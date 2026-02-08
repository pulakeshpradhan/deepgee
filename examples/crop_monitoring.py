"""
Example: Crop Monitoring with DeepGEE

This example demonstrates crop health monitoring using NDVI time series:
1. Download multi-temporal imagery
2. Calculate vegetation indices
3. Analyze temporal trends
4. Identify crop stress areas
"""

import deepgee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: INITIALIZE GEE
# =============================================================================

print("Initializing Google Earth Engine...")
deepgee.initialize_gee(project='your-project-id')

# =============================================================================
# STEP 2: DOWNLOAD TIME SERIES DATA
# =============================================================================

print("\nDownloading time series data for crop season...")

from deepgee import GEEDataDownloader

downloader = GEEDataDownloader()

# Define agricultural region
roi = [85.5, 20.5, 86.0, 21.0]  # Adjust to your study area

# Define crop season months (example: Kharif season)
months = [
    ('2023-06-01', '2023-06-30', 'June'),
    ('2023-07-01', '2023-07-31', 'July'),
    ('2023-08-01', '2023-08-31', 'August'),
    ('2023-09-01', '2023-09-30', 'September'),
    ('2023-10-01', '2023-10-31', 'October'),
]

# Download composites for each month
composites = []
for start, end, month_name in months:
    print(f"Processing {month_name}...")
    
    composite = downloader.create_composite(
        roi=roi,
        start_date=start,
        end_date=end,
        sensor='landsat8',
        add_indices=True,
        add_elevation=False
    )
    
    # Download
    output_path = f'data/crop_composite_{month_name.lower()}.tif'
    downloader.download_image(composite, output_path, roi=roi, scale=30)
    
    composites.append((month_name, output_path))

# =============================================================================
# STEP 3: ANALYZE NDVI TIME SERIES
# =============================================================================

print("\nAnalyzing NDVI time series...")

from deepgee import load_geotiff

ndvi_series = []
month_names = []

for month_name, filepath in composites:
    image, meta = load_geotiff(filepath)
    
    # Extract NDVI (adjust band index based on your data)
    ndvi = image[7]  # Assuming NDVI is at index 7
    
    ndvi_series.append(ndvi)
    month_names.append(month_name)
    
    # Calculate statistics
    ndvi_mean = np.nanmean(ndvi)
    ndvi_std = np.nanstd(ndvi)
    
    print(f"{month_name}: Mean NDVI = {ndvi_mean:.3f} ± {ndvi_std:.3f}")

# =============================================================================
# STEP 4: IDENTIFY CROP STRESS AREAS
# =============================================================================

print("\nIdentifying crop stress areas...")

# Calculate mean NDVI across the season
ndvi_stack = np.stack(ndvi_series, axis=0)
ndvi_mean = np.nanmean(ndvi_stack, axis=0)
ndvi_std = np.nanstd(ndvi_stack, axis=0)

# Identify stress areas (low NDVI)
stress_threshold = 0.3  # Adjust based on crop type
stress_areas = (ndvi_mean < stress_threshold).astype(np.uint8)

# Identify healthy areas (high NDVI)
healthy_threshold = 0.6
healthy_areas = (ndvi_mean > healthy_threshold).astype(np.uint8)

# Calculate statistics
from deepgee import ChangeDetector

detector = ChangeDetector()

stress_stats = detector.calculate_change_statistics(stress_areas, pixel_area=900)
healthy_stats = detector.calculate_change_statistics(healthy_areas, pixel_area=900)

print(f"\nStress Areas: {stress_stats['changed_area_km2']:.2f} km²")
print(f"Healthy Areas: {healthy_stats['changed_area_km2']:.2f} km²")

# =============================================================================
# STEP 5: SAVE RESULTS
# =============================================================================

print("\nSaving results...")

from deepgee import save_geotiff

# Save mean NDVI
save_geotiff(ndvi_mean, 'outputs/crop_ndvi_mean.tif', meta)

# Save stress map
save_geotiff(stress_areas, 'outputs/crop_stress_map.tif', meta, nodata=255)

# =============================================================================
# STEP 6: VISUALIZE RESULTS
# =============================================================================

print("\nVisualizing results...")

# Plot NDVI time series
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (ndvi, month_name) in enumerate(zip(ndvi_series, month_names)):
    im = axes[idx].imshow(ndvi, cmap='RdYlGn', vmin=0, vmax=0.8)
    axes[idx].set_title(f'NDVI - {month_name}', fontsize=12, fontweight='bold')
    axes[idx].axis('off')
    plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

# Plot stress map in the last subplot
from matplotlib.colors import ListedColormap
cmap_stress = ListedColormap(['white', 'red'])
im = axes[5].imshow(stress_areas, cmap=cmap_stress, vmin=0, vmax=1)
axes[5].set_title('Crop Stress Areas', fontsize=12, fontweight='bold')
axes[5].axis('off')
cbar = plt.colorbar(im, ax=axes[5], fraction=0.046, pad=0.04, ticks=[0, 1])
cbar.set_ticklabels(['Normal', 'Stress'])

plt.tight_layout()
plt.savefig('outputs/crop_monitoring_timeseries.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot NDVI temporal profile
fig, ax = plt.subplots(figsize=(12, 6))

# Calculate mean NDVI for each month
mean_ndvi_values = [np.nanmean(ndvi) for ndvi in ndvi_series]
std_ndvi_values = [np.nanstd(ndvi) for ndvi in ndvi_series]

ax.plot(month_names, mean_ndvi_values, marker='o', linewidth=2, markersize=10, color='green')
ax.fill_between(
    range(len(month_names)),
    [m - s for m, s in zip(mean_ndvi_values, std_ndvi_values)],
    [m + s for m, s in zip(mean_ndvi_values, std_ndvi_values)],
    alpha=0.3, color='green'
)

ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean NDVI', fontsize=12, fontweight='bold')
ax.set_title('Crop NDVI Temporal Profile', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

# Add threshold lines
ax.axhline(y=stress_threshold, color='red', linestyle='--', label='Stress Threshold', linewidth=2)
ax.axhline(y=healthy_threshold, color='darkgreen', linestyle='--', label='Healthy Threshold', linewidth=2)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('outputs/crop_ndvi_profile.png', dpi=300, bbox_inches='tight')
plt.show()

# Create summary report
summary = {
    'Month': month_names,
    'Mean_NDVI': [f"{v:.3f}" for v in mean_ndvi_values],
    'Std_NDVI': [f"{v:.3f}" for v in std_ndvi_values]
}

summary_df = pd.DataFrame(summary)
summary_df.to_csv('outputs/crop_monitoring_summary.csv', index=False)

print("\n" + "="*60)
print("CROP MONITORING COMPLETE!")
print("="*60)
print("\nOutputs saved to 'outputs/' directory:")
print("  - crop_ndvi_mean.tif")
print("  - crop_stress_map.tif")
print("  - crop_monitoring_timeseries.png")
print("  - crop_ndvi_profile.png")
print("  - crop_monitoring_summary.csv")
print("\nSummary:")
print(f"  Stress Areas: {stress_stats['changed_area_km2']:.2f} km²")
print(f"  Healthy Areas: {healthy_stats['changed_area_km2']:.2f} km²")
print("="*60)
