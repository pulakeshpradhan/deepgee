"""
Example: Quick Start with DeepGEE

This is a minimal example to get started quickly with DeepGEE.
"""

import deepgee

# =============================================================================
# 1. AUTHENTICATE AND INITIALIZE
# =============================================================================

# First time only - authenticate
# deepgee.authenticate_gee()

# Initialize with your project
deepgee.initialize_gee(project='your-project-id')

# =============================================================================
# 2. DOWNLOAD DATA
# =============================================================================

from deepgee import GEEDataDownloader

# Create downloader
downloader = GEEDataDownloader()

# Define region (example coordinates)
roi = [85.0, 20.0, 87.0, 22.0]  # [lon_min, lat_min, lon_max, lat_max]

# Create and download composite
print("Creating composite...")
composite = downloader.create_composite(
    roi=roi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    sensor='landsat8'
)

print("Downloading...")
downloader.download_image(composite, 'my_composite.tif', roi=roi, scale=30)

# =============================================================================
# 3. VISUALIZE (OPTIONAL)
# =============================================================================

# Create interactive map
Map = downloader.visualize_map(
    composite,
    vis_params={'min': 0, 'max': 0.3, 'bands': ['B5', 'B4', 'B3']},
    name='False Color Composite'
)

# Display map (in Jupyter notebook)
# Map

print("\n✓ Done! Image saved to 'my_composite.tif'")
