# Quick Start

Get up and running with DeepGEE in just a few minutes!

## Prerequisites

- DeepGEE installed ([Installation Guide](installation.md))
- Google Earth Engine account
- Google Cloud Project ID

## Step 1: Authenticate with GEE

First time only - authenticate with Google Earth Engine:

```python
import deepgee

# Authenticate (opens browser)
deepgee.authenticate_gee()
```

## Step 2: Initialize GEE

In every script, initialize GEE with your project ID:

```python
import deepgee

# Replace with your project ID
deepgee.initialize_gee(project='your-project-id')
```

## Step 3: Download Satellite Data

```python
from deepgee import GEEDataDownloader

# Create downloader
downloader = GEEDataDownloader()

# Define region of interest (bounding box)
roi = [85.0, 20.0, 87.0, 22.0]  # [lon_min, lat_min, lon_max, lat_max]

# Create composite
composite = downloader.create_composite(
    roi=roi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    sensor='landsat8',
    add_indices=True,  # Add spectral indices
    add_elevation=True  # Add elevation data
)

# Download to file
downloader.download_image(
    composite,
    output_path='my_composite.tif',
    roi=roi,
    scale=30  # 30m resolution for Landsat
)
```

## Step 4: Visualize (Optional)

Create an interactive map:

```python
# Visualize on interactive map
Map = downloader.visualize_map(
    composite,
    vis_params={'min': 0, 'max': 0.3, 'bands': ['B5', 'B4', 'B3']},
    name='False Color Composite'
)

# Display map (in Jupyter notebook)
Map
```

## Complete Example

Here's a complete working example:

```python
import deepgee
from deepgee import GEEDataDownloader

# Initialize
deepgee.initialize_gee(project='your-project-id')

# Create downloader
downloader = GEEDataDownloader()

# Define region
roi = [85.0, 20.0, 87.0, 22.0]

# Download data
print("Creating composite...")
composite = downloader.create_composite(
    roi=roi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    sensor='landsat8'
)

print("Downloading...")
downloader.download_image(composite, 'output.tif', roi=roi, scale=30)

print("✓ Done! Image saved to 'output.tif'")
```

## What's Next?

### Learn More

- [GEE Setup Guide](gee-setup.md) - Detailed GEE configuration
- [User Guide](../user-guide/overview.md) - Comprehensive documentation
- [API Reference](../api/data.md) - Detailed API docs

### Try Examples

- [Land Cover Classification](../examples/land-cover.md) - Complete ML workflow
- [Change Detection](../examples/change-detection.md) - Temporal analysis
- [Crop Monitoring](../examples/crop-monitoring.md) - Time series analysis

### Common Tasks

#### Download Different Sensors

```python
# Sentinel-2
composite = downloader.create_composite(
    roi=roi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    sensor='sentinel2',  # Change sensor
    add_indices=True
)
```

#### Calculate Specific Indices

```python
from deepgee import SpectralIndices

# Add NDVI only
composite_ndvi = SpectralIndices.add_ndvi(composite, sensor='landsat8')

# Add all indices
composite_all = SpectralIndices.add_all_indices(composite, sensor='landsat8')
```

#### Extract Training Samples

```python
# Extract samples from points
samples = downloader.extract_training_samples(
    composite,
    points=training_points,  # ee.FeatureCollection
    scale=30
)
```

## Tips

!!! tip "Project ID"
    Always use your Google Cloud Project ID when initializing GEE.

!!! tip "Scale"
    Use 30m for Landsat, 10m for Sentinel-2.

!!! tip "ROI Format"
    ROI should be `[lon_min, lat_min, lon_max, lat_max]`.

!!! warning "Authentication"
    You only need to authenticate once. After that, just initialize with your project ID.

## Getting Help

- [User Guide](../user-guide/overview.md) - Detailed documentation
- [Examples](../examples/land-cover.md) - Complete workflows
- [GitHub Issues](https://github.com/pulakeshpradhan/deepgee/issues) - Report problems
- Email: <pulakesh.mid@gmail.com>
