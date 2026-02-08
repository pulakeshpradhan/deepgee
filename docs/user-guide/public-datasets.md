# Using Public Training Datasets

DeepGEE provides built-in support for using publicly available land cover datasets as training data for deep learning classification. This eliminates the need for manual training data collection and ensures high-quality, scientifically validated training samples.

## 📊 Available Datasets

### 1. ESA WorldCover (10m resolution)

**Best for:** High-resolution Sentinel-2 classification

```python
from deepgee import GEEDataDownloader

downloader = GEEDataDownloader()

# Get training samples from ESA WorldCover
samples = downloader.get_training_from_esa_worldcover(
    roi=[85.0, 20.0, 87.0, 22.0],
    year=2021,  # 2020 or 2021 available
    samples_per_class=500,
    scale=10
)
```

**Class Mapping:**

- 10: Tree cover
- 20: Shrubland
- 30: Grassland
- 40: Cropland
- 50: Built-up
- 60: Bare / sparse vegetation
- 70: Snow and ice
- 80: Permanent water bodies
- 90: Herbaceous wetland
- 95: Mangroves
- 100: Moss and lichen

**Advantages:**

- ✅ High quality (10m resolution)
- ✅ Recent data (2020-2021)
- ✅ 11 detailed classes
- ✅ Global coverage

**Use with:** Sentinel-2 imagery

---

### 2. Dynamic World (10m resolution)

**Best for:** Near real-time classification and time series

```python
# Get training samples from Dynamic World
samples = downloader.get_training_from_dynamic_world(
    roi=[85.0, 20.0, 87.0, 22.0],
    start_date='2023-01-01',
    end_date='2023-12-31',
    samples_per_class=500,
    scale=10
)
```

**Class Mapping:**

- 0: Water
- 1: Trees
- 2: Grass
- 3: Flooded vegetation
- 4: Crops
- 5: Shrub and scrub
- 6: Built area
- 7: Bare ground
- 8: Snow and ice

**Advantages:**

- ✅ Near real-time (updated every 2-5 days)
- ✅ Flexible date ranges
- ✅ 9 classes
- ✅ Based on Sentinel-2

**Use with:** Sentinel-2 imagery

---

### 3. MODIS Land Cover (500m resolution)

**Best for:** Large areas and long time series

```python
# Get training samples from MODIS
samples = downloader.get_training_from_modis_lc(
    roi=[85.0, 20.0, 87.0, 22.0],
    year=2020,
    samples_per_class=300,
    scale=500,
    lc_type=1  # IGBP classification
)
```

**Classification Schemes:**

- `lc_type=1`: IGBP (17 classes) - Default
- `lc_type=2`: UMD (15 classes)
- `lc_type=3`: LAI (8 classes)
- `lc_type=4`: BGC (8 classes)
- `lc_type=5`: PFT (11 classes)

**IGBP Classes (LC_Type1):**

1. Evergreen Needleleaf Forests
2. Evergreen Broadleaf Forests
3. Deciduous Needleleaf Forests
4. Deciduous Broadleaf Forests
5. Mixed Forests
6. Closed Shrublands
7. Open Shrublands
8. Woody Savannas
9. Savannas
10. Grasslands
11. Permanent Wetlands
12. Croplands
13. Urban and Built-up Lands
14. Cropland/Natural Vegetation Mosaics
15. Permanent Snow and Ice
16. Barren
17. Water Bodies

**Advantages:**

- ✅ Long time series (2001-present)
- ✅ Multiple classification schemes
- ✅ Global coverage
- ✅ Well-validated

**Use with:** Landsat or coarser imagery

---

### 4. Copernicus Land Cover (100m resolution)

**Best for:** Medium resolution with detailed forest classes

```python
# Get training samples from Copernicus
samples = downloader.get_training_from_copernicus_lc(
    roi=[85.0, 20.0, 87.0, 22.0],
    year=2019,  # 2015-2019 available
    samples_per_class=300,
    scale=100
)
```

**Class Mapping:**

- 0: Unknown
- 20: Shrubs
- 30: Herbaceous vegetation
- 40: Cultivated and managed vegetation/agriculture
- 50: Urban / built up
- 60: Bare / sparse vegetation
- 70: Snow and ice
- 80: Permanent water bodies
- 90: Herbaceous wetland
- 100: Moss and lichen
- 111-116: Closed forest (various types)
- 121-126: Open forest (various types)
- 200: Oceans, seas

**Advantages:**

- ✅ 23 detailed classes
- ✅ Excellent for forest mapping
- ✅ 100m resolution
- ✅ Global coverage

**Use with:** Landsat imagery

---

## 🚀 Complete Workflow Example

```python
import deepgee
from deepgee import GEEDataDownloader, LandCoverClassifier
import pandas as pd

# 1. Initialize
deepgee.initialize_gee(project='your-project-id')
downloader = GEEDataDownloader()

# 2. Define area
roi = [85.0, 20.0, 87.0, 22.0]

# 3. Get training samples from ESA WorldCover
training_samples = downloader.get_training_from_esa_worldcover(
    roi=roi,
    year=2021,
    samples_per_class=500
)

# 4. Download Sentinel-2 composite
composite = downloader.create_composite(
    roi, '2021-01-01', '2021-12-31', sensor='sentinel2'
)

downloader.download_image(composite, 'data.tif', roi=roi, scale=10)

# 5. Extract features
training_data = downloader.extract_training_samples(
    composite, training_samples, scale=10, output_path='training.csv'
)

# 6. Train model
samples = pd.read_csv('training.csv')
feature_cols = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12',
                'NDVI', 'EVI', 'NDWI', 'NDBI', 'NBR', 'NDMI', 'NDBaI']

X = samples[feature_cols].values
y = samples['class'].values

classifier = LandCoverClassifier(n_classes=11)
classifier.build_model(input_shape=(len(feature_cols),))

X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)
history = classifier.train(X_train, y_train, epochs=100)

# 7. Evaluate
results = classifier.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Kappa: {results['kappa']:.3f}")
```

---

## 📋 Dataset Comparison

| Dataset | Resolution | Classes | Time Range | Best For |
|---------|-----------|---------|------------|----------|
| **ESA WorldCover** | 10m | 11 | 2020-2021 | High-res Sentinel-2 |
| **Dynamic World** | 10m | 9 | 2015-present | Real-time, time series |
| **MODIS** | 500m | 17 | 2001-present | Large areas, long series |
| **Copernicus** | 100m | 23 | 2015-2019 | Forest mapping |

---

## 💡 Best Practices

### Choosing the Right Dataset

**For Sentinel-2 classification:**

```python
# Use ESA WorldCover (best quality)
samples = downloader.get_training_from_esa_worldcover(roi, year=2021)

# OR Dynamic World (for specific dates)
samples = downloader.get_training_from_dynamic_world(
    roi, '2023-01-01', '2023-12-31'
)
```

**For Landsat classification:**

```python
# Use MODIS (for large areas)
samples = downloader.get_training_from_modis_lc(roi, year=2020)

# OR Copernicus (for detailed forest classes)
samples = downloader.get_training_from_copernicus_lc(roi, year=2019)
```

### Sample Size Guidelines

```python
# Small study area (<100 km²)
samples_per_class = 300

# Medium study area (100-1000 km²)
samples_per_class = 500

# Large study area (>1000 km²)
samples_per_class = 1000
```

### Scale Matching

Always match the scale of training data to your imagery:

```python
# Sentinel-2 (10m)
samples = downloader.get_training_from_esa_worldcover(roi, scale=10)
composite = downloader.create_composite(roi, sensor='sentinel2')
downloader.download_image(composite, 'data.tif', scale=10)

# Landsat (30m)
samples = downloader.get_training_from_modis_lc(roi, scale=30)
composite = downloader.create_composite(roi, sensor='landsat8')
downloader.download_image(composite, 'data.tif', scale=30)
```

---

## 🎯 Advantages of Public Datasets

### 1. **No Manual Digitization**

- ✅ Save time - no need to manually create training polygons
- ✅ Immediate availability
- ✅ Global coverage

### 2. **High Quality**

- ✅ Scientifically validated
- ✅ Consistent methodology
- ✅ Expert-reviewed

### 3. **Reproducibility**

- ✅ Same training data for everyone
- ✅ Standardized classes
- ✅ Documented methodology

### 4. **Scalability**

- ✅ Works anywhere in the world
- ✅ Easy to generate thousands of samples
- ✅ Automated workflow

---

## ⚠️ Limitations & Considerations

### Temporal Mismatch

If your imagery date doesn't match the reference dataset:

```python
# Your imagery: 2023
# ESA WorldCover: 2021

# Solution: Use Dynamic World for matching dates
samples = downloader.get_training_from_dynamic_world(
    roi, '2023-01-01', '2023-12-31'
)
```

### Class Mapping

Different datasets have different class schemes:

```python
# ESA WorldCover: 11 classes (10, 20, 30, ...)
# Dynamic World: 9 classes (0, 1, 2, ...)
# MODIS: 17 classes (1, 2, 3, ...)

# Always check class values in your training data!
import pandas as pd
samples = pd.read_csv('training.csv')
print(samples['class'].unique())
```

### Resolution Mismatch

Match dataset resolution to your imagery:

```python
# ✓ Good: Sentinel-2 (10m) + ESA WorldCover (10m)
# ✓ Good: Landsat (30m) + MODIS (500m resampled to 30m)
# ✗ Poor: Sentinel-2 (10m) + MODIS (500m) - too coarse
```

---

## 📚 References

### ESA WorldCover

- Zanaga, D., et al. (2021). ESA WorldCover 10 m 2020 v100.
- <https://doi.org/10.5281/zenodo.5571936>

### Dynamic World

- Brown, C.F., et al. (2022). Dynamic World, Near real-time global 10 m land use land cover mapping.
- Scientific Data, 9(1), 251.

### MODIS Land Cover

- Friedl, M., & Sulla-Menashe, D. (2019). MCD12Q1 MODIS/Terra+Aqua Land Cover Type Yearly L3 Global 500m SIN Grid V006.
- <https://doi.org/10.5067/MODIS/MCD12Q1.006>

### Copernicus Land Cover

- Buchhorn, M., et al. (2020). Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe.
- <https://doi.org/10.5281/zenodo.3939050>

---

## 🎓 Next Steps

1. **Try the example:** Run `examples/public_dataset_training.py`
2. **Compare datasets:** Test different datasets for your area
3. **Optimize:** Adjust sample sizes and parameters
4. **Validate:** Check accuracy against ground truth

---

**Copyright © 2026 Pulakesh Pradhan**
