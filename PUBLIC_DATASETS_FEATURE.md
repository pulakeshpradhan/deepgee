# 🎉 DeepGEE - Public Training Datasets Feature

## ✅ **NEW FEATURE: Public Training Datasets**

DeepGEE now supports using publicly available, scientifically validated land cover datasets as training data for deep learning classification!

---

## 🌟 **What's New**

### 4 Public Datasets Integrated

1. **ESA WorldCover** (10m resolution)
   - High-quality global land cover
   - 11 classes
   - Years: 2020, 2021

2. **Dynamic World** (10m resolution)
   - Near real-time land cover
   - 9 classes
   - Updated every 2-5 days

3. **MODIS Land Cover** (500m resolution)
   - Long time series (2001-present)
   - 17 classes (IGBP)
   - Multiple classification schemes

4. **Copernicus Land Cover** (100m resolution)
   - Detailed forest classes
   - 23 classes
   - Years: 2015-2019

---

## 🚀 **Quick Start**

### Using ESA WorldCover

```python
import deepgee
from deepgee import GEEDataDownloader, LandCoverClassifier

# Initialize
deepgee.initialize_gee(project='your-project-id')
downloader = GEEDataDownloader()

# Get training samples from ESA WorldCover
roi = [85.0, 20.0, 87.0, 22.0]
training_samples = downloader.get_training_from_esa_worldcover(
    roi=roi,
    year=2021,
    samples_per_class=500
)

# Download Sentinel-2 data
composite = downloader.create_composite(
    roi, '2021-01-01', '2021-12-31', sensor='sentinel2'
)

# Extract features and train
training_data = downloader.extract_training_samples(
    composite, training_samples, output_path='training.csv'
)

# Train your model!
```

### Using Dynamic World

```python
# Get near real-time training samples
training_samples = downloader.get_training_from_dynamic_world(
    roi=roi,
    start_date='2023-01-01',
    end_date='2023-12-31',
    samples_per_class=500
)
```

### Using MODIS

```python
# Get training samples from MODIS
training_samples = downloader.get_training_from_modis_lc(
    roi=roi,
    year=2020,
    samples_per_class=300,
    lc_type=1  # IGBP classification
)
```

### Using Copernicus

```python
# Get training samples from Copernicus
training_samples = downloader.get_training_from_copernicus_lc(
    roi=roi,
    year=2019,
    samples_per_class=300
)
```

---

## 📊 **Dataset Comparison**

| Dataset | Resolution | Classes | Time Range | Best For |
|---------|-----------|---------|------------|----------|
| **ESA WorldCover** | 10m | 11 | 2020-2021 | High-res Sentinel-2 |
| **Dynamic World** | 10m | 9 | 2015-present | Real-time, time series |
| **MODIS** | 500m | 17 | 2001-present | Large areas, long series |
| **Copernicus** | 100m | 23 | 2015-2019 | Forest mapping |

---

## 💡 **Key Benefits**

### 1. **No Manual Digitization**

✅ Save hours/days of manual work  
✅ No need to create training polygons  
✅ Immediate availability  

### 2. **High Quality**

✅ Scientifically validated  
✅ Consistent methodology  
✅ Expert-reviewed  

### 3. **Global Coverage**

✅ Works anywhere in the world  
✅ Standardized classes  
✅ Reproducible results  

### 4. **Easy to Use**

✅ One function call  
✅ Automatic stratified sampling  
✅ Consistent output format  

---

## 📚 **Complete Example**

See `examples/public_dataset_training.py` for a comprehensive example showing:

- ✅ All 4 datasets
- ✅ Complete workflow
- ✅ Model training
- ✅ Evaluation
- ✅ Visualization
- ✅ Comparison guide

---

## 🎓 **Documentation**

**New Documentation Page:** [Public Training Datasets](https://pulakeshpradhan.github.io/deepgee/user-guide/public-datasets/)

Includes:

- Detailed dataset descriptions
- Class mappings
- Best practices
- Complete workflows
- Scientific references

---

## 🔬 **Scientific References**

### ESA WorldCover

Zanaga, D., et al. (2021). ESA WorldCover 10 m 2020 v100.  
<https://doi.org/10.5281/zenodo.5571936>

### Dynamic World

Brown, C.F., et al. (2022). Dynamic World, Near real-time global 10 m land use land cover mapping.  
Scientific Data, 9(1), 251.

### MODIS Land Cover

Friedl, M., & Sulla-Menashe, D. (2019). MCD12Q1 MODIS/Terra+Aqua Land Cover Type Yearly L3 Global 500m SIN Grid V006.  
<https://doi.org/10.5067/MODIS/MCD12Q1.006>

### Copernicus Land Cover

Buchhorn, M., et al. (2020). Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe.  
<https://doi.org/10.5281/zenodo.3939050>

---

## 📦 **Updated Package Contents**

### New Methods in `deepgee.data.GEEDataDownloader`

1. `get_training_from_esa_worldcover()` - ESA WorldCover samples
2. `get_training_from_dynamic_world()` - Dynamic World samples
3. `get_training_from_modis_lc()` - MODIS Land Cover samples
4. `get_training_from_copernicus_lc()` - Copernicus samples

### New Files

- `examples/public_dataset_training.py` - Complete example
- `docs/user-guide/public-datasets.md` - Documentation

### Updated

- Copyright updated to **2026**
- Package rebuilt with new features
- Documentation site updated

---

## 🌐 **All Resources**

- **📚 Documentation:** <https://pulakeshpradhan.github.io/deepgee/>
- **📖 Public Datasets Guide:** <https://pulakeshpradhan.github.io/deepgee/user-guide/public-datasets/>
- **💻 GitHub:** <https://github.com/pulakeshpradhan/deepgee>
- **📦 Install:** `pip install git+https://github.com/pulakeshpradhan/deepgee.git`

---

## 🎯 **Use Cases**

### 1. Quick Prototyping

```python
# Get training data in seconds
samples = downloader.get_training_from_esa_worldcover(roi, year=2021)
```

### 2. Global Analysis

```python
# Works anywhere in the world
roi_africa = [20.0, -10.0, 30.0, 0.0]
roi_asia = [85.0, 20.0, 95.0, 30.0]
roi_americas = [-80.0, -10.0, -70.0, 0.0]
```

### 3. Time Series

```python
# Use Dynamic World for different dates
for year in [2020, 2021, 2022, 2023]:
    samples = downloader.get_training_from_dynamic_world(
        roi, f'{year}-01-01', f'{year}-12-31'
    )
```

### 4. Multi-Scale Analysis

```python
# High-res: ESA WorldCover + Sentinel-2
# Medium-res: Copernicus + Landsat
# Coarse-res: MODIS + Landsat
```

---

## ✅ **Summary**

### What We Added

✅ **4 Public Datasets** - ESA WorldCover, Dynamic World, MODIS, Copernicus  
✅ **4 New Methods** - Easy access to training data  
✅ **Complete Example** - Full workflow demonstration  
✅ **Comprehensive Documentation** - Detailed guide  
✅ **Scientific References** - Peer-reviewed sources  
✅ **Copyright Updated** - 2026  

### Benefits

✅ **Save Time** - No manual digitization  
✅ **High Quality** - Scientifically validated  
✅ **Global Coverage** - Works anywhere  
✅ **Easy to Use** - One function call  
✅ **Reproducible** - Standardized data  

---

## 🎊 **DeepGEE is Now Even More Powerful!**

**From beginner to advanced, from local to global, DeepGEE makes Earth observation with deep learning accessible to everyone!**

---

**Documentation:** <https://pulakeshpradhan.github.io/deepgee/>  
**Repository:** <https://github.com/pulakeshpradhan/deepgee>  
**Author:** Pulakesh Pradhan  
**Email:** <pulakesh.mid@gmail.com>  
**Version:** 0.1.0  
**Copyright:** © 2026 Pulakesh Pradhan  
**Status:** ✅ Production-Ready with Public Training Datasets!

---

**Made with ❤️ for the Earth Observation community! 🛰️🧠🌍**
