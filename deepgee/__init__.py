"""
DeepGEE: Earth Observation with Google Earth Engine and Deep Learning
=====================================================================

A Python package for integrating Google Earth Engine with deep learning
for advanced Earth observation analysis.

Features:
- GEE authentication and initialization
- Data download using geemap
- Spectral indices calculation
- Deep learning model training
- Land cover classification
- Change detection
- And more...
"""

__version__ = "0.1.0"
__copyright__ = "Copyright 2026, Pulakesh Pradhan"
__author__ = "Pulakesh Pradhan"

from .auth import authenticate_gee, initialize_gee
from .data import GEEDataDownloader, SpectralIndices
from .models import LandCoverClassifier, ChangeDetector
from .utils import load_geotiff, save_geotiff, calculate_area_stats

__all__ = [
    'authenticate_gee',
    'initialize_gee',
    'GEEDataDownloader',
    'SpectralIndices',
    'LandCoverClassifier',
    'ChangeDetector',
    'load_geotiff',
    'save_geotiff',
    'calculate_area_stats',
]
