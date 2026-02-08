"""
Data download and processing module using geemap and GEE
"""

import ee
import geemap
import os
from typing import Optional, List, Dict, Tuple, Union
import numpy as np


class SpectralIndices:
    """
    Calculate common spectral indices for remote sensing.
    """
    
    @staticmethod
    def ndvi(image: ee.Image, nir: str = 'B5', red: str = 'B4') -> ee.Image:
        """Calculate Normalized Difference Vegetation Index."""
        return image.normalizedDifference([nir, red]).rename('NDVI')
    
    @staticmethod
    def evi(image: ee.Image, nir: str = 'B5', red: str = 'B4', blue: str = 'B2') -> ee.Image:
        """Calculate Enhanced Vegetation Index."""
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select(nir),
                'RED': image.select(red),
                'BLUE': image.select(blue)
            }
        ).rename('EVI')
        return evi
    
    @staticmethod
    def ndwi(image: ee.Image, green: str = 'B3', nir: str = 'B5') -> ee.Image:
        """Calculate Normalized Difference Water Index."""
        return image.normalizedDifference([green, nir]).rename('NDWI')
    
    @staticmethod
    def ndbi(image: ee.Image, swir1: str = 'B6', nir: str = 'B5') -> ee.Image:
        """Calculate Normalized Difference Built-up Index."""
        return image.normalizedDifference([swir1, nir]).rename('NDBI')
    
    @staticmethod
    def nbr(image: ee.Image, nir: str = 'B5', swir2: str = 'B7') -> ee.Image:
        """Calculate Normalized Burn Ratio."""
        return image.normalizedDifference([nir, swir2]).rename('NBR')
    
    @staticmethod
    def ndmi(image: ee.Image, nir: str = 'B5', swir1: str = 'B6') -> ee.Image:
        """Calculate Normalized Difference Moisture Index."""
        return image.normalizedDifference([nir, swir1]).rename('NDMI')
    
    @staticmethod
    def ndbai(image: ee.Image, swir1: str = 'B6', swir2: str = 'B7') -> ee.Image:
        """Calculate Normalized Difference Bareness Index."""
        return image.normalizedDifference([swir1, swir2]).rename('NDBaI')
    
    @staticmethod
    def add_all_indices(image: ee.Image, sensor: str = 'landsat8') -> ee.Image:
        """
        Add all spectral indices to an image.
        
        Parameters:
        -----------
        image : ee.Image
            Input image
        sensor : str
            Sensor type: 'landsat8', 'landsat9', 'sentinel2'
        
        Returns:
        --------
        ee.Image : Image with all indices added as bands
        """
        if sensor in ['landsat8', 'landsat9']:
            indices = ee.Image([
                SpectralIndices.ndvi(image, 'B5', 'B4'),
                SpectralIndices.evi(image, 'B5', 'B4', 'B2'),
                SpectralIndices.ndwi(image, 'B3', 'B5'),
                SpectralIndices.ndbi(image, 'B6', 'B5'),
                SpectralIndices.nbr(image, 'B5', 'B7'),
                SpectralIndices.ndmi(image, 'B5', 'B6'),
                SpectralIndices.ndbai(image, 'B6', 'B7')
            ])
        elif sensor == 'sentinel2':
            indices = ee.Image([
                SpectralIndices.ndvi(image, 'B8', 'B4'),
                SpectralIndices.ndwi(image, 'B3', 'B8'),
                SpectralIndices.ndbi(image, 'B11', 'B8'),
                SpectralIndices.nbr(image, 'B8', 'B12'),
                SpectralIndices.ndmi(image, 'B8', 'B11')
            ])
        else:
            raise ValueError(f"Unknown sensor: {sensor}")
        
        return image.addBands(indices)


class GEEDataDownloader:
    """
    Download and process data from Google Earth Engine using geemap.
    """
    
    def __init__(self, project: Optional[str] = None):
        """
        Initialize the data downloader.
        
        Parameters:
        -----------
        project : str, optional
            GEE project ID
        """
        self.project = project
        if not self._check_ee_initialized():
            print("Warning: GEE not initialized. Call deepgee.initialize_gee() first.")
    
    @staticmethod
    def _check_ee_initialized() -> bool:
        """Check if Earth Engine is initialized."""
        try:
            ee.Image(0).getInfo()
            return True
        except:
            return False
    
    @staticmethod
    def cloud_mask_landsat89(image: ee.Image) -> ee.Image:
        """
        Apply cloud mask to Landsat 8/9 Collection 2 Level 2 images.
        
        Parameters:
        -----------
        image : ee.Image
            Landsat 8 or 9 image
        
        Returns:
        --------
        ee.Image : Cloud-masked image
        """
        qa = image.select('QA_PIXEL')
        dilated = 1 << 1
        cirrus = 1 << 2
        cloud = 1 << 3
        shadow = 1 << 4
        
        mask = qa.bitwiseAnd(dilated).eq(0) \
            .And(qa.bitwiseAnd(cirrus).eq(0)) \
            .And(qa.bitwiseAnd(cloud).eq(0)) \
            .And(qa.bitwiseAnd(shadow).eq(0))
        
        return image.select(['SR_B.*'], ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']) \
            .updateMask(mask) \
            .multiply(0.0000275) \
            .add(-0.2)
    
    @staticmethod
    def cloud_mask_sentinel2(image: ee.Image) -> ee.Image:
        """
        Apply cloud mask to Sentinel-2 images.
        
        Parameters:
        -----------
        image : ee.Image
            Sentinel-2 image
        
        Returns:
        --------
        ee.Image : Cloud-masked image
        """
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0) \
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        
        return image.updateMask(mask) \
            .divide(10000) \
            .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']) \
            .rename(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'])
    
    def create_composite(
        self,
        roi: Union[ee.Geometry, List[float]],
        start_date: str,
        end_date: str,
        sensor: str = 'landsat8',
        add_indices: bool = True,
        add_elevation: bool = True
    ) -> ee.Image:
        """
        Create a cloud-free composite image.
        
        Parameters:
        -----------
        roi : ee.Geometry or list
            Region of interest (geometry or [lon_min, lat_min, lon_max, lat_max])
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        sensor : str
            Sensor: 'landsat8', 'landsat9', 'sentinel2'
        add_indices : bool
            Add spectral indices
        add_elevation : bool
            Add elevation from SRTM
        
        Returns:
        --------
        ee.Image : Composite image
        
        Examples:
        ---------
        >>> downloader = GEEDataDownloader()
        >>> roi = [85.0, 20.0, 87.0, 22.0]
        >>> composite = downloader.create_composite(
        ...     roi, '2023-01-01', '2023-12-31', sensor='landsat8'
        ... )
        """
        # Convert ROI to geometry if needed
        if isinstance(roi, list):
            roi = ee.Geometry.Rectangle(roi)
        
        # Select collection and cloud mask function
        if sensor == 'landsat8':
            collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            cloud_mask = self.cloud_mask_landsat89
        elif sensor == 'landsat9':
            collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
            cloud_mask = self.cloud_mask_landsat89
        elif sensor == 'sentinel2':
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            cloud_mask = self.cloud_mask_sentinel2
        else:
            raise ValueError(f"Unknown sensor: {sensor}")
        
        # Create composite
        composite = collection \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .map(cloud_mask) \
            .median() \
            .clip(roi)
        
        # Add spectral indices
        if add_indices:
            composite = SpectralIndices.add_all_indices(composite, sensor)
        
        # Add elevation
        if add_elevation:
            srtm = ee.Image('USGS/SRTMGL1_003').select('elevation')
            composite = composite.addBands(srtm.clip(roi))
        
        return composite
    
    def download_image(
        self,
        image: ee.Image,
        output_path: str,
        roi: Optional[Union[ee.Geometry, List[float]]] = None,
        scale: int = 30,
        crs: str = 'EPSG:4326'
    ) -> str:
        """
        Download an Earth Engine image to local file using geemap.
        
        Parameters:
        -----------
        image : ee.Image
            Image to download
        output_path : str
            Output file path (GeoTIFF)
        roi : ee.Geometry or list, optional
            Region of interest
        scale : int
            Resolution in meters
        crs : str
            Coordinate reference system
        
        Returns:
        --------
        str : Path to downloaded file
        
        Examples:
        ---------
        >>> downloader = GEEDataDownloader()
        >>> composite = downloader.create_composite(...)
        >>> downloader.download_image(
        ...     composite, 'output.tif', scale=30
        ... )
        """
        try:
            if roi is not None:
                if isinstance(roi, list):
                    roi = ee.Geometry.Rectangle(roi)
                
                geemap.ee_export_image(
                    image,
                    filename=output_path,
                    scale=scale,
                    region=roi,
                    crs=crs,
                    file_per_band=False
                )
            else:
                geemap.ee_export_image(
                    image,
                    filename=output_path,
                    scale=scale,
                    crs=crs,
                    file_per_band=False
                )
            
            print(f"✓ Image downloaded to: {output_path}")
            return output_path
        except Exception as e:
            print(f"✗ Download failed: {e}")
            raise
    
    def extract_training_samples(
        self,
        image: ee.Image,
        samples: ee.FeatureCollection,
        scale: int = 30,
        output_path: Optional[str] = None
    ) -> Union[ee.FeatureCollection, str]:
        """
        Extract training samples from an image.
        
        Parameters:
        -----------
        image : ee.Image
            Image to sample
        samples : ee.FeatureCollection
            Training sample polygons/points
        scale : int
            Sampling resolution
        output_path : str, optional
            If provided, save to CSV
        
        Returns:
        --------
        ee.FeatureCollection or str : Extracted samples or path to CSV
        
        Examples:
        ---------
        >>> samples = ee.FeatureCollection([...])
        >>> extracted = downloader.extract_training_samples(
        ...     composite, samples, scale=30, output_path='samples.csv'
        ... )
        """
        extracted = image.sampleRegions(
            collection=samples,
            scale=scale,
            geometries=True
        )
        
        if output_path:
            geemap.ee_export_vector(extracted, output_path)
            print(f"✓ Samples exported to: {output_path}")
            return output_path
        else:
            return extracted
    
    def visualize_map(
        self,
        image: ee.Image,
        vis_params: Optional[Dict] = None,
        name: str = 'Image',
        center: Optional[List[float]] = None,
        zoom: int = 10
    ) -> geemap.Map:
        """
        Create an interactive map with the image.
        
        Parameters:
        -----------
        image : ee.Image
            Image to visualize
        vis_params : dict, optional
            Visualization parameters
        name : str
            Layer name
        center : list, optional
            Map center [lon, lat]
        zoom : int
            Zoom level
        
        Returns:
        --------
        geemap.Map : Interactive map
        
        Examples:
        ---------
        >>> Map = downloader.visualize_map(
        ...     composite,
        ...     vis_params={'min': 0, 'max': 0.3, 'bands': ['B5', 'B4', 'B3']},
        ...     name='False Color'
        ... )
        >>> Map
        """
        Map = geemap.Map()
        
        if center:
            Map.setCenter(center[0], center[1], zoom)
        
        if vis_params is None:
            vis_params = {}
        
        Map.addLayer(image, vis_params, name)
        return Map
