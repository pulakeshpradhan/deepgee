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
    
    def download_image_tiled(
        self,
        image: ee.Image,
        output_path: str,
        roi: Union[ee.Geometry, List[float]],
        scale: int = 30,
        crs: str = 'EPSG:4326',
        tile_size: float = 0.5,
        temp_dir: Optional[str] = None
    ) -> str:
        """
        Download large images using tiled approach and merge.
        
        This method splits large areas into tiles, downloads each tile,
        and merges them into a single GeoTIFF. Useful for avoiding
        GEE memory limits.
        
        Parameters:
        -----------
        image : ee.Image
            Image to download
        output_path : str
            Output file path (GeoTIFF)
        roi : ee.Geometry or list
            Region of interest
        scale : int
            Resolution in meters
        crs : str
            Coordinate reference system
        tile_size : float
            Tile size in degrees (default: 0.5 degrees)
        temp_dir : str, optional
            Temporary directory for tiles (default: './temp_tiles')
        
        Returns:
        --------
        str : Path to merged file
        
        Examples:
        ---------
        >>> downloader = GEEDataDownloader()
        >>> composite = downloader.create_composite(...)
        >>> downloader.download_image_tiled(
        ...     composite, 'large_area.tif', roi=[85, 20, 88, 23],
        ...     scale=30, tile_size=0.5
        ... )
        """
        import rasterio
        from rasterio.merge import merge
        import shutil
        
        # Convert ROI to list if needed
        if isinstance(roi, ee.Geometry):
            bounds = roi.bounds().getInfo()['coordinates'][0]
            lon_min = min([p[0] for p in bounds])
            lat_min = min([p[1] for p in bounds])
            lon_max = max([p[0] for p in bounds])
            lat_max = max([p[1] for p in bounds])
            roi = [lon_min, lat_min, lon_max, lat_max]
        
        # Create temp directory
        if temp_dir is None:
            temp_dir = './temp_tiles'
        os.makedirs(temp_dir, exist_ok=True)
        
        lon_min, lat_min, lon_max, lat_max = roi
        
        # Calculate tiles
        tiles = []
        tile_files = []
        
        lat = lat_min
        tile_idx = 0
        
        print(f"Downloading large area in tiles (tile size: {tile_size}°)...")
        
        while lat < lat_max:
            lon = lon_min
            while lon < lon_max:
                # Define tile bounds
                tile_lon_max = min(lon + tile_size, lon_max)
                tile_lat_max = min(lat + tile_size, lat_max)
                tile_roi = [lon, lat, tile_lon_max, tile_lat_max]
                tile_geom = ee.Geometry.Rectangle(tile_roi)
                
                # Download tile
                tile_path = os.path.join(temp_dir, f'tile_{tile_idx}.tif')
                
                try:
                    print(f"  Downloading tile {tile_idx + 1} [{lon:.2f}, {lat:.2f}, {tile_lon_max:.2f}, {tile_lat_max:.2f}]...")
                    geemap.ee_export_image(
                        image,
                        filename=tile_path,
                        scale=scale,
                        region=tile_geom,
                        crs=crs,
                        file_per_band=False
                    )
                    tile_files.append(tile_path)
                    tile_idx += 1
                except Exception as e:
                    print(f"  Warning: Tile {tile_idx} failed: {e}")
                
                lon += tile_size
            lat += tile_size
        
        print(f"✓ Downloaded {len(tile_files)} tiles")
        
        # Merge tiles
        print("Merging tiles...")
        src_files_to_mosaic = []
        
        for tile_file in tile_files:
            src = rasterio.open(tile_file)
            src_files_to_mosaic.append(src)
        
        mosaic, out_trans = merge(src_files_to_mosaic)
        
        # Get metadata from first tile
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })
        
        # Write merged file
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        # Close all source files
        for src in src_files_to_mosaic:
            src.close()
        
        # Clean up temp directory
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        
        print(f"✓ Merged image saved to: {output_path}")
        return output_path
    
    def generate_training_samples(
        self,
        roi: Union[ee.Geometry, List[float]],
        class_values: List[int],
        class_names: List[str],
        samples_per_class: int = 500,
        scale: int = 30,
        seed: int = 42
    ) -> ee.FeatureCollection:
        """
        Generate stratified random training samples for land cover classification.
        
        This method creates random points within the ROI and assigns class labels
        for training. For real applications, replace this with actual ground truth data.
        
        Parameters:
        -----------
        roi : ee.Geometry or list
            Region of interest
        class_values : list
            Class values (e.g., [0, 1, 2, 3, 4, 5, 6, 7, 8])
        class_names : list
            Class names (e.g., ['Water', 'Forest', ...])
        samples_per_class : int
            Number of samples per class
        scale : int
            Sampling resolution
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        ee.FeatureCollection : Training samples with 'class' property
        
        Examples:
        ---------
        >>> downloader = GEEDataDownloader()
        >>> roi = [85.0, 20.0, 87.0, 22.0]
        >>> class_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        >>> class_names = ['Water', 'Forest', 'Grassland', 'Cropland',
        ...                'Urban', 'Bareland', 'Wetland', 'Shrubland', 'Snow']
        >>> samples = downloader.generate_training_samples(
        ...     roi, class_values, class_names, samples_per_class=500
        ... )
        """
        # Convert ROI to geometry if needed
        if isinstance(roi, list):
            roi = ee.Geometry.Rectangle(roi)
        
        # Generate random points for each class
        all_samples = []
        
        for class_val, class_name in zip(class_values, class_names):
            # Generate random points
            points = ee.FeatureCollection.randomPoints(
                region=roi,
                points=samples_per_class,
                seed=seed + class_val
            )
            
            # Add class property
            points = points.map(lambda f: f.set('class', class_val).set('class_name', class_name))
            all_samples.append(points)
        
        # Merge all samples
        training_samples = ee.FeatureCollection(all_samples).flatten()
        
        print(f"✓ Generated {len(class_values) * samples_per_class} training samples")
        print(f"  Classes: {', '.join(class_names)}")
        print(f"  Samples per class: {samples_per_class}")
        
        return training_samples
    
    def create_stratified_samples_from_classification(
        self,
        classified_image: ee.Image,
        roi: Union[ee.Geometry, List[float]],
        class_band: str = 'classification',
        samples_per_class: int = 500,
        scale: int = 30,
        seed: int = 42
    ) -> ee.FeatureCollection:
        """
        Create stratified random samples from an existing classification.
        
        Useful for creating training data from existing land cover maps.
        
        Parameters:
        -----------
        classified_image : ee.Image
            Classified image with class values
        roi : ee.Geometry or list
            Region of interest
        class_band : str
            Name of classification band
        samples_per_class : int
            Number of samples per class
        scale : int
            Sampling resolution
        seed : int
            Random seed
        
        Returns:
        --------
        ee.FeatureCollection : Stratified samples
        
        Examples:
        ---------
        >>> # Use existing land cover map
        >>> lc_map = ee.Image('MODIS/006/MCD12Q1/2020_01_01').select('LC_Type1')
        >>> samples = downloader.create_stratified_samples_from_classification(
        ...     lc_map, roi, class_band='LC_Type1', samples_per_class=300
        ... )
        """
        # Convert ROI to geometry if needed
        if isinstance(roi, list):
            roi = ee.Geometry.Rectangle(roi)
        
        # Create stratified sample
        samples = classified_image.select(class_band).stratifiedSample(
            numPoints=samples_per_class,
            classBand=class_band,
            region=roi,
            scale=scale,
            seed=seed,
            geometries=True
        )
        
        # Rename class band to 'class' for consistency
        samples = samples.map(lambda f: f.set('class', f.get(class_band)))
        
        print(f"✓ Created stratified samples from classification")
        
        return samples
    
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
    
    def get_training_from_esa_worldcover(
        self,
        roi: Union[ee.Geometry, List[float]],
        year: int = 2021,
        samples_per_class: int = 500,
        scale: int = 10,
        seed: int = 42
    ) -> ee.FeatureCollection:
        """
        Generate training samples from ESA WorldCover dataset.
        
        ESA WorldCover provides global land cover maps at 10m resolution.
        This is a high-quality, publicly available dataset perfect for training.
        
        Parameters:
        -----------
        roi : ee.Geometry or list
            Region of interest
        year : int
            Year (2020 or 2021 available)
        samples_per_class : int
            Number of samples per class
        scale : int
            Sampling resolution (10m for WorldCover)
        seed : int
            Random seed
        
        Returns:
        --------
        ee.FeatureCollection : Training samples with 'class' property
        
        Class Mapping:
        --------------
        10: Tree cover
        20: Shrubland
        30: Grassland
        40: Cropland
        50: Built-up
        60: Bare / sparse vegetation
        70: Snow and ice
        80: Permanent water bodies
        90: Herbaceous wetland
        95: Mangroves
        100: Moss and lichen
        
        Examples:
        ---------
        >>> downloader = GEEDataDownloader()
        >>> roi = [85.0, 20.0, 87.0, 22.0]
        >>> samples = downloader.get_training_from_esa_worldcover(
        ...     roi, year=2021, samples_per_class=500
        ... )
        """
        # Convert ROI to geometry if needed
        if isinstance(roi, list):
            roi = ee.Geometry.Rectangle(roi)
        
        # Load ESA WorldCover
        worldcover = ee.ImageCollection('ESA/WorldCover/v200').first()
        if year == 2020:
            worldcover = ee.ImageCollection('ESA/WorldCover/v100').first()
        
        # Create stratified sample
        samples = worldcover.stratifiedSample(
            numPoints=samples_per_class,
            classBand='Map',
            region=roi,
            scale=scale,
            seed=seed,
            geometries=True
        )
        
        # Rename to 'class' for consistency
        samples = samples.map(lambda f: f.set('class', f.get('Map')))
        
        print(f"✓ Generated training samples from ESA WorldCover {year}")
        print(f"  Samples per class: {samples_per_class}")
        print(f"  Scale: {scale}m")
        
        return samples
    
    def get_training_from_dynamic_world(
        self,
        roi: Union[ee.Geometry, List[float]],
        start_date: str,
        end_date: str,
        samples_per_class: int = 500,
        scale: int = 10,
        seed: int = 42
    ) -> ee.FeatureCollection:
        """
        Generate training samples from Google Dynamic World dataset.
        
        Dynamic World provides near real-time land cover at 10m resolution
        using Sentinel-2 data. Updated every 2-5 days.
        
        Parameters:
        -----------
        roi : ee.Geometry or list
            Region of interest
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        samples_per_class : int
            Number of samples per class
        scale : int
            Sampling resolution (10m for Dynamic World)
        seed : int
            Random seed
        
        Returns:
        --------
        ee.FeatureCollection : Training samples with 'class' property
        
        Class Mapping:
        --------------
        0: Water
        1: Trees
        2: Grass
        3: Flooded vegetation
        4: Crops
        5: Shrub and scrub
        6: Built area
        7: Bare ground
        8: Snow and ice
        
        Examples:
        ---------
        >>> downloader = GEEDataDownloader()
        >>> roi = [85.0, 20.0, 87.0, 22.0]
        >>> samples = downloader.get_training_from_dynamic_world(
        ...     roi, '2023-01-01', '2023-12-31', samples_per_class=500
        ... )
        """
        # Convert ROI to geometry if needed
        if isinstance(roi, list):
            roi = ee.Geometry.Rectangle(roi)
        
        # Load Dynamic World
        dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .select('label') \
            .mode()  # Most common class
        
        # Create stratified sample
        samples = dw.stratifiedSample(
            numPoints=samples_per_class,
            classBand='label',
            region=roi,
            scale=scale,
            seed=seed,
            geometries=True
        )
        
        # Rename to 'class' for consistency
        samples = samples.map(lambda f: f.set('class', f.get('label')))
        
        print(f"✓ Generated training samples from Dynamic World")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Samples per class: {samples_per_class}")
        print(f"  Scale: {scale}m")
        
        return samples
    
    def get_training_from_modis_lc(
        self,
        roi: Union[ee.Geometry, List[float]],
        year: int = 2020,
        samples_per_class: int = 500,
        scale: int = 500,
        seed: int = 42,
        lc_type: int = 1
    ) -> ee.FeatureCollection:
        """
        Generate training samples from MODIS Land Cover dataset.
        
        MODIS provides global land cover at 500m resolution.
        Long time series available (2001-present).
        
        Parameters:
        -----------
        roi : ee.Geometry or list
            Region of interest
        year : int
            Year (2001-2022)
        samples_per_class : int
            Number of samples per class
        scale : int
            Sampling resolution (500m for MODIS)
        seed : int
            Random seed
        lc_type : int
            Land cover classification scheme (1-5)
            1: IGBP (17 classes) - Default
            2: UMD (15 classes)
            3: LAI (8 classes)
            4: BGC (8 classes)
            5: PFT (11 classes)
        
        Returns:
        --------
        ee.FeatureCollection : Training samples with 'class' property
        
        IGBP Classes (LC_Type1):
        -------------------------
        1: Evergreen Needleleaf Forests
        2: Evergreen Broadleaf Forests
        3: Deciduous Needleleaf Forests
        4: Deciduous Broadleaf Forests
        5: Mixed Forests
        6: Closed Shrublands
        7: Open Shrublands
        8: Woody Savannas
        9: Savannas
        10: Grasslands
        11: Permanent Wetlands
        12: Croplands
        13: Urban and Built-up Lands
        14: Cropland/Natural Vegetation Mosaics
        15: Permanent Snow and Ice
        16: Barren
        17: Water Bodies
        
        Examples:
        ---------
        >>> downloader = GEEDataDownloader()
        >>> roi = [85.0, 20.0, 87.0, 22.0]
        >>> samples = downloader.get_training_from_modis_lc(
        ...     roi, year=2020, samples_per_class=300
        ... )
        """
        # Convert ROI to geometry if needed
        if isinstance(roi, list):
            roi = ee.Geometry.Rectangle(roi)
        
        # Load MODIS Land Cover
        modis_lc = ee.ImageCollection('MODIS/006/MCD12Q1') \
            .filterDate(f'{year}-01-01', f'{year}-12-31') \
            .first() \
            .select(f'LC_Type{lc_type}')
        
        # Create stratified sample
        samples = modis_lc.stratifiedSample(
            numPoints=samples_per_class,
            classBand=f'LC_Type{lc_type}',
            region=roi,
            scale=scale,
            seed=seed,
            geometries=True
        )
        
        # Rename to 'class' for consistency
        samples = samples.map(lambda f: f.set('class', f.get(f'LC_Type{lc_type}')))
        
        print(f"✓ Generated training samples from MODIS Land Cover {year}")
        print(f"  Classification: LC_Type{lc_type}")
        print(f"  Samples per class: {samples_per_class}")
        print(f"  Scale: {scale}m")
        
        return samples
    
    def get_training_from_copernicus_lc(
        self,
        roi: Union[ee.Geometry, List[float]],
        year: int = 2020,
        samples_per_class: int = 500,
        scale: int = 100,
        seed: int = 42
    ) -> ee.FeatureCollection:
        """
        Generate training samples from Copernicus Global Land Cover.
        
        Copernicus provides global land cover at 100m resolution.
        
        Parameters:
        -----------
        roi : ee.Geometry or list
            Region of interest
        year : int
            Year (2015-2019 available)
        samples_per_class : int
            Number of samples per class
        scale : int
            Sampling resolution (100m for Copernicus)
        seed : int
            Random seed
        
        Returns:
        --------
        ee.FeatureCollection : Training samples with 'class' property
        
        Class Mapping:
        --------------
        0: Unknown
        20: Shrubs
        30: Herbaceous vegetation
        40: Cultivated and managed vegetation/agriculture
        50: Urban / built up
        60: Bare / sparse vegetation
        70: Snow and ice
        80: Permanent water bodies
        90: Herbaceous wetland
        100: Moss and lichen
        111: Closed forest, evergreen needle leaf
        112: Closed forest, evergreen broad leaf
        113: Closed forest, deciduous needle leaf
        114: Closed forest, deciduous broad leaf
        115: Closed forest, mixed
        116: Closed forest, not matching any of the other definitions
        121: Open forest, evergreen needle leaf
        122: Open forest, evergreen broad leaf
        123: Open forest, deciduous needle leaf
        124: Open forest, deciduous broad leaf
        125: Open forest, mixed
        126: Open forest, not matching any of the other definitions
        200: Oceans, seas
        
        Examples:
        ---------
        >>> downloader = GEEDataDownloader()
        >>> roi = [85.0, 20.0, 87.0, 22.0]
        >>> samples = downloader.get_training_from_copernicus_lc(
        ...     roi, year=2019, samples_per_class=300
        ... )
        """
        # Convert ROI to geometry if needed
        if isinstance(roi, list):
            roi = ee.Geometry.Rectangle(roi)
        
        # Load Copernicus Land Cover
        copernicus_lc = ee.Image(f'COPERNICUS/Landcover/100m/Proba-V-C3/Global/{year}') \
            .select('discrete_classification')
        
        # Create stratified sample
        samples = copernicus_lc.stratifiedSample(
            numPoints=samples_per_class,
            classBand='discrete_classification',
            region=roi,
            scale=scale,
            seed=seed,
            geometries=True
        )
        
        # Rename to 'class' for consistency
        samples = samples.map(lambda f: f.set('class', f.get('discrete_classification')))
        
        print(f"✓ Generated training samples from Copernicus Land Cover {year}")
        print(f"  Samples per class: {samples_per_class}")
        print(f"  Scale: {scale}m")
        
        return samples

