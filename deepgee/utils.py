"""
Utility functions for DeepGEE
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import seaborn as sns


def load_geotiff(filepath: str) -> Tuple[np.ndarray, Dict]:
    """
    Load a GeoTIFF file.
    
    Parameters:
    -----------
    filepath : str
        Path to GeoTIFF file
    
    Returns:
    --------
    tuple : (image_array, metadata)
    
    Examples:
    ---------
    >>> image, meta = load_geotiff('composite.tif')
    >>> print(image.shape)
    """
    with rasterio.open(filepath) as src:
        image = src.read()
        meta = src.meta.copy()
        meta['transform'] = src.transform
        meta['bounds'] = src.bounds
        meta['crs'] = src.crs
    
    return image, meta


def save_geotiff(
    image: np.ndarray,
    filepath: str,
    meta: Dict,
    nodata: Optional[float] = None
) -> str:
    """
    Save array as GeoTIFF.
    
    Parameters:
    -----------
    image : np.ndarray
        Image array (bands, height, width) or (height, width)
    filepath : str
        Output file path
    meta : dict
        Metadata dictionary
    nodata : float, optional
        No data value
    
    Returns:
    --------
    str : Path to saved file
    
    Examples:
    ---------
    >>> save_geotiff(classified, 'output.tif', meta, nodata=255)
    """
    # Ensure 3D array
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    
    # Update metadata
    meta.update({
        'count': image.shape[0],
        'dtype': image.dtype
    })
    
    if nodata is not None:
        meta['nodata'] = nodata
    
    # Write file
    with rasterio.open(filepath, 'w', **meta) as dst:
        dst.write(image)
    
    print(f"✓ GeoTIFF saved to: {filepath}")
    return filepath


def calculate_area_stats(
    classified_image: np.ndarray,
    class_names: List[str],
    pixel_size: float = 30.0,
    nodata: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate area statistics for classified image.
    
    Parameters:
    -----------
    classified_image : np.ndarray
        Classified image (2D array)
    class_names : list
        List of class names
    pixel_size : float
        Pixel size in meters
    nodata : int, optional
        No data value to exclude
    
    Returns:
    --------
    pd.DataFrame : Area statistics
    
    Examples:
    ---------
    >>> stats = calculate_area_stats(
    ...     classified, class_names, pixel_size=30
    ... )
    >>> print(stats)
    """
    # Flatten image
    if classified_image.ndim > 2:
        classified_image = classified_image[0]
    
    # Remove nodata
    if nodata is not None:
        mask = classified_image != nodata
        classified_image = classified_image[mask]
    
    # Calculate pixel area
    pixel_area_m2 = pixel_size ** 2
    pixel_area_km2 = pixel_area_m2 / 1e6
    
    # Count pixels per class
    unique, counts = np.unique(classified_image, return_counts=True)
    
    # Create dataframe
    stats = []
    for class_id, count in zip(unique, counts):
        if class_id < len(class_names):
            stats.append({
                'Class': class_names[class_id],
                'Class_ID': int(class_id),
                'Pixels': int(count),
                'Area_m2': float(count * pixel_area_m2),
                'Area_km2': float(count * pixel_area_km2),
                'Percentage': float(count / counts.sum() * 100)
            })
    
    df = pd.DataFrame(stats)
    return df


def plot_training_history(
    history,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> None:
    """
    Plot training history.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    cm : np.ndarray
        Confusion matrix
    class_names : list
        Class names
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    
    plt.show()


def plot_classification_map(
    classified_image: np.ndarray,
    class_names: List[str],
    class_colors: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    title: str = 'Land Cover Classification'
) -> None:
    """
    Plot classification map.
    
    Parameters:
    -----------
    classified_image : np.ndarray
        Classified image (2D)
    class_names : list
        Class names
    class_colors : list
        Class colors (hex codes)
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    title : str
        Plot title
    """
    from matplotlib.colors import ListedColormap
    
    # Flatten if needed
    if classified_image.ndim > 2:
        classified_image = classified_image[0]
    
    # Create colormap
    cmap = ListedColormap(class_colors)
    
    plt.figure(figsize=figsize)
    im = plt.imshow(classified_image, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    
    # Colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_ticks(range(len(class_names)))
    cbar.set_ticklabels(class_names)
    cbar.set_label('Land Cover Class', fontsize=12)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    
    plt.show()


def plot_area_distribution(
    stats_df: pd.DataFrame,
    class_colors: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot area distribution bar chart.
    
    Parameters:
    -----------
    stats_df : pd.DataFrame
        Area statistics dataframe
    class_colors : list, optional
        Class colors
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    if class_colors is None:
        class_colors = plt.cm.tab10(range(len(stats_df)))
    
    plt.bar(
        stats_df['Class'],
        stats_df['Area_km2'],
        color=class_colors,
        edgecolor='black',
        linewidth=1.5
    )
    
    plt.xlabel('Land Cover Class', fontsize=12, fontweight='bold')
    plt.ylabel('Area (km²)', fontsize=12, fontweight='bold')
    plt.title('Land Cover Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    
    plt.show()


def create_rgb_composite(
    image: np.ndarray,
    rgb_bands: Tuple[int, int, int] = (3, 2, 1),
    stretch: float = 2.5
) -> np.ndarray:
    """
    Create RGB composite from multi-band image.
    
    Parameters:
    -----------
    image : np.ndarray
        Multi-band image (bands, height, width)
    rgb_bands : tuple
        Band indices for RGB (0-indexed)
    stretch : float
        Contrast stretch factor
    
    Returns:
    --------
    np.ndarray : RGB composite (height, width, 3)
    """
    rgb = np.stack([
        image[rgb_bands[0]],
        image[rgb_bands[1]],
        image[rgb_bands[2]]
    ], axis=-1)
    
    # Normalize
    rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb_norm = np.clip(rgb_norm * stretch, 0, 1)
    
    return rgb_norm


def print_model_summary(results: Dict) -> None:
    """
    Print model evaluation summary.
    
    Parameters:
    -----------
    results : dict
        Evaluation results from classifier.evaluate()
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    print(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Kappa Coefficient: {results['kappa']:.4f}")
    print("="*60)
    
    if 'classification_report' in results:
        print("\nCLASSIFICATION REPORT:")
        print("-"*60)
        print(results['classification_report'])
    
    print("="*60 + "\n")
