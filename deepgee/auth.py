"""
Authentication module for Google Earth Engine
"""

import ee
import os
from typing import Optional


def authenticate_gee(auth_mode: str = 'notebook') -> None:
    """
    Authenticate with Google Earth Engine.
    
    Parameters:
    -----------
    auth_mode : str, optional
        Authentication mode: 'notebook', 'gcloud', or 'service_account'
        Default is 'notebook'
    
    Examples:
    ---------
    >>> import deepgee
    >>> deepgee.authenticate_gee()
    >>> deepgee.initialize_gee(project='your-project-id')
    """
    try:
        if auth_mode == 'notebook':
            ee.Authenticate()
            print("✓ GEE authentication successful!")
        elif auth_mode == 'gcloud':
            ee.Authenticate(auth_mode='gcloud')
            print("✓ GEE authentication successful via gcloud!")
        elif auth_mode == 'service_account':
            print("For service account, use initialize_gee() with service_account parameter")
        else:
            raise ValueError(f"Unknown auth_mode: {auth_mode}")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        raise


def initialize_gee(
    project: Optional[str] = None,
    service_account: Optional[str] = None,
    key_file: Optional[str] = None,
    opt_url: Optional[str] = None
) -> None:
    """
    Initialize Google Earth Engine.
    
    Parameters:
    -----------
    project : str, optional
        GEE project ID (required for most operations)
    service_account : str, optional
        Service account email for authentication
    key_file : str, optional
        Path to service account key file (JSON)
    opt_url : str, optional
        Optional URL for high-volume endpoint
    
    Examples:
    ---------
    # Standard initialization
    >>> import deepgee
    >>> deepgee.initialize_gee(project='your-project-id')
    
    # Service account initialization
    >>> deepgee.initialize_gee(
    ...     project='your-project-id',
    ...     service_account='your-sa@project.iam.gserviceaccount.com',
    ...     key_file='path/to/key.json'
    ... )
    """
    try:
        if service_account and key_file:
            # Service account authentication
            credentials = ee.ServiceAccountCredentials(service_account, key_file)
            ee.Initialize(credentials, project=project, opt_url=opt_url)
            print(f"✓ GEE initialized with service account: {service_account}")
        elif project:
            # Standard authentication
            ee.Initialize(project=project, opt_url=opt_url)
            print(f"✓ GEE initialized with project: {project}")
        else:
            # Try to initialize without project (legacy)
            ee.Initialize(opt_url=opt_url)
            print("✓ GEE initialized (no project specified)")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        print("\nTroubleshooting:")
        print("1. Run: deepgee.authenticate_gee()")
        print("2. Ensure you have a valid GEE project")
        print("3. Check your internet connection")
        raise


def check_gee_status() -> dict:
    """
    Check if GEE is properly initialized.
    
    Returns:
    --------
    dict : Status information
    
    Examples:
    ---------
    >>> import deepgee
    >>> status = deepgee.auth.check_gee_status()
    >>> print(status)
    """
    try:
        # Try a simple operation
        test_image = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_044034_20140318')
        info = test_image.getInfo()
        
        return {
            'initialized': True,
            'status': 'OK',
            'test_image': info['id'],
            'message': 'GEE is properly initialized and working'
        }
    except Exception as e:
        return {
            'initialized': False,
            'status': 'ERROR',
            'error': str(e),
            'message': 'GEE is not initialized or not working properly'
        }


def get_project_info() -> dict:
    """
    Get information about the current GEE project.
    
    Returns:
    --------
    dict : Project information
    """
    try:
        # Get project assets
        root_assets = ee.data.getAssetRoots()
        
        return {
            'status': 'OK',
            'root_assets': root_assets,
            'message': 'Successfully retrieved project information'
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'error': str(e),
            'message': 'Failed to retrieve project information'
        }
