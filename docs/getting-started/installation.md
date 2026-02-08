# Installation

## Requirements

- Python 3.8 or higher
- pip package manager
- Google Earth Engine account
- Google Cloud Project

## Installation Methods

### Option 1: Install from GitHub (Recommended)

```bash
pip install git+https://github.com/pulakeshpradhan/deepgee.git
```

This will install the latest version directly from the GitHub repository.

### Option 2: Install from Wheel File

If you have downloaded the wheel file:

```bash
pip install deepgee-0.1.0-py3-none-any.whl
```

### Option 3: Install from Source

For development or customization:

```bash
# Clone the repository
git clone https://github.com/pulakeshpradhan/deepgee.git
cd deepgee

# Install in development mode
pip install -e .
```

### Option 4: Install with TensorFlow

To include TensorFlow for deep learning:

```bash
pip install git+https://github.com/pulakeshpradhan/deepgee.git[tensorflow]
```

## Dependencies

DeepGEE will automatically install the following dependencies:

### Core Dependencies

- `earthengine-api` >= 0.1.300
- `geemap` >= 0.20.0
- `numpy` >= 1.20.0
- `pandas` >= 1.3.0
- `rasterio` >= 1.2.0
- `matplotlib` >= 3.4.0
- `seaborn` >= 0.11.0
- `scikit-learn` >= 1.0.0
- `joblib` >= 1.0.0

### Optional Dependencies

- `tensorflow` >= 2.10.0 (for deep learning)
- `torch` >= 1.12.0 (for PyTorch support)

## Verify Installation

After installation, verify that DeepGEE is installed correctly:

```python
import deepgee

# Check version
print(deepgee.__version__)
# Output: 0.1.0

# Check available modules
print(dir(deepgee))
# Output: ['GEEDataDownloader', 'LandCoverClassifier', ...]
```

## Google Earth Engine Setup

Before using DeepGEE, you need to set up Google Earth Engine:

### 1. Sign Up for GEE

1. Visit [Google Earth Engine](https://earthengine.google.com/)
2. Click "Sign Up"
3. Sign in with your Google account
4. Wait for approval (usually 24-48 hours)

### 2. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Note your project ID (e.g., `my-gee-project-123456`)
4. Enable the Earth Engine API for your project

### 3. Authenticate (First Time Only)

```python
import deepgee

# This will open a browser for authentication
deepgee.authenticate_gee()
```

Follow the prompts to complete authentication.

## Troubleshooting

### Import Errors

If you encounter import errors:

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Install TensorFlow if needed
pip install tensorflow
```

### GEE Authentication Issues

If authentication fails:

```python
# Try re-authenticating
deepgee.authenticate_gee()

# Check status
status = deepgee.auth.check_gee_status()
print(status)
```

### Permission Errors

If you get permission errors during installation:

```bash
# Install with user flag
pip install --user git+https://github.com/pulakeshpradhan/deepgee.git
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade git+https://github.com/pulakeshpradhan/deepgee.git
```

## Uninstalling

To uninstall DeepGEE:

```bash
pip uninstall deepgee
```

## Next Steps

- [Quick Start Guide](quick-start.md) - Get started with DeepGEE
- [GEE Setup](gee-setup.md) - Detailed GEE configuration
- [User Guide](../user-guide/overview.md) - Comprehensive documentation

## Getting Help

If you encounter any issues:

- Check the [User Guide](../user-guide/overview.md)
- Review [Examples](../examples/land-cover.md)
- Open an [Issue on GitHub](https://github.com/pulakeshpradhan/deepgee/issues)
- Email: <pulakesh.mid@gmail.com>
