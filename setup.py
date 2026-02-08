"""
DeepGEE: Earth Observation with Google Earth Engine and Deep Learning
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deepgee",
    version="0.1.0",
    author="Pulakesh Pradhan",
    author_email="pulakesh.mid@gmail.com",
    description="Earth Observation with Google Earth Engine and Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    repo_url="https://github.com/pulakeshpradhan/deepgee",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "earthengine-api>=0.1.300",
        "geemap>=0.20.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "rasterio>=1.2.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.10.0"],
        "pytorch": ["torch>=1.12.0", "torchvision>=0.13.0"],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.5.0",
        ],
    },
    keywords="earth-observation gee google-earth-engine deep-learning remote-sensing",
    project_urls={
        "Documentation": "https://github.com/pulakeshpradhan/deepgee#readme",
        "Source": "https://github.com/pulakeshpradhan/deepgee",
        "Bug Reports": "https://github.com/pulakeshpradhan/deepgee/issues",
    },
)
