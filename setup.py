"""
Setup script for the Protein Structure Generation Benchmarker.

NOTE: This package requires PyTorch Geometric and its dependencies.
For proper installation, run the install script first:
    bash install_dependencies.sh

Or install manually:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install torch-geometric
    pip install torch-scatter torch-sparse torch-cluster --find-links https://data.pyg.org/whl/torch-2.0.0+cu118.html
"""

from setuptools import setup, find_packages

setup(
    name="protein-benchmarker",
    version="0.1.0",
    author="Research Team",
    description="Standalone package for benchmarking protein structure generation models",
    long_description=__doc__,
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies (installed automatically)
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "pandas>=1.3.0",
        "biopandas>=0.4.0",
        "requests>=2.25.0",
        "wget>=3.2",
        "cpdb-protein>=0.1.0",
        "loguru>=0.6.0",
        "jaxtyping>=0.2.0",
        "biopython>=1.81",
        
        # PyTorch dependencies (may need manual installation)
        "torch>=1.12.0",
        "torch-geometric>=2.1.0",
        
        # Note: torch-scatter, torch-sparse, torch-cluster need to be installed
        # manually with the correct CUDA version. See install_dependencies.sh
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "full": [
            # These require manual installation with correct CUDA version
            "torch-scatter",
            "torch-sparse", 
            "torch-cluster",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 