"""
Protein Structure Generation Benchmarker

A standalone package for benchmarking protein structure generation models.
"""

from benchmark import ProteinBenchmarker
from dataset_utils import download_d_fs_dataset, get_d_fs_files

__version__ = "0.1.0"
__all__ = [
    "ProteinBenchmarker",
    "download_d_fs_dataset", 
    "get_d_fs_files"
] 