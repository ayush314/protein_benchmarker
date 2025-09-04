"""
Dataset utilities for downloading and working with AlphaFold Database structures.

This module provides utilities to reconstruct the D_FS dataset using the
provided index files and download structures from the AlphaFold Database.
"""

import os
import urllib.request
from typing import List, Optional
from pathlib import Path
import time


def read_d_fs_index(index_file_path: str) -> List[str]:
    """
    Read the D_FS index file to get AlphaFold Database IDs.
    
    Args:
        index_file_path: Path to d_FS_index.txt file
        
    Returns:
        List of AlphaFold Database IDs (e.g., "AF-A0A009IHW8-F1-model_v4")
    """
    if not os.path.exists(index_file_path):
        raise FileNotFoundError(f"Index file not found: {index_file_path}")
    
    af_ids = []
    with open(index_file_path, 'r') as f:
        for line in f:
            af_id = line.strip()
            if af_id:
                af_ids.append(af_id)
    
    print(f"Loaded {len(af_ids)} AlphaFold IDs from {index_file_path}")
    return af_ids


def download_alphafold_structure(af_id: str, output_dir: str, max_retries: int = 3) -> Optional[str]:
    """
    Download a single AlphaFold structure.
    
    Args:
        af_id: AlphaFold ID (e.g., "AF-A0A009IHW8-F1-model_v4")
        output_dir: Directory to save the PDB file
        max_retries: Maximum number of retry attempts
        
    Returns:
        Path to downloaded file or None if failed
    """
    # AlphaFold Database URL pattern
    url = f"https://alphafold.ebi.ac.uk/files/{af_id}.pdb"
    
    # Output file path
    output_file = os.path.join(output_dir, f"{af_id}.pdb")
    
    # Skip if already downloaded
    if os.path.exists(output_file):
        return output_file
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Download with retries
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, output_file)
            return output_file
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Download failed for {af_id}, retrying... ({attempt + 1}/{max_retries})")
                time.sleep(1)
            else:
                print(f"Failed to download {af_id} after {max_retries} attempts: {e}")
                return None


def download_d_fs_dataset(
    index_file_path: str,
    output_dir: str,
    max_structures: Optional[int] = None
) -> List[str]:
    """
    Download D_FS dataset structures from AlphaFold Database.
    
    Args:
        index_file_path: Path to d_FS_index.txt file
        output_dir: Directory to save downloaded PDB files
        max_structures: Maximum number of structures to download (None = all)
        
    Returns:
        List of successfully downloaded PDB file paths
    """
    # Read AlphaFold IDs from index file
    af_ids = read_d_fs_index(index_file_path)
    
    # Limit number of structures if specified
    if max_structures is not None:
        af_ids = af_ids[:max_structures]
        print(f"Limiting download to {max_structures} structures")
    
    print(f"Downloading {len(af_ids)} structures to {output_dir}")
    
    # Download structures
    downloaded_files = []
    for i, af_id in enumerate(af_ids):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(af_ids)} ({i/len(af_ids)*100:.1f}%)")
        
        downloaded_file = download_alphafold_structure(af_id, output_dir)
        if downloaded_file:
            downloaded_files.append(downloaded_file)
    
    print(f"Successfully downloaded {len(downloaded_files)}/{len(af_ids)} structures")
    return downloaded_files


def get_d_fs_files(d_fs_directory: str) -> List[str]:
    """
    Get list of D_FS PDB files from a directory.
    
    Args:
        d_fs_directory: Directory containing D_FS PDB files
        
    Returns:
        List of PDB file paths
    """
    if not os.path.exists(d_fs_directory):
        raise FileNotFoundError(f"D_FS directory not found: {d_fs_directory}")
    
    pdb_files = []
    for filename in os.listdir(d_fs_directory):
        if filename.endswith('.pdb'):
            pdb_files.append(os.path.join(d_fs_directory, filename))
    
    pdb_files.sort()
    print(f"Found {len(pdb_files)} D_FS PDB files in {d_fs_directory}")
    return pdb_files 
