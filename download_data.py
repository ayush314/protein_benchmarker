#!/usr/bin/env python3
"""
Download and setup required files for Protein Structure Generation Benchmarker
"""

import os
import sys
import subprocess
import zipfile
import shutil

def download_and_extract_essential_files():
    """Download and extract only the essential files needed for benchmarking."""
    
    # Create directory structure
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("proteina_features", exist_ok=True) 
    os.makedirs("data", exist_ok=True)
    
    print("=" * 60)
    print("DOWNLOADING ESSENTIAL PROTEINA FILES")
    print("=" * 60)
    
    success = True
    
    # Download indices zip and extract only d_FS_index.txt
    print("\n1. Downloading Training Data Indices...")
    try:
        subprocess.run([
            "curl", "-L", 
            "https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/proteina_training_data_indices/1.0/files?redirect=true&path=proteina_training_data_indices.zip",
            "-o", "indices.zip"
        ], check=True)
        
        with zipfile.ZipFile("indices.zip", 'r') as z:
            # Extract d_FS_index.txt (note the capital S!)
            if "d_FS_index.txt" in z.namelist():
                z.extract("d_FS_index.txt", ".")
                shutil.move("d_FS_index.txt", "data/d_FS_index.txt")
                print("  ✓ Extracted d_FS_index.txt → data/")
            else:
                print("  ✗ d_FS_index.txt not found in archive")
                success = False
        
        os.remove("indices.zip")
        
    except Exception as e:
        print(f"  ✗ Failed to download indices: {e}")
        success = False
    
    # Download additional files zip and extract essential files
    print("\n2. Downloading Additional Files...")
    try:
        subprocess.run([
            "curl", "-L",
            "https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/proteina_additional_files/1.0/files?redirect=true&path=proteina_additional_files.zip", 
            "-o", "additional.zip"
        ], check=True)
        
        with zipfile.ZipFile("additional.zip", 'r') as z:
            # Extract to temp directory
            z.extractall("temp_additional")
            
            # Find and move essential files
            essential_files = {
                "gearnet_ca.pth": "checkpoints/",
                "D_FS_eval_ca_features.pth": "proteina_features/",
                "pdb_eval_ca_features.pth": "proteina_features/"
            }
            
            for filename, target_dir in essential_files.items():
                found = False
                for root, dirs, files in os.walk("temp_additional"):
                    if filename in files:
                        source = os.path.join(root, filename)
                        target = os.path.join(target_dir, filename)
                        shutil.move(source, target)
                        print(f"  ✓ Extracted {filename} → {target_dir}")
                        found = True
                        break
                if not found:
                    print(f"  ✗ {filename} not found in archive")
                    success = False
        
        # Clean up
        shutil.rmtree("temp_additional", ignore_errors=True)
        os.remove("additional.zip")
        
    except Exception as e:
        print(f"  ✗ Failed to download additional files: {e}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ DOWNLOAD COMPLETE!")
        print("\nEssential files ready:")
        print("  checkpoints/gearnet_ca.pth")
        print("  proteina_features/D_FS_eval_ca_features.pth") 
        print("  proteina_features/pdb_eval_ca_features.pth")
        print("  data/d_FS_index.txt")
    else:
        print("✗ DOWNLOAD FAILED")
        sys.exit(1)

if __name__ == "__main__":
    download_and_extract_essential_files() 