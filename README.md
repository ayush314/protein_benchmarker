# Protein Structure Generation Benchmarker

A standalone package for benchmarking protein structure generation models. Computes FID, fJSD, and fold scores between generated and reference protein structures.

## Installation

```bash
./install_dependencies.sh
```

## Downloads

Before running the benchmarker, you'll need to download the required files:

### GearNet Checkpoint and Proteina Features
Download from [NVIDIA NGC - Proteina Auxiliary Files](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/proteina_additional_files/files):
- `gearnet_ca.pth`: GearNet classifier weights for alpha carbon representations
- `D_FS_eval_ca_features.pth`: Pre-computed features for D_FS dataset 
- `pdb_eval_ca_features.pth`: Pre-computed features for PDB dataset

### D_FS Index File  
Download from [NVIDIA NGC - Proteina Training Data Indices](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/proteina_training_data_indices/files):
- `d_fs_index.txt`: Indices of AlphaFold Database entries for D_FS dataset

## Usage

### Use pre-computed Proteina features
```bash
python benchmark.py \
    --generated_dir /path/to/generated/pdbs \
    --gearnet_checkpoint /path/to/gearnet_ca.pth \
    --use_proteina_features
```

### Download D_FS dataset as reference
```bash
python benchmark.py \
    --generated_dir /path/to/generated/pdbs \
    --gearnet_checkpoint /path/to/gearnet_ca.pth \
    --download_d_fs /path/to/d_FS_index.txt \
    --max_structures 1000
```

### Use existing D_FS dataset
```bash
python benchmark.py \
    --generated_dir /path/to/generated/pdbs \
    --gearnet_checkpoint /path/to/gearnet_ca.pth \
    --d_fs_dir /path/to/d_fs_dataset
```

## Arguments

**Required:**
- `--generated_dir`: Directory with generated PDB files
- `--gearnet_checkpoint`: Path to GearNet checkpoint

**Reference dataset (choose one):**
- `--proteina_features_dir`: Path to pre-computed features directory
- `--download_d_fs`: Path to d_FS_index.txt file to download D_FS dataset
- `--d_fs_dir`: Path to existing D_FS dataset

**Optional (with defaults):**
- `--max_structures`: Max structures to download (default: 1000)
- `--batch_size`: Batch size (default: 12)
- `--num_workers`: Number of workers (default: 8)
- `--device`: cuda/cpu (default: cuda)

## Output Metrics

- **FID**: Fréchet distance between generated and reference features
- **fJSD**: Fold Jensen-Shannon divergence at C/A/T levels
- **fS**: Fold scores at C/A/T levels