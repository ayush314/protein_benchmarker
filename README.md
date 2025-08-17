# Protein Structure Generation Benchmarker

A standalone package for benchmarking protein structure generation models. Computes FID, fJSD, and fold scores between generated and reference protein structures.

## Setup

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

Required files will be automatically downloaded when you run the benchmark. Alternatively, download them manually:

```bash
python download_data.py
```

## Usage

### Basic usage (recommended)

Compare against Proteina's pre-computed GearNet features:
```bash
python benchmark.py \
    --generated_dir path/to/generated/pdbs \
    --reference_features proteina_features
```

### Alternative reference datasets

Download and compare against Proteina's D_FS dataset:
```bash
python benchmark.py \
    --generated_dir path/to/generated/pdbs \
    --download_indices data/d_FS_index.txt \
    --max_structures None
```

Compare against a previously downloaded Proteina D_FS dataset:
```bash
python benchmark.py \
    --generated_dir path/to/generated/pdbs \
    --reference_dataset data/d_FS_dataset
```

## Arguments

**Required:**
- `--generated_dir`: Directory with generated PDB files
- `--gearnet_checkpoint`: Path to GearNet checkpoint

**Reference (choose one):**
- `--reference_features`: Use pre-computed GearNet features (default: proteina_features)
- `--download_indices`: Download and use a dataset from AFDB indices (default: data/d_FS_index.txt) 
- `--reference_dataset`: Use an existing dataset dir (default: data/d_FS_dataset)

**Optional (with defaults):**
- `--max_structures`: Max structures to download from indices (default: 1000, use 'None' for all 588K)
- `--batch_size`: Batch size for feature extraction (default: 12)
- `--num_workers`: Number of workers (default: 8)
- `--device`: cuda/cpu (default: cuda)

## Output Metrics

- **FID**: Fréchet distance between generated and reference features
- **fJSD**: Fold Jensen-Shannon divergence at C/A/T levels
- **fS**: Fold scores at C/A/T levels