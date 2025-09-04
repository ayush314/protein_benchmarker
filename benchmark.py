#!/usr/bin/env python3
"""
Protein Structure Generation Benchmarker
"""

from __future__ import annotations
import os
import sys
import argparse
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

# Local imports (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gearnet_utils import CAGearNet
from graphein_utils import protein_to_pyg
from constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from dataset_utils import download_d_fs_dataset, get_d_fs_files

# ------------------------------
# Utilities
# ------------------------------

def check_and_download_data():
    """Check if required files exist, and download them if not."""
    required_files = [
        "checkpoints/gearnet_ca.pth",
        "proteina_features/D_FS_eval_ca_features.pth", 
        "proteina_features/pdb_eval_ca_features.pth",
        "data/d_FS_index.txt"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nAttempting to download required files...")
        
        try:
            result = subprocess.run([sys.executable, "download_data.py"], 
                                  check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
            print("✓ Download completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Download failed: {e}")
            print("Please run 'python download_data.py' manually or check your internet connection.")
            sys.exit(1)
        except FileNotFoundError:
            print("✗ download_data.py not found. Please ensure it's in the same directory as benchmark.py")
            sys.exit(1)

def covariance(x: torch.Tensor, shrink: float = 0.0, eps: float = 1e-6) -> torch.Tensor:
    """Sample covariance of rows as samples with optional diagonal shrinkage.
    Args:
        x: (n, d)
        shrink: [0,1], 0=sample cov, 1=diag only
        eps: diagonal jitter
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    n, d = x.shape
    if n <= 1:
        return torch.eye(d, dtype=x.dtype, device=x.device) * eps
    mu = x.mean(0, keepdim=True)
    xm = x - mu
    cov = (xm.T @ xm) / (n - 1)
    if shrink > 0:
        var = torch.diag(cov)
        cov = (1 - shrink) * cov + shrink * torch.diag(var)
    return cov + torch.eye(d, device=x.device, dtype=x.dtype) * eps


# ------------------------------
# Data / Model wrappers
# ------------------------------

class PDBDataset(Dataset):
    def __init__(self, pdb_files: List[str]):
        self.pdb_files = pdb_files

    def __len__(self) -> int:
        return len(self.pdb_files)

    def __getitem__(self, idx: int):
        pdb_file = self.pdb_files[idx]
        graph = protein_to_pyg(pdb_file, deprotonate=False)
        coord_mask = graph.coords != 1e-5
        graph.coord_mask = coord_mask[..., 0]

        graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
        graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
        graph.node_id = torch.arange(graph.coords.shape[0]).unsqueeze(-1)
        return graph

@dataclass
class EncodedBatch:
    features: torch.Tensor  # (n, d)
    logits_C: torch.Tensor  # (n, C)
    logits_A: torch.Tensor  # (n, A)
    logits_T: torch.Tensor  # (n, T)


class FeatureExtractor:
    def __init__(self, ckpt_path: str, device: Union[str, torch.device] = "cuda"):
        self.device = torch.device(device)
        self.model = CAGearNet(ckpt_path=ckpt_path)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        self.model.eval()

    def _patch_cpdb_to_string_mode(self):
        import cpdb, gzip, os
        _orig = cpdb.parse
        def _robust_parse(fname=None, pdb_str=None, **kw):
            if fname is not None and pdb_str is None:
                fn = str(fname)
                raw = gzip.open(fn, "rb").read() if fn.endswith((".pdb.gz", ".ent.gz")) else open(fn, "rb").read()
                pdb_str = raw.decode("utf-8", errors="ignore")
                fname = None  # force string path
            return _orig(fname=fname, pdb_str=pdb_str, **kw)
        cpdb.parse = _robust_parse

    @torch.no_grad()
    def encode(self, pdb_files: List[str], batch_size: int = 12, num_workers: int = 8) -> EncodedBatch:
        self._patch_cpdb_to_string_mode()
        ds = PDBDataset(pdb_files)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        feats, c_logits, a_logits, t_logits = [], [], [], []
        for batch in tqdm(dl, desc="Encoding PDBs"):
            batch = batch.to(self.device)
            out = self.model(batch)
            feats.append(out["protein_feature"].detach().cpu())
            c_logits.append(out["pred_C"].detach().cpu())
            a_logits.append(out["pred_A"].detach().cpu())
            t_logits.append(out["pred_T"].detach().cpu())
        return EncodedBatch(
            features=torch.cat(feats, dim=0),
            logits_C=torch.cat(c_logits, dim=0),
            logits_A=torch.cat(a_logits, dim=0),
            logits_T=torch.cat(t_logits, dim=0),
        )


# ------------------------------
# Reference providers
# ------------------------------

class ReferenceProvider:
    name: str
    def get_feature_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    def get_fold_means(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class PrecomputedReference(ReferenceProvider):
    def __init__(self, name: str, features_path: str):
        self.name = name
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Missing precomputed features: {features_path}")
        self.payload = torch.load(features_path, map_location="cpu", weights_only=False)
        self._mu_cov: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._fold_means: Optional[Dict[str, torch.Tensor]] = None

    def get_feature_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._mu_cov is not None:
            return self._mu_cov
        p = self.payload
        need = ["FID_real_features_sum", "FID_real_features_cov_sum", "FID_real_features_num_samples"]
        if not all(k in p for k in need):
            raise KeyError(f"{self.name} lacks required FID stats: {need}")
        s = p["FID_real_features_sum"].double()
        sxx = p["FID_real_features_cov_sum"].double()
        n = float(p["FID_real_features_num_samples"]) or 1.0
        mu = s / n
        mu_outer = torch.outer(mu, mu)
        cov_num = sxx - n * mu_outer
        cov = cov_num / (n - 1)
        d = cov.size(0)
        cov = cov + torch.eye(d, dtype=cov.dtype, device=cov.device) * 1e-6
        self._mu_cov = (mu, cov)
        return self._mu_cov

    def get_fold_means(self) -> Dict[str, torch.Tensor]:
        if self._fold_means is not None:
            return self._fold_means
        out: Dict[str, torch.Tensor] = {}
        for lvl in ("C", "A", "T"):
            s_key = f"fJSD_{lvl}_real_features_sum"
            n_key = f"fJSD_{lvl}_real_features_num_samples"
            if s_key in self.payload and n_key in self.payload:
                s = self.payload[s_key].float()
                n = float(self.payload[n_key]) or 1.0
                out[lvl] = s / n
        if not out:
            raise KeyError(f"{self.name} lacks fold mean distributions")
        self._fold_means = out
        return out


class DatasetReference(ReferenceProvider):
    def __init__(self, name: str, pdb_files: List[str], encoder: FeatureExtractor,
                 batch_size: int, num_workers: int, shrink: float = 0.0):
        self.name = name
        self.pdb_files = pdb_files
        self.encoder = encoder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shrink = shrink
        self._encoded: Optional[EncodedBatch] = None
        self._mu_cov: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._fold_means: Optional[Dict[str, torch.Tensor]] = None

    def _encode_once(self) -> EncodedBatch:
        if self._encoded is None:
            self._encoded = self.encoder.encode(self.pdb_files, self.batch_size, self.num_workers)
        return self._encoded

    def get_feature_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._mu_cov is not None:
            return self._mu_cov
        enc = self._encode_once()
        x = enc.features.float()
        mu = x.mean(0)
        cov = covariance(x, shrink=self.shrink)
        self._mu_cov = (mu, cov)
        return self._mu_cov

    def get_fold_means(self) -> Dict[str, torch.Tensor]:
        if self._fold_means is not None:
            return self._fold_means
        enc = self._encode_once()
        out = {
            "C": F.softmax(enc.logits_C, dim=-1).mean(0).float(),
            "A": F.softmax(enc.logits_A, dim=-1).mean(0).float(),
            "T": F.softmax(enc.logits_T, dim=-1).mean(0).float(),
        }
        self._fold_means = out
        return out


# ------------------------------
# Metrics
# ------------------------------

class MetricComputer:
    def __init__(self, eps: float = 1e-6, shrink: float = 0.0):
        self.eps = eps
        self.shrink = shrink

    def fid(self, gen_feats: torch.Tensor, ref_mu: torch.Tensor, ref_cov: torch.Tensor) -> float:
        """Compute FID like in Proteina's _compute_fid function."""
        # Use double precision
        x = gen_feats.double()
        ref_mu = ref_mu.double()
        ref_cov = ref_cov.double()
        
        mu_g = x.mean(0)
        cov_g = covariance(x, shrink=self.shrink, eps=self.eps)
        
        # FID calculation
        a = (mu_g - ref_mu).square().sum(dim=-1)
        b = cov_g.trace() + ref_cov.trace()
        c = torch.linalg.eigvals(cov_g @ ref_cov).sqrt().real.sum(dim=-1)
        
        return float((a + b - 2 * c).item())

    def fold_scores(self, logits: torch.Tensor, splits: int = 1) -> float:
        """Compute fold score like in Proteina's ProteinFoldScore."""
        # Random permutation
        idx = torch.randperm(logits.shape[0])
        features = logits[idx]
        
        # Calculate probs and log_probs
        prob = F.softmax(features, dim=1)
        log_prob = F.log_softmax(features, dim=1)
        
        # Split into groups (using chunk)
        prob_chunks = prob.chunk(splits, dim=0)
        log_prob_chunks = log_prob.chunk(splits, dim=0)
        
        # Calculate score per split
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob_chunks]
        kl_scores = []
        
        for p, log_p, m_p in zip(prob_chunks, log_prob_chunks, mean_prob):
            # KL divergence: p * (log_p - log(m_p))
            kl = p * (log_p - (m_p + 1e-10).log())
            # Sum over features, mean over batch, then exponentiate
            kl_score = kl.sum(dim=1).mean().exp()
            kl_scores.append(kl_score)
        
        return torch.stack(kl_scores).mean().item()

    def fjsd_against_means(self, gen_logits: torch.Tensor, ref_mean: torch.Tensor) -> float:
        p_g = F.softmax(gen_logits, dim=-1).mean(0, keepdim=True)
        p_r = ref_mean.unsqueeze(0)
        m = 0.5 * (p_g + p_r)
        js = 0.5 * (F.kl_div((m + 1e-10).log(), p_g, reduction="batchmean") +
                    F.kl_div((m + 1e-10).log(), p_r, reduction="batchmean"))
        return float(js.item() * 10.0)


# ------------------------------
# Benchmark
# ------------------------------

def create_reference_features(dataset_path: str, output_path: str, checkpoint_path: str = "checkpoints/gearnet_ca.pth",
                            batch_size: int = 12, num_workers: int = 8, device: str = "cuda"):
    """
    Create reference feature files for a dataset that can be used with --reference_features.

    Args:
        dataset_path: Path to directory containing PDB files
        output_path: Path to save the feature file (.pth)
        checkpoint_path: Path to GearNet checkpoint
        batch_size: Batch size for encoding
        num_workers: Number of workers for data loading
        device: Device for computation
    """
    import os
    from pathlib import Path

    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset path is not a directory: {dataset_path}")

    pdb_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.pdb')]
    if not pdb_files:
        raise ValueError(f"No PDB files found in: {dataset_path}")

    print(f"Found {len(pdb_files)} PDB files in {dataset_path}")

    # Initialize encoder
    extractor = FeatureExtractor(checkpoint_path, device=device)

    # Encode all PDB files
    print("Encoding PDB files...")
    encoded = extractor.encode(pdb_files, batch_size=batch_size, num_workers=num_workers)

    # Compute FID statistics
    features = encoded.features.float()
    n_samples = features.shape[0]

    # FID stats
    fid_sum = features.sum(dim=0)
    fid_cov_sum = (features.unsqueeze(1) * features.unsqueeze(0)).sum(dim=2)

    # fJSD stats for each class
    fjsd_stats = {}
    for lvl, logits in [("C", encoded.logits_C), ("A", encoded.logits_A), ("T", encoded.logits_T)]:
        probs = F.softmax(logits, dim=-1).float()
        fjsd_stats[f"fJSD_{lvl}_real_features_sum"] = probs.sum(dim=0)
        fjsd_stats[f"fJSD_{lvl}_real_features_num_samples"] = n_samples

    # Create payload dictionary
    payload = {
        "FID_real_features_sum": fid_sum,
        "FID_real_features_cov_sum": fid_cov_sum,
        "FID_real_features_num_samples": n_samples,
        **fjsd_stats
    }

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    print(f"✓ Saved reference features to {output_path}")
    print(f"  - Dataset: {len(pdb_files)} PDB files")
    print(f"  - Feature dimension: {features.shape[1]}")
    print(f"  - Classes: C={encoded.logits_C.shape[1]}, A={encoded.logits_A.shape[1]}, T={encoded.logits_T.shape[1]}")


def run_benchmark(args: argparse.Namespace) -> Dict[str, float]:
    if not os.path.isdir(args.generated_dir):
        raise FileNotFoundError(f"Generated directory not found or not a directory: {args.generated_dir}")

    gen_files = [os.path.join(args.generated_dir, f) for f in os.listdir(args.generated_dir) if f.endswith('.pdb')]
    if not gen_files:
        raise RuntimeError(f"No .pdb files found in generated_dir: {args.generated_dir}")

    extractor = FeatureExtractor(args.gearnet_checkpoint, device=args.device)

    providers: List[ReferenceProvider] = []
    
    # Determine reference method based on what's specified
    if args.download_indices:
        # Download dataset from indices
        idx_path = args.download_indices
        d_fs_files = download_d_fs_dataset(idx_path,
                                           os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/d_FS_dataset"),
                                           max_structures=args.max_structures)
        if not d_fs_files:
            raise RuntimeError("No reference D_FS pdbs found")
        providers.append(DatasetReference("D_FS", d_fs_files, extractor, args.batch_size, args.num_workers, shrink=args.shrink))
    elif args.reference_dataset:
        # Use existing dataset
        ds_path = args.reference_dataset
        d_fs_files = get_d_fs_files(ds_path)
        if not d_fs_files:
            raise RuntimeError("No reference D_FS pdbs found")
        providers.append(DatasetReference("D_FS", d_fs_files, extractor, args.batch_size, args.num_workers, shrink=args.shrink))
    else:
        # Default to proteina features
        pdir = args.reference_features
        providers.append(PrecomputedReference("PDB", os.path.join(pdir, "pdb_eval_ca_features.pth")))
        providers.append(PrecomputedReference("D_FS", os.path.join(pdir, "D_FS_eval_ca_features.pth")))
        for file in os.listdir(pdir):
            if file.endswith(".pth"):
                providers.append(PrecomputedReference(file.replace(".pth", ""), os.path.join(pdir, file)))

    gen_encoded = extractor.encode(gen_files, args.batch_size, args.num_workers)

    mc = MetricComputer(eps=args.eps, shrink=args.shrink)
    metrics: Dict[str, float] = {
        "fS_C": mc.fold_scores(gen_encoded.logits_C, splits=args.fold_splits),
        "fS_A": mc.fold_scores(gen_encoded.logits_A, splits=args.fold_splits),
        "fS_T": mc.fold_scores(gen_encoded.logits_T, splits=args.fold_splits),
    }

    for prov in providers:
        mu_r, cov_r = prov.get_feature_stats()
        metrics[f"{prov.name}_FID"] = mc.fid(gen_encoded.features, mu_r, cov_r)
        fold_means = prov.get_fold_means()
        if "C" in fold_means:
            metrics[f"{prov.name}_fJSD_C"] = mc.fjsd_against_means(gen_encoded.logits_C, fold_means["C"])
        if "A" in fold_means:
            metrics[f"{prov.name}_fJSD_A"] = mc.fjsd_against_means(gen_encoded.logits_A, fold_means["A"])
        if "T" in fold_means:
            metrics[f"{prov.name}_fJSD_T"] = mc.fjsd_against_means(gen_encoded.logits_T, fold_means["T"])

    return metrics


# ------------------------------
# CLI
# ------------------------------

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark protein structure generation models (simplified)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--generated_dir", type=str, required=True, help="Directory with generated .pdb files")
    p.add_argument("--gearnet_checkpoint", type=str, default="checkpoints/gearnet_ca.pth", 
                   help="Path to GearNet checkpoint (gearnet_ca.pth)")

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--reference_features",
        nargs="?",
        const="proteina_features",
        help="Use precomputed GearNet features (dir). Default if no value: proteina_features",
    )
    grp.add_argument(
        "--download_indices",
        nargs="?",
        const="data/d_FS_index.txt",
        help="Download D_FS from AFDB indices (path to index.txt). Default if no value: data/d_FS_index.txt",
    )
    grp.add_argument(
        "--reference_dataset",
        nargs="?",
        const="data/d_FS_dataset",
        help="Use existing dataset directory. Default if no value: data/d_FS_dataset",
    )

    p.add_argument("--max_structures", type=lambda x: None if x.lower() == 'none' else int(x), 
                   default=1000, help="Max D_FS structures to download out of 588K (use 'None' for all)")

    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    
    p.add_argument("--eps", type=float, default=1e-6, help="Numerical jitter added to covariances")
    p.add_argument("--shrink", type=float, default=0.0, help="Covariance shrinkage in [0,1]")
    p.add_argument("--fold_splits", type=int, default=1, help="Number of splits for fold entropy scores")

    return p


def main() -> None:
    args = build_cli().parse_args()
    print("=" * 60)
    print("PROTEIN STRUCTURE GENERATION BENCHMARKER")
    print("=" * 60)

    # Check and download required files if missing
    check_and_download_data()

    try:
        metrics = run_benchmark(args)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for k in sorted(metrics.keys()):
        v = metrics[k]
        print(f"{k:20s}: {v:.6f}")


if __name__ == "__main__":
    main()
