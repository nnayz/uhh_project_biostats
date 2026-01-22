#!/usr/bin/env python3
"""
Generate a slide-ready spatial clusters figure.

Loads the raw h5ad file, runs Leiden clustering if needed, and creates a 
scatter plot showing the spatial distribution of clusters.

Output: diagnostics/figures/spatial_clusters.png (or custom path via --out_path)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_spatial_clusters(
    raw_data_path: str,
    out_path: Path,
    sample_size: int = 50000,
    dpi: int = 200,
    seed: int = 42,
    resolution: float = 0.8,
) -> bool:
    """
    Plot spatial distribution of clusters from raw h5ad data.
    
    If no cluster column exists, runs quick Leiden clustering.
    
    Args:
        raw_data_path: Path to h5ad file
        out_path: Output path for figure
        sample_size: Number of cells to sample for plotting (for speed)
        dpi: Figure resolution
        seed: Random seed for sampling
        resolution: Leiden clustering resolution
        
    Returns:
        True if successful, False otherwise
    """
    import matplotlib.pyplot as plt
    
    try:
        import scanpy as sc
    except ImportError:
        _log("[ERROR] scanpy not installed - please run: pip install scanpy")
        return False
    
    raw_path = Path(raw_data_path)
    if not raw_path.exists():
        _log(f"[ERROR] Raw data not found: {raw_path}")
        return False
    
    _log(f"Loading spatial data from {raw_path}...")
    
    try:
        # Load full data (need it for clustering if no cluster column)
        adata = sc.read_h5ad(raw_path)
        obs = adata.obs
        
        _log(f"  Loaded {adata.n_obs:,} cells, {adata.n_vars} genes")
        
        # Get coordinates
        x_col = None
        y_col = None
        for col in ['x_centroid', 'x', 'X_centroid']:
            if col in obs.columns:
                x_col = col
                break
        for col in ['y_centroid', 'y', 'Y_centroid']:
            if col in obs.columns:
                y_col = col
                break
        
        if x_col is None or y_col is None:
            # Check obsm for spatial coordinates
            if 'spatial' in adata.obsm:
                coords = adata.obsm['spatial']
                adata.obs['x'] = coords[:, 0]
                adata.obs['y'] = coords[:, 1]
                x_col, y_col = 'x', 'y'
            else:
                _log("[ERROR] Could not find spatial coordinates")
                return False
        
        # Get cluster labels - check multiple potential columns
        cluster_col = None
        for col in ['cluster', 'leiden', 'louvain', 'cell_type', 'annotation', 'niche', 'region']:
            if col in obs.columns:
                unique_vals = obs[col].dropna().unique()
                # Check if column has meaningful values (not just 'nan' string)
                if len(unique_vals) > 1 or (len(unique_vals) == 1 and str(unique_vals[0]).lower() != 'nan'):
                    cluster_col = col
                    break
        
        # If no cluster column found, run Leiden clustering
        if cluster_col is None:
            _log("  No cluster column found - running Leiden clustering...")
            
            # Preprocess for clustering
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            
            # Use standard flavor (doesn't require skmisc)
            sc.pp.highly_variable_genes(adata, n_top_genes=min(200, adata.n_vars), flavor='seurat')
            sc.pp.pca(adata, n_comps=min(30, adata.n_vars - 1), mask_var='highly_variable')
            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(30, adata.n_vars - 1))
            
            # Use igraph flavor for Leiden (more compatible)
            sc.tl.leiden(adata, resolution=resolution, flavor='igraph', n_iterations=2, directed=False)
            
            cluster_col = 'leiden'
            _log(f"  Created {adata.obs[cluster_col].nunique()} Leiden clusters")
        
        _log(f"  Using coordinates: {x_col}, {y_col}")
        _log(f"  Using cluster column: {cluster_col}")
        
        # Sample for plotting
        np.random.seed(seed)
        n_total = len(adata)
        if n_total > sample_size:
            idx = np.random.choice(n_total, sample_size, replace=False)
            obs_sample = adata.obs.iloc[idx].copy()
            _log(f"  Sampled {sample_size:,} of {n_total:,} cells for plotting")
        else:
            obs_sample = adata.obs.copy()
            _log(f"  Using all {n_total:,} cells")
        
        # Get coordinates and clusters
        x = obs_sample[x_col].values
        y = obs_sample[y_col].values
        clusters = obs_sample[cluster_col].astype(str).values
        
        # Get unique clusters and assign colors (sort numerically if possible)
        unique_clusters = sorted(set(clusters), key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else c))
        n_clusters = len(unique_clusters)
        
        # Use a good colormap for clusters
        if n_clusters <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_clusters]
        else:
            colors = plt.cm.turbo(np.linspace(0.1, 0.9, n_clusters))
        
        cluster_to_color = {c: colors[i] for i, c in enumerate(unique_clusters)}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Plot each cluster
        for cluster in unique_clusters:
            mask = clusters == cluster
            ax.scatter(
                x[mask], y[mask],
                c=[cluster_to_color[cluster]],
                s=0.5,
                alpha=0.7,
                label=f"Cluster {cluster}",
                rasterized=True  # For smaller file size
            )
        
        ax.set_title(f"Spatial Distribution of Clusters\n({n_clusters} clusters, {len(obs_sample):,} cells shown)", 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel("X coordinate (μm)", fontsize=13)
        ax.set_ylabel("Y coordinate (μm)", fontsize=13)
        ax.set_aspect('equal')
        
        # Add legend (compact for many clusters)
        if n_clusters <= 10:
            ax.legend(loc='upper right', fontsize=9, markerscale=8)
        elif n_clusters <= 25:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                      fontsize=8, markerscale=6, ncol=1)
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                      fontsize=6, markerscale=5, ncol=2)
        
        plt.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        _log(f"Saved: {out_path}")
        
        # Clean up memory
        del adata
        
        return True
        
    except Exception as e:
        _log(f"[ERROR] Failed to generate spatial_clusters.png: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate spatial clusters figure from raw h5ad data."
    )
    parser.add_argument(
        "--raw_data",
        type=str,
        default="data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad",
        help="Path to raw h5ad file",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="diagnostics/figures/spatial_clusters.png",
        help="Output path for figure",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_size", type=int, default=50000, help="Max cells to plot")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--resolution", type=float, default=0.8, help="Leiden resolution")
    
    args = parser.parse_args()
    
    _set_seed(args.seed)
    
    out_path = Path(args.out_path)
    _safe_mkdir(out_path.parent)
    
    success = plot_spatial_clusters(
        raw_data_path=args.raw_data,
        out_path=out_path,
        sample_size=args.sample_size,
        dpi=args.dpi,
        seed=args.seed,
        resolution=args.resolution,
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
