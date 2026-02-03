"""
Preprocess SpatialCorpus .h5ad into a single table with batch IDs and labels.

Mouse brain dataset is small enough to load and process in memory: filter, normalize,
PCA + Leiden clustering, then write one parquet.

Usage:
  python scripts/data_preparation/preprocess.py
  python scripts/data_preparation/preprocess.py --raw_path data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad --batch_col library_key
"""

import os
import json
import argparse
import gc
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

OUT_DIR = "data/processed"
DEFAULT_RAW = "data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad"
DEFAULT_BATCH_COL = "library_key"
MIN_COUNTS = 10
N_PCS = 30
LEIDEN_RESOLUTION = 0.5


def detect_batch_column(adata):
    """Try common batch-related column names if not specified (e.g. library_key for mouse brain)."""
    for c in ["library_key", "replicate", "section", "donor_id", "donor", "condition_id", "dataset", "sample_id", "dataset_id", "batch", "sample"]:
        if c in adata.obs.columns:
            return c
    raise ValueError("No batch column found. obs: " + ", ".join(adata.obs.columns[:20].tolist()))


def resolve_spatial_columns(adata):
    """Return (x_col, y_col) from obs. Prefer 'x'/'y', then 'x_centroid'/'y_centroid' (mouse brain)."""
    if "x" in adata.obs.columns and "y" in adata.obs.columns:
        return "x", "y"
    if "x_centroid" in adata.obs.columns and "y_centroid" in adata.obs.columns:
        return "x_centroid", "y_centroid"
    raise ValueError("No spatial columns found. Need obs['x']/['y'] or obs['x_centroid']/['y_centroid']")


def main():
    parser = argparse.ArgumentParser(description="Preprocess SpatialCorpus h5ad for FL (mouse brain)")
    parser.add_argument("--raw_path", type=str, default=DEFAULT_RAW, help="Path to raw .h5ad")
    parser.add_argument("--batch_col", type=str, default=None, help="obs column for batch ID (default: auto-detect)")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR, help="Output directory")
    parser.add_argument("--min_counts", type=int, default=MIN_COUNTS, help="Min total counts per cell")
    args = parser.parse_args()

    raw_path = args.raw_path
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data...")
    adata = sc.read_h5ad(raw_path)
    print(f"  Cells: {adata.n_obs:,}, Genes: {adata.n_vars}")

    batch_col = args.batch_col or detect_batch_column(adata)
    x_col, y_col = resolve_spatial_columns(adata)
    if batch_col not in adata.obs.columns:
        raise ValueError(f"Batch column '{batch_col}' not in obs: {list(adata.obs.columns)}")
    print(f"  Batch column: {batch_col} (unique: {adata.obs[batch_col].nunique()})")
    print(f"  Spatial columns: {x_col}, {y_col}")

    print("Filtering cells (min_counts)...")
    sc.pp.filter_cells(adata, min_counts=args.min_counts, inplace=True)
    print(f"  After filtering: {adata.n_obs:,} cells")

    print("Normalizing...")
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata, copy=False)

    print("PCA + neighbors + Leiden...")
    n_comps = min(N_PCS, adata.n_vars - 1, adata.n_obs - 1)
    n_pcs_use = min(30, n_comps)
    for step_name, step_fn in tqdm([
        ("PCA", lambda: sc.pp.pca(adata, n_comps=n_comps)),
        ("Neighbors", lambda: sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs_use)),
        ("Leiden", lambda: sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added="cluster")),
    ], desc="PCA + neighbors + Leiden"):
        step_fn()

    labels = adata.obs["cluster"].astype(str)
    uniq = sorted(labels.unique())
    label_map = {lab: i for i, lab in enumerate(uniq)}
    with open(os.path.join(args.out_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    encoded_labels = labels.map(label_map).astype("int32")
    print(f"  Clusters: {len(uniq)}")

    gene_names = list(adata.var_names)
    with open(os.path.join(args.out_dir, "genes.txt"), "w") as f:
        f.write("\n".join(gene_names))

    # Build expression matrix (chunked if large, with progress)
    n_cells = adata.n_obs
    chunk_size = 50_000
    chunks = []
    for start in tqdm(range(0, n_cells, chunk_size), desc="Building expression matrix"):
        end = min(start + chunk_size, n_cells)
        X_chunk = adata.X[start:end]
        if hasattr(X_chunk, "toarray"):
            X_chunk = np.asarray(X_chunk.toarray(), dtype=np.float32)
        else:
            X_chunk = np.asarray(X_chunk, dtype=np.float32)
        chunks.append(pd.DataFrame(X_chunk, columns=gene_names))
    expr_df = pd.concat(chunks, axis=0, ignore_index=True)
    del chunks
    gc.collect()

    meta_df = pd.DataFrame({
        "id": adata.obs_names.astype(str),
        "sample_id": adata.obs[batch_col].astype(str).values,
        "batch_id": adata.obs[batch_col].astype(str).values,
        "x": adata.obs[x_col].astype("float32").values,
        "y": adata.obs[y_col].astype("float32").values,
        "label": encoded_labels.values,
    })
    if "cell_type" in adata.obs.columns:
        meta_df["cell_type"] = adata.obs["cell_type"].astype(str).values
    if "organism" in adata.obs.columns:
        meta_df["organism"] = adata.obs["organism"].astype(str).values

    out = pd.concat([meta_df, expr_df], axis=1)
    out_path = os.path.join(args.out_dir, "processed_table.parquet")
    out.to_parquet(out_path, index=False)

    with open(os.path.join(args.out_dir, "preprocess_config.json"), "w") as f:
        json.dump({
            "raw_path": raw_path,
            "batch_col": batch_col,
            "n_batches": int(meta_df["batch_id"].nunique()),
            "batch_ids": sorted(meta_df["batch_id"].unique().tolist()),
        }, f, indent=2)

    print("Done.")
    print(f"  Output: {out_path} ({len(out):,} rows, {len(gene_names)} genes, {meta_df['batch_id'].nunique()} batches)")


if __name__ == "__main__":
    main()
