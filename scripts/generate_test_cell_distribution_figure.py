#!/usr/bin/env python3
"""
Generate a slide-ready figure showing the *global test set* cell-type distribution.

This intentionally does NOT perform batch correction.

Outputs a single image:
  - Left: UMAP colored by cell type/label
  - Right: Cell-type proportions in the plotted test set

Recommended input for this repo: --data_path data/processed
  - expects: data/processed/global/test.parquet
  - expects: data/processed/genes.txt (optional; used to keep gene order stable)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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


def _infer_obs_key(
    obs_columns: Iterable[str],
    candidates: List[str],
    *,
    override: Optional[str],
    label: str,
) -> str:
    cols = list(obs_columns)
    if override is not None:
        if override in cols:
            return override
        raise KeyError(f"Requested {label} '{override}' not found. Available keys: {cols}")
    for k in candidates:
        if k in cols:
            return k
    raise KeyError(f"Could not infer {label}. Tried: {candidates}. Available keys: {cols}")


def _load_adata_from_processed_dir_test(
    data_dir: Path,
    *,
    max_points: int,
    seed: int,
) -> Tuple["anndata.AnnData", str]:
    import anndata as ad
    import pyarrow.parquet as pq

    parquet_path = data_dir / "global" / "test.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Global test parquet not found: {parquet_path}")

    genes_path = data_dir / "genes.txt"
    gene_cols: Optional[List[str]] = None
    if genes_path.exists():
        gene_cols = [ln.strip() for ln in genes_path.read_text().splitlines() if ln.strip()]
        _log(f"Loaded gene schema: {len(gene_cols)} genes from {genes_path}")

    # Get actual columns from parquet schema
    parquet_schema = pq.read_schema(parquet_path)
    available_cols = set(parquet_schema.names)
    
    meta_candidates = ["id", "sample_id", "client_id", "x", "y", "label"]
    # Only include meta columns that actually exist
    meta_cols_present = [c for c in meta_candidates if c in available_cols]
    
    columns = None
    if gene_cols is not None:
        # Only include gene columns that exist
        gene_cols_present = [c for c in gene_cols if c in available_cols]
        columns = meta_cols_present + gene_cols_present
        if len(gene_cols_present) < len(gene_cols):
            _log(f"[WARNING] {len(gene_cols) - len(gene_cols_present)} genes from schema not in parquet")

    df = pd.read_parquet(parquet_path, columns=columns)
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=seed).reset_index(drop=True)

    if gene_cols is None:
        meta_cols = [c for c in meta_candidates if c in df.columns]
        gene_cols = [c for c in df.columns if c not in meta_cols]
        _log(f"[WARNING] genes.txt not found; inferred {len(gene_cols)} gene columns by exclusion.")
    else:
        # Use the filtered gene columns that exist in parquet
        gene_cols = [c for c in gene_cols if c in df.columns]

    obs_cols = [c for c in df.columns if c not in set(gene_cols)]
    obs = df[obs_cols].copy()

    X = df[gene_cols].to_numpy(dtype=np.float32, copy=True)
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = pd.Index(gene_cols)

    if "id" in adata.obs.columns:
        adata.obs_names = adata.obs["id"].astype(str).values

    return adata, f"Global test parquet ({parquet_path})"


def _load_adata(
    data_path: Path,
    *,
    max_points: int,
    seed: int,
) -> Tuple["anndata.AnnData", str]:
    if data_path.is_dir():
        return _load_adata_from_processed_dir_test(data_path, max_points=max_points, seed=seed)

    if data_path.suffix.lower() == ".h5ad":
        import scanpy as sc

        _log(f"Loading AnnData from {data_path}")
        adata = sc.read_h5ad(str(data_path))
        if adata.n_obs > max_points:
            adata = adata[adata.obs.sample(n=max_points, random_state=seed).index].copy()
        return adata, f"h5ad (n={adata.n_obs:,})"

    raise ValueError(f"Unsupported --data_path: {data_path} (expected directory or .h5ad)")


def _maybe_normalize_and_log1p(adata: "anndata.AnnData") -> None:
    """
    Apply normalize_total+log1p only if data looks count-like.
    This matches the intent of the presentation UMAP script, but keeps this script standalone.
    """
    import scipy.sparse as sp
    import scanpy as sc

    X = adata.X
    if X is None:
        raise ValueError("AnnData.X is None")

    if sp.issparse(X):
        Xs = X[: min(200, X.shape[0]), : min(200, X.shape[1])].toarray()
    else:
        Xs = np.asarray(X[: min(200, X.shape[0]), : min(200, X.shape[1])])

    if Xs.size == 0:
        return

    max_val = float(np.nanmax(Xs))
    is_integer_like = bool(np.allclose(Xs, np.round(Xs), atol=1e-6))

    if is_integer_like and max_val > 20:
        _log("Data appears count-like; applying normalize_total + log1p.")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return

    if max_val > 30:
        _log("Data max>30; applying normalize_total + log1p (heuristic).")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return

    _log("Data appears already normalized/log1p; skipping normalize_total/log1p.")


def _compute_umap(adata_in: "anndata.AnnData", *, seed: int, n_pcs: int) -> "anndata.AnnData":
    import scanpy as sc

    adata = adata_in.copy()
    _maybe_normalize_and_log1p(adata)

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs)
    sc.tl.umap(adata, random_state=seed)
    return adata


def _category_with_optional_other(series: pd.Series, *, max_categories: int) -> Tuple[pd.Series, List[str]]:
    s = series.astype(str).fillna("Unknown")
    counts = s.value_counts()
    if int(counts.size) <= max_categories:
        order = list(counts.index)
        return s, order

    keep = list(counts.iloc[: max_categories - 1].index)
    s2 = s.where(s.isin(keep), other="Other")
    counts2 = s2.value_counts()
    order2 = list(counts2.index)
    return s2, order2


def _make_palette(n: int) -> List[str]:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    colors: List[str] = []
    for cmap_name in ["tab20", "tab20b", "tab20c", "Set3", "Paired", "Accent"]:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            colors.extend([mcolors.to_hex(c) for c in getattr(cmap, "colors")])
        else:
            colors.extend([mcolors.to_hex(cmap(i / 20.0)) for i in range(20)])

    if n > len(colors):
        cmap = plt.get_cmap("hsv")
        extra = [mcolors.to_hex(cmap(i / max(1, n - 1))) for i in range(n - len(colors))]
        colors.extend(extra)

    return colors[:n]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a test-set cell distribution figure (UMAP + proportions), without batch correction."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed",
        help="Processed data directory (recommended) or .h5ad path. For directory, expects global/test.parquet.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="diagnostics/figures/test_cell_distribution_umap.png",
        help="Where to write the figure.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument(
        "--celltype_key",
        type=str,
        default=None,
        help="Optional override for the cell type/label column in adata.obs (defaults to inference).",
    )
    args = parser.parse_args()

    _set_seed(int(args.seed))

    # Some scanpy/numba stacks cache compiled artifacts; keep it inside the repo for portability.
    if "NUMBA_CACHE_DIR" not in os.environ:
        cache_dir = PROJECT_ROOT / ".numba_cache"
        _safe_mkdir(cache_dir)
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)

    data_path = Path(args.data_path)
    out_path = Path(args.out_path)
    _safe_mkdir(out_path.parent)

    adata, source_desc = _load_adata(data_path, max_points=int(args.max_points), seed=int(args.seed))
    _log(f"Loaded test data: {source_desc}")
    _log(f"n_obs={adata.n_obs:,}, n_vars={adata.n_vars:,}")

    celltype_key = _infer_obs_key(
        adata.obs.columns,
        ["cell_type", "celltype", "label", "y", "annotation", "cluster"],
        override=args.celltype_key,
        label="celltype_key",
    )
    _log(f"Using celltype_key: {celltype_key}")

    cell_series = adata.obs[celltype_key]
    if pd.api.types.is_numeric_dtype(cell_series):
        cell_series = cell_series.astype(int).astype(str).radd("L")
    else:
        cell_series = cell_series.astype(str)

    cell_series, order = _category_with_optional_other(cell_series, max_categories=30)
    counts = cell_series.value_counts()
    proportions = counts / max(1, int(counts.sum()))
    _log(f"Cell types: {int(counts.size)} | top10={counts.head(10).to_dict()}")

    max_allowed_pcs = min(int(adata.n_obs), int(adata.n_vars)) - 1
    if max_allowed_pcs < 2:
        raise ValueError(f"Not enough data for PCA/UMAP (n_obs={adata.n_obs}, n_vars={adata.n_vars}).")
    n_pcs = int(min(30, max_allowed_pcs))
    _log(f"UMAP settings: n_pcs={n_pcs}, max_points={args.max_points}, seed={args.seed}")

    adata_umap = _compute_umap(adata, seed=int(args.seed), n_pcs=n_pcs)
    umap_xy = np.asarray(adata_umap.obsm["X_umap"])

    # Keep a stable order: frequency-descending
    order = list(counts.index)
    palette = _make_palette(len(order))
    color_map = {cat: palette[i] for i, cat in enumerate(order)}
    colors = cell_series.map(color_map).values

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.35, 0.65], wspace=0.15)
    ax_umap = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    ax_umap.scatter(
        umap_xy[:, 0],
        umap_xy[:, 1],
        s=2.0,
        c=colors,
        alpha=0.7,
        linewidths=0,
        rasterized=True,
    )
    ax_umap.set_title("UMAP (global test) — colored by cell type", fontsize=14, pad=8)
    ax_umap.set_xticks([])
    ax_umap.set_yticks([])

    # Bar plot as a readable "legend" + distribution summary
    bar_cats = list(counts.index)[::-1]
    bar_vals = [float(proportions[c]) for c in bar_cats]
    bar_cols = [color_map[c] for c in bar_cats]
    ax_bar.barh(bar_cats, bar_vals, color=bar_cols)
    ax_bar.set_title("Cell type proportions (test set)", fontsize=14, pad=8)
    ax_bar.set_xlabel("Proportion")
    ax_bar.set_xlim(0, min(1.0, float(max(bar_vals) * 1.15) if bar_vals else 1.0))
    ax_bar.grid(True, axis="x", alpha=0.25)

    fig.suptitle(
        "Global test set: cell-type distribution (no batch correction)",
        fontsize=18,
        y=0.98,
    )
    fig.text(
        0.01,
        0.01,
        f"Source: {source_desc} | n={adata.n_obs:,} | celltype_key={celltype_key}",
        ha="left",
        va="bottom",
        fontsize=10,
        color="#555555",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()

