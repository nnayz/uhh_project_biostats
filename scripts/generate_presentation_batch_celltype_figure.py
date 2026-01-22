#!/usr/bin/env python3
"""
Generate a slide-ready, multi-panel diagnostic figure:

- UMAP (uncorrected) colored by batch and cell type
- UMAP (corrected) colored by batch and cell type
- Bottom panel: accuracy comparison (centralized vs federated, optional SMPC)

Uncorrected = PCA→UMAP without batch correction.
Corrected   = same pipeline with batch correction using the batch/client key.
Preferred correction methods: Harmony → ComBat → regress_out (fallback).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _set_global_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


def _infer_obs_key(
    obs_columns: Iterable[str],
    candidates: List[str],
    *,
    override: Optional[str] = None,
    required: bool = True,
    label: str = "key",
) -> Optional[str]:
    cols = list(obs_columns)

    if override is not None:
        if override in cols:
            return override
        raise KeyError(
            f"Requested {label} '{override}' not found in obs. "
            f"Available keys: {cols}"
        )

    for k in candidates:
        if k in cols:
            return k

    if required:
        raise KeyError(
            f"Could not infer {label}. Tried: {candidates}. "
            f"Available keys: {cols}"
        )

    return None


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def _load_parquet_sample(
    parquet_path: Path,
    *,
    columns: Optional[List[str]],
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """
    Prefer sampling without reading the entire parquet when possible.
    Falls back to full read if pyarrow isn't available.
    """
    if max_rows <= 0:
        raise ValueError("--max_points must be > 0")

    try:
        import pyarrow.parquet as pq
    except Exception:
        _log("[WARNING] pyarrow.parquet not available; reading full parquet into memory.")
        df = pd.read_parquet(parquet_path, columns=columns)
        if len(df) > max_rows:
            return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        return df.reset_index(drop=True)

    pf = pq.ParquetFile(parquet_path)
    n_rows = pf.metadata.num_rows if pf.metadata is not None else None
    _log(f"Parquet rows: {n_rows if n_rows is not None else 'unknown'}")

    if n_rows is not None and n_rows <= max_rows:
        return pf.read(columns=columns).to_pandas().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    row_groups = list(range(pf.num_row_groups))
    rng.shuffle(row_groups)

    remaining = max_rows
    parts: List[pd.DataFrame] = []

    for rg in row_groups:
        if remaining <= 0:
            break

        df_rg = pf.read_row_group(rg, columns=columns).to_pandas()
        if len(df_rg) > remaining:
            df_rg = df_rg.sample(n=remaining, random_state=seed)
        parts.append(df_rg)
        remaining -= len(df_rg)

    if not parts:
        return pf.read(columns=columns).to_pandas().head(max_rows).reset_index(drop=True)

    return pd.concat(parts, ignore_index=True).reset_index(drop=True)


def _find_preprocessed_parquet(data_path: Path) -> Tuple[Path, str]:
    """
    Resolve a directory containing preprocessed artifacts to a concrete parquet file.
    Priority is chosen to keep memory reasonable for plotting.
    """
    candidates = [
        (data_path / "global" / "test.parquet", "global test set"),
        (data_path / "processed_table.parquet", "full processed table"),
    ]

    for p, desc in candidates:
        if p.exists():
            return p, desc

    # Fallback: try to pool client train/val (read later)
    clients_dir = data_path / "clients"
    if clients_dir.exists():
        client_parquets = sorted(clients_dir.glob("client_*/*.parquet"))
        if client_parquets:
            return client_parquets[0], "client parquet (fallback; will pool)"

    raise FileNotFoundError(
        "Could not locate preprocessed parquet under data_path. "
        "Expected one of: processed_table.parquet, global/test.parquet, or clients/client_*/{train,val}.parquet"
    )


def _load_adata_from_preprocessed_dir(
    data_dir: Path,
    *,
    max_points: int,
    seed: int,
) -> "anndata.AnnData":
    import anndata as ad

    parquet_path, desc = _find_preprocessed_parquet(data_dir)
    _log(f"Using preprocessed data: {desc} ({parquet_path})")

    genes_path = data_dir / "genes.txt"
    gene_cols: Optional[List[str]] = None
    if genes_path.exists():
        gene_cols = [ln.strip() for ln in genes_path.read_text().splitlines() if ln.strip()]
        _log(f"Loaded gene schema: {len(gene_cols)} genes from {genes_path}")

    # Read parquet columns: meta + genes (if available)
    meta_candidates = ["id", "sample_id", "client_id", "x", "y", "label"]
    columns = None
    if gene_cols is not None:
        columns = meta_candidates + gene_cols

    df = _load_parquet_sample(parquet_path, columns=columns, max_rows=max_points, seed=seed)

    # If we used a single client parquet fallback, pool all available client parquets.
    if parquet_path.parent.name.startswith("client_") and parquet_path.parts[-3:-2] == ("clients",):
        _log("[INFO] Pooling all client parquet files for plotting (train/val only).")
        client_parquets = sorted((data_dir / "clients").glob("client_*/*.parquet"))
        frames = []
        for p in client_parquets:
            try:
                frames.append(pd.read_parquet(p, columns=columns))
            except Exception as e:
                _log(f"[WARNING] Skipping {p} ({type(e).__name__}: {e})")
        if not frames:
            raise RuntimeError("Found clients/ but could not read any parquet files.")
        df = pd.concat(frames, ignore_index=True)
        if len(df) > max_points:
            df = df.sample(n=max_points, random_state=seed).reset_index(drop=True)

    if gene_cols is None:
        # Infer gene columns by exclusion
        meta_cols = [c for c in meta_candidates if c in df.columns]
        gene_cols = [c for c in df.columns if c not in meta_cols]
        _log(f"[WARNING] genes.txt not found; inferred {len(gene_cols)} gene columns by exclusion.")

    missing_genes = [g for g in gene_cols if g not in df.columns]
    if missing_genes:
        raise ValueError(f"Missing gene columns in parquet: {missing_genes[:10]}")

    obs_cols = [c for c in df.columns if c not in set(gene_cols)]
    obs = df[obs_cols].copy()

    # Derive a stable client_id if we only have sample_id
    if "client_id" not in obs.columns and "sample_id" in obs.columns:
        uniq = sorted(pd.Series(obs["sample_id"].astype(str)).unique())
        mapping = {v: f"client_{i + 1:02d}" for i, v in enumerate(uniq)}
        obs["client_id"] = obs["sample_id"].astype(str).map(mapping)

    X = df[gene_cols].to_numpy(dtype=np.float32, copy=True)
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = pd.Index(gene_cols)

    if "id" in adata.obs.columns:
        adata.obs_names = adata.obs["id"].astype(str).values

    return adata


def _load_adata(data_path: Path, *, max_points: int, seed: int) -> "anndata.AnnData":
    if data_path.is_dir():
        return _load_adata_from_preprocessed_dir(data_path, max_points=max_points, seed=seed)

    if data_path.suffix.lower() in {".h5ad"}:
        import scanpy as sc

        _log(f"Loading AnnData from {data_path}")

        rng = np.random.default_rng(seed)
        adata_backed = sc.read_h5ad(str(data_path), backed="r")
        _log(f"Loaded (backed): n_obs={adata_backed.n_obs:,}, n_vars={adata_backed.n_vars:,}")

        if adata_backed.n_obs > max_points:
            idx = rng.choice(adata_backed.n_obs, size=max_points, replace=False)
        else:
            idx = np.arange(adata_backed.n_obs)

        # Some anndata/scanpy versions return X=None when calling to_memory() on a backed slice.
        # If that happens, fall back to an in-memory read.
        try:
            adata = adata_backed[idx].to_memory()
            if getattr(adata, "X", None) is None:
                raise RuntimeError("Backed subset .to_memory() returned X=None")
            _log(f"Subsampled to n_obs={adata.n_obs:,} for plotting/computation")
            return adata
        except Exception as e:
            _log(f"[INFO] Backed subset load failed ({type(e).__name__}: {e}); falling back to in-memory read.")

        adata = sc.read_h5ad(str(data_path))
        _log(f"Loaded (in-memory): n_obs={adata.n_obs:,}, n_vars={adata.n_vars:,}")
        if adata.n_obs > max_points:
            adata = adata[idx].copy()
            _log(f"Subsampled to n_obs={adata.n_obs:,} for plotting/computation")
        return adata

    raise ValueError(f"Unsupported --data_path: {data_path} (expected a directory or .h5ad file)")


def _maybe_normalize_and_log1p(adata: "anndata.AnnData") -> Tuple[bool, bool]:
    """
    Returns (did_normalize, did_log1p).
    Uses conservative heuristics to avoid double-normalizing/logging.
    """
    import scanpy as sc
    import scipy.sparse as sp

    if "log1p" in getattr(adata, "uns", {}):
        _log("Detected adata.uns['log1p']; assuming already log1p, skipping normalize_total/log1p.")
        return False, False

    X = adata.X
    if X is None:
        raise ValueError(
            "AnnData.X is None. If you loaded an .h5ad in backed mode, "
            "try pointing --data_path to the preprocessed parquet directory instead."
        )
    if sp.issparse(X):
        Xs = X[: min(200, X.shape[0]), : min(200, X.shape[1])].toarray()
    else:
        Xs = np.asarray(X[: min(200, X.shape[0]), : min(200, X.shape[1])])

    max_val = float(np.nanmax(Xs)) if Xs.size else 0.0
    is_integer_like = bool(np.allclose(Xs, np.round(Xs), atol=1e-6))

    if is_integer_like and max_val > 20:
        _log("Data appears count-like (integer-ish, max>20); applying normalize_total + log1p.")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return True, True

    if max_val > 30:
        _log("Data max>30; applying normalize_total + log1p (heuristic).")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return True, True

    _log("Data appears already normalized/log1p; skipping normalize_total/log1p.")
    return False, False


def _compute_umap_uncorrected(
    adata_in: "anndata.AnnData",
    *,
    seed: int,
    n_pcs: int,
) -> "anndata.AnnData":
    import scanpy as sc

    adata = adata_in.copy()
    _maybe_normalize_and_log1p(adata)

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs)
    sc.tl.umap(adata, random_state=seed)

    return adata


def _compute_umap_corrected(
    adata_in: "anndata.AnnData",
    *,
    batch_key: str,
    seed: int,
    n_pcs: int,
) -> Tuple["anndata.AnnData", str]:
    import scanpy as sc

    adata = adata_in.copy()
    _maybe_normalize_and_log1p(adata)

    # Preferred: Harmony in PCA space
    try:
        import scanpy.external as sce  # type: ignore

        _log("Attempting batch correction: Harmony")
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
        sce.pp.harmony_integrate(
            adata,
            key=batch_key,
            basis="X_pca",
            adjusted_basis="X_pca_harmony",
        )
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca_harmony")
        sc.tl.umap(adata, random_state=seed)
        return adata, "harmony"
    except Exception as e:
        _log(f"[INFO] Harmony not available/failed ({type(e).__name__}: {e}); falling back.")

    # Next: ComBat on expression
    try:
        _log("Attempting batch correction: ComBat")
        # Ensure X is writable for in-place correction
        if getattr(adata, "X", None) is not None:
            adata.X = np.array(adata.X, copy=True)
        sc.pp.combat(adata, key=batch_key)
        # ComBat may leave X non-writeable; copy again before downstream in-place ops
        if getattr(adata, "X", None) is not None:
            adata.X = np.array(adata.X, copy=True)
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs)
        sc.tl.umap(adata, random_state=seed)
        return adata, "combat"
    except Exception as e:
        _log(f"[INFO] ComBat not available/failed ({type(e).__name__}: {e}); falling back.")

    # Minimal fallback: regress out batch if supported
    try:
        _log("Attempting batch correction fallback: regress_out")
        sc.pp.regress_out(adata, keys=[batch_key])
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs)
        sc.tl.umap(adata, random_state=seed)
        return adata, "regress_out"
    except Exception as e:
        _log(f"[WARNING] regress_out failed ({type(e).__name__}: {e}); using PCA-centering fallback.")

    # Last-resort: center PCA by batch and run UMAP from corrected PCs
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    Xp = np.asarray(adata.obsm["X_pca"]).copy()
    batches = pd.Series(adata.obs[batch_key].astype(str)).values
    for b in np.unique(batches):
        m = batches == b
        if m.sum() == 0:
            continue
        Xp[m] = Xp[m] - Xp[m].mean(axis=0, keepdims=True)
    adata.obsm["X_pca_batch_centered"] = Xp
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca_batch_centered")
    sc.tl.umap(adata, random_state=seed)
    return adata, "pca_center"


def _category_with_optional_other(
    series: pd.Series,
    *,
    max_categories: int,
) -> Tuple[pd.Series, Optional[List[str]]]:
    s = series.astype(str).fillna("Unknown")
    uniq = s.nunique(dropna=False)
    if uniq <= max_categories:
        return s, None

    counts = s.value_counts()
    keep = list(counts.iloc[: max_categories - 1].index)
    s2 = s.where(s.isin(keep), other="Other")
    return s2, keep + ["Other"]


def _categorical_colors(n: int) -> List[Tuple[float, float, float, float]]:
    import matplotlib.pyplot as plt

    if n <= 20:
        cm = plt.get_cmap("tab20")
        return [cm(i) for i in range(n)]

    colors: List[Tuple[float, float, float, float]] = []
    for name in ("tab20", "tab20b", "tab20c"):
        cm = plt.get_cmap(name)
        colors.extend([cm(i) for i in range(cm.N)])
    if n <= len(colors):
        return colors[:n]

    cm = plt.get_cmap("hsv")
    return [cm(i / n) for i in range(n)]


def _scatter_umap(
    ax,
    umap_xy: np.ndarray,
    categories: pd.Series,
    *,
    title: Optional[str] = None,
    point_size: float = 2.0,
    alpha: float = 0.6,
) -> Tuple[List, List[str]]:
    import matplotlib.lines as mlines

    cats = pd.Categorical(categories.astype(str))
    ordered = list(cats.categories)

    palette = np.asarray(_categorical_colors(len(ordered)), dtype=float)
    color_map: Dict[str, Tuple[float, float, float, float]] = {
        c: tuple(palette[i]) for i, c in enumerate(ordered)
    }
    codes = cats.codes
    colors = palette[codes]

    ax.scatter(
        umap_xy[:, 0],
        umap_xy[:, 1],
        s=point_size,
        c=colors,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=14, pad=6)

    handles = [
        mlines.Line2D(
            [],
            [],
            color=color_map[c],
            marker="o",
            linestyle="None",
            markersize=6,
            label=c,
        )
        for c in ordered
    ]

    return handles, ordered


def _extract_client_accuracy_from_df(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {}

    client_col = None
    for c in ("client", "client_id", "batch", "site", "donor", "sample_id"):
        if c in df.columns:
            client_col = c
            break

    acc_col = None
    for c in ("accuracy", "acc", "test_accuracy", "global_test_accuracy", "val_accuracy"):
        if c in df.columns:
            acc_col = c
            break

    if client_col is None or acc_col is None:
        return {}

    out: Dict[str, float] = {}
    tmp = df[[client_col, acc_col]].dropna()
    if tmp.empty:
        return {}

    # If there are multiple rows per client, take the last row as the "final" value.
    for client, sub in tmp.groupby(client_col, sort=False):
        try:
            out[str(client)] = float(sub[acc_col].iloc[-1])
        except Exception:
            continue

    return out


def _extract_client_accuracy_from_json(obj: object) -> Dict[str, float]:
    if not isinstance(obj, dict):
        return {}

    out: Dict[str, float] = {}

    # Common nested containers
    for container_key in ("per_client", "by_client", "client_metrics", "client_results", "per_client_metrics"):
        if container_key in obj:
            nested = _extract_client_accuracy_from_json(obj[container_key])
            if nested:
                return nested

    # Direct mapping: {client: accuracy} or {client: {accuracy: ...}}
    for k, v in obj.items():
        if not str(k).startswith("client_"):
            continue
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
        elif isinstance(v, dict) and "accuracy" in v:
            try:
                out[str(k)] = float(v["accuracy"])
            except Exception:
                continue

    return out


def _load_client_accuracy_from_results(results_dir: Path) -> Tuple[Dict[str, float], Optional[Path]]:
    candidates = [
        "per_client_eval.csv",
        "per_client_metrics.csv",
        "client_metrics.csv",
        "eval_by_client.csv",
        "per_client_eval.json",
        "per_client_metrics.json",
        "client_metrics.json",
        "eval_by_client.json",
    ]

    for name in candidates:
        p = results_dir / name
        if not p.exists():
            continue
        try:
            if p.suffix == ".csv":
                df = pd.read_csv(p)
                out = _extract_client_accuracy_from_df(df)
            else:
                out = _extract_client_accuracy_from_json(_read_json(p))
            if out:
                return out, p
        except Exception:
            continue

    return {}, None


def _load_accuracy_data(
    project_root: Path,
) -> Tuple[Optional[float], Optional[float], Optional[float], List[str], Dict[str, float]]:
    cent = _read_json(project_root / "results" / "centralized" / "eval_summary.json")
    fed = _read_json(project_root / "results" / "federated" / "eval_summary.json")

    smpc = None
    for cand in [
        project_root / "results" / "smpc" / "eval_summary.json",
        project_root / "results" / "federated_smpc" / "eval_summary.json",
    ]:
        smpc_json = _read_json(cand)
        if smpc_json is not None:
            smpc = smpc_json
            break

    clients: List[str] = []
    for src in (cent, fed, smpc):
        if isinstance(src, dict) and isinstance(src.get("clients"), list):
            clients = [str(c) for c in src["clients"]]
            break

    cent_acc = None
    if isinstance(cent, dict):
        cent_acc = float(cent.get("global_test", {}).get("accuracy")) if cent.get("global_test") else None

    fed_acc = None
    if isinstance(fed, dict):
        fed_acc = float(fed.get("global_test", {}).get("accuracy")) if fed.get("global_test") else None

    smpc_acc = None
    if isinstance(smpc, dict):
        smpc_acc = float(smpc.get("global_test", {}).get("accuracy")) if smpc.get("global_test") else None

    if smpc is None:
        _log("[INFO] SMPC variant not found in results; skipping.")

    client_acc, source = _load_client_accuracy_from_results(project_root / "results" / "federated")
    if not client_acc:
        client_acc, source = _load_client_accuracy_from_results(project_root / "results" / "centralized")
    if client_acc and source is not None:
        _log(f"[INFO] Loaded per-client accuracies from {source}")
    else:
        _log("[INFO] Per-client accuracies not found; skipping client scatter points.")

    return cent_acc, fed_acc, smpc_acc, clients, client_acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate presentation figure: UMAP before/after batch correction + accuracy panel"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed",
        help="Path to .h5ad OR processed data directory (e.g., data/processed)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="diagnostics/figures/batch_correction_umap_and_accuracy.png",
        help="Output path for the composite figure",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--batch_key", type=str, default=None)
    parser.add_argument("--celltype_key", type=str, default=None)
    args = parser.parse_args()

    _set_global_seed(args.seed)

    out_path = (PROJECT_ROOT / args.out_path).resolve()
    out_dir = out_path.parent
    _safe_mkdir(out_dir)

    # Help scanpy/numba work in environments where site-packages is not writable.
    # Must be set before importing scanpy.
    if "NUMBA_CACHE_DIR" not in os.environ:
        import tempfile

        candidates = [
            Path.home() / ".cache" / "numba",
            Path.home() / ".numba_cache",
            Path(tempfile.gettempdir()) / "numba_cache",
            out_dir / ".numba_cache",
        ]
        for cand in candidates:
            try:
                _safe_mkdir(cand)
                os.environ["NUMBA_CACHE_DIR"] = str(cand)
                _log(f"[INFO] Using NUMBA_CACHE_DIR={cand}")
                break
            except Exception as e:
                _log(f"[INFO] Could not create NUMBA_CACHE_DIR at {cand} ({type(e).__name__}: {e})")

    # Headless-safe backend
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Heavy imports after NUMBA_CACHE_DIR and backend
    import scanpy as sc

    sc.settings.verbosity = 1

    data_path = (PROJECT_ROOT / args.data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"--data_path does not exist: {data_path}")

    if data_path.is_dir():
        gm = _read_json(data_path / "global_metadata.json")
        if isinstance(gm, dict) and "global_test_size" in gm:
            _log(f"Global metadata: global_test_size={gm.get('global_test_size')} | num_clients={gm.get('num_clients')}")

    adata = _load_adata(data_path, max_points=args.max_points, seed=args.seed)
    _log(f"Loaded data for plotting: n_obs={adata.n_obs:,}, n_vars={adata.n_vars:,}")

    batch_key = _infer_obs_key(
        adata.obs.columns,
        ["client", "client_id", "batch", "site", "donor", "sample_id", "library_key"],
        override=args.batch_key,
        label="batch_key",
    )
    celltype_key = _infer_obs_key(
        adata.obs.columns,
        ["cell_type", "celltype", "label", "y", "annotation", "cluster"],
        override=args.celltype_key,
        label="celltype_key",
    )

    _log(f"Using batch_key: {batch_key}")
    _log(f"Using celltype_key: {celltype_key}")

    # Normalize categories for plotting (and optionally group into "Other")
    batch_series = adata.obs[batch_key].astype(str)
    cell_series = adata.obs[celltype_key]
    if pd.api.types.is_numeric_dtype(cell_series):
        cell_series = cell_series.astype(int).astype(str).radd("L")
    else:
        cell_series = cell_series.astype(str)

    cell_series, kept = _category_with_optional_other(cell_series, max_categories=25)
    if kept is not None:
        _log(f"[INFO] Too many cell types; keeping top {len(kept) - 1} by frequency + 'Other'.")

    batch_counts = batch_series.value_counts()
    _log(f"Batches: {int(batch_counts.size)} | counts={batch_counts.to_dict()}")
    cell_counts = cell_series.value_counts()
    _log(f"Cell types: {int(cell_counts.size)} | top10={cell_counts.head(10).to_dict()}")

    # UMAP computation (separate graphs; corrected must not reuse uncorrected neighbors)
    max_allowed_pcs = min(int(adata.n_obs), int(adata.n_vars)) - 1
    if max_allowed_pcs < 2:
        raise ValueError(f"Not enough data for PCA/UMAP (n_obs={adata.n_obs}, n_vars={adata.n_vars}).")
    n_pcs = int(min(30, max_allowed_pcs))
    _log(f"UMAP settings: n_pcs={n_pcs}, max_points={args.max_points}, seed={args.seed}")

    adata_unc = _compute_umap_uncorrected(adata, seed=args.seed, n_pcs=n_pcs)
    adata_cor, correction_method = _compute_umap_corrected(
        adata, batch_key=batch_key, seed=args.seed, n_pcs=n_pcs
    )
    _log(f"Batch correction method used: {correction_method}")

    # Build composite figure: 2x2 UMAP grid + bottom accuracy panel
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.0, 1.0, 0.6], hspace=0.18, wspace=0.08)

    ax_ub = fig.add_subplot(gs[0, 0])
    ax_uc = fig.add_subplot(gs[1, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    ax_cc = fig.add_subplot(gs[1, 1])
    ax_acc = fig.add_subplot(gs[2, :])

    handles_batch, labels_batch = _scatter_umap(
        ax_ub,
        np.asarray(adata_unc.obsm["X_umap"]),
        batch_series,
        title="a. Uncorrected",
        point_size=2.0,
    )
    handles_cell, labels_cell = _scatter_umap(
        ax_uc,
        np.asarray(adata_unc.obsm["X_umap"]),
        cell_series,
        title=None,
        point_size=2.0,
    )

    _scatter_umap(
        ax_cb,
        np.asarray(adata_cor.obsm["X_umap"]),
        batch_series,
        title="b. Corrected",
        point_size=2.0,
    )
    _scatter_umap(
        ax_cc,
        np.asarray(adata_cor.obsm["X_umap"]),
        cell_series,
        title=None,
        point_size=2.0,
    )

    # Row labels (left side), like the reference layout
    fig.text(0.015, 0.78, "Batch", rotation=90, va="center", ha="left", fontsize=14)
    fig.text(0.015, 0.48, "Cell type", rotation=90, va="center", ha="left", fontsize=14)

    # Accuracy panel (d)
    cent_acc, fed_acc, smpc_acc, clients, client_acc = _load_accuracy_data(PROJECT_ROOT)

    client_order = clients if clients else sorted(client_acc.keys())

    if client_order:
        x = np.arange(len(client_order))
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels([c.replace("client_", "Client ") for c in client_order], rotation=0)
        ax_acc.set_xlim(-0.5, len(client_order) - 0.5)
    else:
        ax_acc.set_xticks([])
        ax_acc.set_xlim(0, 1)

    if client_acc and client_order:
        xs: List[float] = []
        ys: List[float] = []
        for i, c in enumerate(client_order):
            if c in client_acc:
                xs.append(float(i))
                ys.append(float(client_acc[c]))
        if xs:
            ax_acc.scatter(
                xs,
                ys,
                s=60,
                color="#2ca02c",
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
                label="Clients (per-client)",
            )

    if cent_acc is not None:
        ax_acc.axhline(cent_acc, linestyle="--", color="black", linewidth=2, label="Centralized (global test)")
    if fed_acc is not None:
        ax_acc.axhline(fed_acc, linestyle="-", color="#1f77b4", linewidth=2, label="Federated (global test)")
    if smpc_acc is not None:
        ax_acc.axhline(smpc_acc, linestyle="-", color="#ff7f0e", linewidth=2, label="Federated+SMPC (global test)")

    ax_acc.set_title("d. Accuracy", fontsize=14, pad=8)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, axis="y", alpha=0.25)
    ax_acc.legend(loc="lower right", frameon=False)

    # Legends to the right, outside plot area
    fig.legend(
        handles_batch,
        labels_batch,
        loc="upper left",
        bbox_to_anchor=(0.83, 0.88),
        title="Batch",
        frameon=False,
    )
    fig.legend(
        handles_cell,
        labels_cell,
        loc="upper left",
        bbox_to_anchor=(0.83, 0.52),
        title="Cell type",
        ncol=(1 if len(labels_cell) <= 20 else 2),
        prop={"size": 9},
        frameon=False,
    )

    fig.subplots_adjust(left=0.05, right=0.82, top=0.96, bottom=0.07)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    _log(f"Saved composite figure to: {out_path}")

    # Optional individual UMAP outputs
    umap_unc_path = out_dir / "umap_uncorrected.png"
    umap_cor_path = out_dir / "umap_corrected.png"

    def _save_single(umap_xy: np.ndarray, title: str, outp: Path) -> None:
        f = plt.figure(figsize=(10, 10))
        g = f.add_gridspec(nrows=2, ncols=1, hspace=0.08)
        a1 = f.add_subplot(g[0, 0])
        a2 = f.add_subplot(g[1, 0])
        _scatter_umap(a1, umap_xy, batch_series, title=title, point_size=2.0)
        _scatter_umap(a2, umap_xy, cell_series, title=None, point_size=2.0)
        f.text(0.02, 0.74, "Batch", rotation=90, va="center", ha="left", fontsize=12)
        f.text(0.02, 0.28, "Cell type", rotation=90, va="center", ha="left", fontsize=12)
        f.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.06)
        f.savefig(outp, dpi=300)
        plt.close(f)

    _save_single(np.asarray(adata_unc.obsm["X_umap"]), "Uncorrected UMAP", umap_unc_path)
    _save_single(np.asarray(adata_cor.obsm["X_umap"]), "Corrected UMAP", umap_cor_path)
    _log(f"Saved: {umap_unc_path}")
    _log(f"Saved: {umap_cor_path}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
