"""
Visualization for batch-based evaluation (advisor feedback).

- UMAPs: colored by batch ID; colored by cell type
- Scatter: final evaluation performance by model type (centralized, federated, batch-wise/local)
  on the held-out batch only

Usage:
  python scripts/milestone3/visualize_evaluation.py --data_dir data/processed --results_dir results
  python scripts/milestone3/visualize_evaluation.py --data_dir data/processed --evaluation_summary results/comparison/evaluation_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_processed_and_umap(data_dir: str, max_cells: Optional[int] = 50000):
    """Load processed table and compute UMAP if not present."""
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for UMAP; install with: pip install scanpy")

    table_path = Path(data_dir) / "processed_table.parquet"
    if not table_path.exists():
        raise FileNotFoundError(f"Processed table not found: {table_path}")

    df = pd.read_parquet(table_path)
    if max_cells and len(df) > max_cells:
        df = df.sample(n=max_cells, random_state=42).reset_index(drop=True)

    # Build minimal AnnData for UMAP
    genes_path = Path(data_dir) / "genes.txt"
    genes = [line.strip() for line in open(genes_path) if line.strip()]
    X = df[genes].values.astype(np.float32)
    adata = sc.AnnData(X)
    adata.obs = df[["id", "sample_id", "x", "y", "label"]].copy()
    if "batch_id" in df.columns:
        adata.obs["batch_id"] = df["batch_id"].values
    if "cell_type" in df.columns:
        adata.obs["cell_type"] = df["cell_type"].values
    adata.var_names = genes

    sc.pp.pca(adata, n_comps=min(30, adata.n_vars - 1, adata.n_obs - 1))
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=adata.obsm["X_pca"].shape[1])
    sc.tl.umap(adata)
    return adata


def plot_umap_batch(adata, output_path: Path):
    """UMAP colored by batch ID."""
    import scanpy as sc
    fig, ax = plt.subplots(figsize=(8, 6))
    batch_col = "batch_id" if "batch_id" in adata.obs.columns else "sample_id"
    sc.pl.umap(adata, color=batch_col, ax=ax, show=False, legend_loc="on data", legend_fontoutline=2)
    ax.set_title("UMAP colored by batch ID")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_umap_celltype(adata, output_path: Path):
    """UMAP colored by cell type (if present)."""
    import scanpy as sc
    if "cell_type" not in adata.obs.columns:
        print("No 'cell_type' in data; skipping UMAP by cell type")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata, color="cell_type", ax=ax, show=False, legend_loc="on data", legend_fontoutline=2)
    ax.set_title("UMAP colored by cell type")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_eval_scatter(
    evaluation_summary: Dict,
    output_path: Path,
    include_local: bool = True,
):
    """
    Scatter: different markers for model types (centralized, federated, batch-wise/local),
    reporting performance on the held-out batch only.
    """
    metrics = []
    # Global test = held-out batch metrics
    if "centralized" in evaluation_summary and "global_test" in evaluation_summary["centralized"]:
        gt = evaluation_summary["centralized"]["global_test"]
        metrics.append({"model": "Centralized", "accuracy": gt.get("accuracy"), "f1_macro": gt.get("f1_macro"), "loss": gt.get("loss")})
    if "federated" in evaluation_summary and "global_test" in evaluation_summary["federated"]:
        gt = evaluation_summary["federated"]["global_test"]
        metrics.append({"model": "Federated", "accuracy": gt.get("accuracy"), "f1_macro": gt.get("f1_macro"), "loss": gt.get("loss")})
    if include_local and "local" in evaluation_summary and evaluation_summary["local"]:
        for client_id, ev in evaluation_summary["local"].items():
            gt = ev.get("global_test", {})
            metrics.append({"model": f"Local ({client_id})", "accuracy": gt.get("accuracy"), "f1_macro": gt.get("f1_macro"), "loss": gt.get("loss")})

    if not metrics:
        print("No metrics found for scatter plot")
        return

    df = pd.DataFrame(metrics)
    markers = {"Centralized": "o", "Federated": "s", "Local": "^"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        m = "o" if "Centralized" in model else ("s" if "Federated" in model else "^")
        ax.scatter(sub["accuracy"], sub["f1_macro"], label=model, marker=m, s=120, alpha=0.8)
    ax.set_xlabel("Accuracy (held-out batch)")
    ax.set_ylabel("F1-Macro (held-out batch)")
    ax.set_title("Final evaluation on held-out batch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="UMAP and evaluation scatter plots")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--evaluation_summary", type=str, default=None)
    parser.add_argument("--max_cells", type=int, default=50000)
    parser.add_argument("--skip_umap", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir or args.results_dir) / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_umap:
        try:
            import scanpy as sc
        except ImportError:
            print("scanpy not installed; skipping UMAP")
            args.skip_umap = True

    if not args.skip_umap:
        print("Loading data and computing UMAP...")
        adata = load_processed_and_umap(args.data_dir, max_cells=args.max_cells)
        plot_umap_batch(adata, out_dir / "umap_batch_id.png")
        plot_umap_celltype(adata, out_dir / "umap_cell_type.png")

    summary_path = Path(args.evaluation_summary or Path(args.results_dir) / "comparison" / "evaluation_summary.json")
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        plot_final_eval_scatter(summary, out_dir / "final_eval_held_out_batch.png", include_local=True)
    else:
        print(f"Evaluation summary not found: {summary_path}; skipping scatter plot")

    print("Done. Outputs in", out_dir)


if __name__ == "__main__":
    main()
