"""
Single analysis script after partitioning (anatomical siloing): client statistics,
non-IID metrics, UMAPs, and plots for dataset/split/client/label in context of
central vs federated vs local training.

Run after: preprocess.py and partition_anatomical_siloing.py.

Outputs (default outputs/analysis/):
  - CSVs: client_summary.csv, client_noniid_metrics.csv, client_label_probabilities.csv
  - Plots: client_sizes, train_val_per_client, split_overview (held-out + clients),
    client_imbalance_max_fraction, client_jsd_to_global, label_proportion_heatmap,
    global_label_distribution, umap_by_client, umap_by_label
  - analysis_summary.md (with interpretation)

Usage:
  python scripts/analysis/run_analysis.py
  python scripts/analysis/run_analysis.py --data_dir data/processed --out_dir outputs/analysis --max_cells_umap 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        a = a[a > 0]
        b = b[: len(a)]
        return float((a * np.log(a / (b + 1e-12))).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    parser = argparse.ArgumentParser(description="Client statistics, non-IID metrics, UMAPs, split/label plots")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Processed data dir")
    parser.add_argument("--out_dir", type=str, default="outputs/analysis", help="Output directory")
    parser.add_argument("--max_cells_umap", type=int, default=50000, help="Max cells for UMAP (subsample if larger)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clients_dir = data_dir / "clients"
    client_dirs = sorted([d for d in clients_dir.glob("client_*") if d.is_dir()])
    if not client_dirs:
        raise RuntimeError(f"No clients found in {clients_dir}")

    # ----- 1. Client statistics from client_meta.json -----
    rows = []
    all_label_ids = set()
    client_label_counts = {}

    for cdir in client_dirs:
        cname = cdir.name
        with open(cdir / "client_meta.json", "r") as f:
            meta = json.load(f)

        counts = meta.get("label_counts_total", {})
        counts = {int(k): int(v) for k, v in counts.items()}
        client_label_counts[cname] = counts
        all_label_ids.update(counts.keys())

        n_total = meta["n_total"]
        n_train = meta["n_train"]
        n_val = meta["n_val"]
        n_test = meta.get("n_test", 0)
        max_frac = max(counts.values()) / max(1, sum(counts.values())) if counts else 0
        n_classes = len(counts)
        group_val = meta.get("group_value", meta.get("batch_id", ""))

        rows.append({
            "client": cname,
            "group_value": group_val,
            "n_total": n_total,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_classes_present": n_classes,
            "max_label_fraction": round(max_frac, 4),
        })

    summary_df = pd.DataFrame(rows).sort_values("client")
    summary_df.to_csv(out_dir / "client_summary.csv", index=False)

    label_ids = sorted(all_label_ids)
    dist_mat = np.array([[client_label_counts[c].get(l, 0) for l in label_ids] for c in summary_df["client"]], dtype=float)
    prob_mat = dist_mat / (dist_mat.sum(axis=1, keepdims=True) + 1e-12)
    global_prob = dist_mat.sum(axis=0) / (dist_mat.sum() + 1e-12)
    global_counts = dist_mat.sum(axis=0)

    non_iid_rows = []
    for i, cname in enumerate(summary_df["client"]):
        p = prob_mat[i]
        non_iid_rows.append({
            "client": cname,
            "entropy": round(entropy(p), 4),
            "js_divergence_to_global": round(js_divergence(p, global_prob), 4),
        })
    non_iid_df = pd.DataFrame(non_iid_rows).merge(summary_df, on="client")
    non_iid_df.to_csv(out_dir / "client_noniid_metrics.csv", index=False)

    prob_df = pd.DataFrame(prob_mat, columns=[f"label_{l}" for l in label_ids])
    prob_df.insert(0, "client", list(summary_df["client"]))
    prob_df.to_csv(out_dir / "client_label_probabilities.csv", index=False)

    # ----- 2. Held-out size (for split overview) -----
    held_out_n = 0
    held_out_meta_path = data_dir / "held_out_batch_meta.json"
    if held_out_meta_path.exists():
        with open(held_out_meta_path, "r") as f:
            held_out_meta = json.load(f)
        held_out_n = held_out_meta.get("n_samples", 0)

    # Combined labels (client01_dorsal, client02_mid, client03_ventral) for graph axes
    def _client_display_label(row):
        cid = row["client"]
        short_id = cid.replace("client_", "client") if str(cid).startswith("client_") else cid
        region = row.get("group_value") or cid
        return f"{short_id}_{region}"
    client_labels = summary_df.apply(_client_display_label, axis=1).tolist()

    # ----- 3. Plots: sizes, train/val, split overview, imbalance, JSD -----
    plt.figure(figsize=(7, 4))
    plt.bar(client_labels, summary_df["n_total"], color="steelblue", edgecolor="black")
    plt.title("Client sizes (total samples per client)")
    plt.ylabel("Number of samples")
    plt.xlabel("Client (anatomical region)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "client_sizes.png", dpi=200)
    plt.close()

    # Train vs val per client (grouped bar)
    x = np.arange(len(summary_df))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, summary_df["n_train"], w, label="Train (80%)", color="steelblue")
    ax.bar(x + w / 2, summary_df["n_val"], w, label="Val (20%)", color="coral", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(client_labels)
    ax.set_ylabel("Number of samples")
    ax.set_title("Train vs validation split per client")
    ax.legend(loc="upper right")
    ax.set_xlabel("Client (anatomical region)")
    plt.tight_layout()
    plt.savefig(out_dir / "train_val_per_client.png", dpi=200)
    plt.close()

    # Split overview: held-out + clients (what central / fed / local see)
    names = ["Held-out\n(eval only)"] + list(client_labels)
    counts_split = [held_out_n] + list(summary_df["n_total"])
    colors = ["gray"] + ["steelblue"] * len(summary_df)
    plt.figure(figsize=(7, 4))
    plt.bar(names, counts_split, color=colors, edgecolor="black")
    plt.title("Data split: held-out (eval) vs clients (training)")
    plt.ylabel("Number of samples")
    plt.xlabel("Split")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "split_overview.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(client_labels, summary_df["max_label_fraction"], color="coral", edgecolor="black")
    plt.title("Imbalance per client (max label fraction)")
    plt.ylabel("Max label fraction")
    plt.xlabel("Client (anatomical region)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "client_imbalance_max_fraction.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(client_labels, non_iid_df["js_divergence_to_global"], color="seagreen", edgecolor="black")
    plt.title("Non-IID severity (Jensen–Shannon divergence to global)")
    plt.ylabel("JSD (nats)")
    plt.xlabel("Client (anatomical region)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "client_jsd_to_global.png", dpi=200)
    plt.close()

    # Label proportion heatmap (clients x labels) — y-axis = anatomical region
    fig, ax = plt.subplots(figsize=(max(8, len(label_ids) * 0.3), 4))
    im = ax.imshow(prob_mat, aspect="auto", cmap="viridis", vmin=0, vmax=0.2)
    ax.set_xticks(np.arange(len(label_ids)))
    ax.set_xticklabels(label_ids, rotation=90)
    ax.set_yticks(np.arange(len(summary_df)))
    ax.set_yticklabels(client_labels)
    ax.set_xlabel("Label (Leiden cluster)")
    ax.set_ylabel("Client")
    ax.set_title("Label proportion per client (non-IID: different clients have different mixes)")
    plt.colorbar(im, ax=ax, label="Proportion")
    plt.tight_layout()
    plt.savefig(out_dir / "label_proportion_heatmap.png", dpi=200)
    plt.close()

    # Global label distribution
    plt.figure(figsize=(max(8, len(label_ids) * 0.35), 4))
    plt.bar(np.arange(len(label_ids)), global_prob, color="steelblue", edgecolor="black")
    plt.xticks(np.arange(len(label_ids)), label_ids, rotation=90)
    plt.xlabel("Label (Leiden cluster)")
    plt.ylabel("Proportion (global)")
    plt.title("Global label distribution (pooled over all clients)")
    plt.tight_layout()
    plt.savefig(out_dir / "global_label_distribution.png", dpi=200)
    plt.close()

    # ----- 4. UMAP with clear legends -----
    try:
        import scanpy as sc
    except ImportError:
        print("  [Skip] scanpy not installed; UMAPs not generated.")
    else:
        genes_path = data_dir / "genes.txt"
        if not genes_path.exists():
            print("  [Skip] genes.txt not found; UMAPs not generated.")
        else:
            genes = [line.strip() for line in open(genes_path) if line.strip()]
            dfs = []
            for cdir in client_dirs:
                for split in ("train", "val"):
                    p = cdir / f"{split}.parquet"
                    if p.exists():
                        dfs.append(pd.read_parquet(p))
            if not dfs:
                print("  [Skip] No client train/val parquets found.")
            else:
                df = pd.concat(dfs, ignore_index=True)
                if len(df) > args.max_cells_umap:
                    df = df.sample(n=args.max_cells_umap, random_state=42).reset_index(drop=True)
                X = df[genes].values.astype(np.float32)
                adata = sc.AnnData(X)
                adata.obs = df[["id", "sample_id", "batch_id", "x", "y", "label"]].copy()
                adata.var_names = genes

                sc.pp.pca(adata, n_comps=min(30, adata.n_vars - 1, adata.n_obs - 1))
                sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(30, adata.obsm["X_pca"].shape[1]))
                sc.tl.umap(adata)

                # UMAP by client: legend on the right for readability
                fig, ax = plt.subplots(figsize=(10, 6))
                sc.pl.umap(
                    adata, color="batch_id", ax=ax, show=False,
                    legend_loc="right margin", legend_fontsize=10, legend_fontoutline=2,
                    title="UMAP colored by client (anatomical region)",
                )
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                plt.tight_layout()
                plt.savefig(out_dir / "umap_by_client.png", dpi=150, bbox_inches="tight")
                plt.close()

                # UMAP by label: legend on the right (many categories)
                adata.obs["label_str"] = adata.obs["label"].astype(str)
                fig, ax = plt.subplots(figsize=(10, 6))
                sc.pl.umap(
                    adata, color="label_str", ax=ax, show=False,
                    legend_loc="right margin", legend_fontsize=8, legend_fontoutline=2,
                    title="UMAP colored by label (Leiden cluster / cell type)",
                )
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                plt.tight_layout()
                plt.savefig(out_dir / "umap_by_label.png", dpi=150, bbox_inches="tight")
                plt.close()

    # ----- 5. analysis_summary.md with interpretation -----
    md_path = out_dir / "analysis_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset & Client Analysis (Post-Partitioning)\n\n")
        f.write("## Outputs\n\n")
        f.write("- **CSVs:** `client_summary.csv`, `client_noniid_metrics.csv`, `client_label_probabilities.csv`\n")
        f.write("- **Plots:** client_sizes, train_val_per_client, split_overview, client_imbalance_max_fraction, ")
        f.write("client_jsd_to_global, label_proportion_heatmap, global_label_distribution, umap_by_client, umap_by_label\n\n")
        f.write("## How to interpret (central vs federated vs local)\n\n")
        f.write("- **split_overview.png** — Shows held-out set (eval only) vs each client’s total size. Central training pools all client data; federated sees each client separately; local sees one client only. All models are evaluated on the held-out set.\n\n")
        f.write("- **train_val_per_client.png** — 80% train / 20% val within each client. Confirms the split used for training and validation.\n\n")
        f.write("- **client_sizes.png** — Number of samples per client. Affects federated aggregation (e.g. FedAvg) and local training capacity.\n\n")
        f.write("- **label_proportion_heatmap.png** — Proportion of each label per client. Different columns (clients) having different patterns = non-IID; explains why federated and local can diverge from central.\n\n")
        f.write("- **global_label_distribution.png** — Pooled label distribution (what central training sees on average).\n\n")
        f.write("- **client_jsd_to_global.png** — Jensen–Shannon divergence of each client’s label distribution to the global one. Higher = more non-IID.\n\n")
        f.write("- **client_imbalance_max_fraction.png** — Within-client class imbalance (max fraction in one label).\n\n")
        f.write("- **umap_by_client.png** — UMAP of client data colored by anatomical region (dorsal/mid/ventral). Legend: client identity.\n\n")
        f.write("- **umap_by_label.png** — Same UMAP colored by Leiden cluster (cell type). Legend: label IDs.\n\n")
        f.write("## Summary table (per client)\n\n")
        f.write(summary_df.to_string(index=False) + "\n\n")
        f.write("## Non-IID metrics (JSD to global)\n\n")
        f.write(non_iid_df[["client", "group_value", "n_total", "entropy", "js_divergence_to_global"]].to_string(index=False) + "\n")
    print(f"  {md_path}")

    print("\nDone. Outputs in:", out_dir)


if __name__ == "__main__":
    main()
