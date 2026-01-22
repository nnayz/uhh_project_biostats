import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

CLIENTS_DIR = os.path.join("data", "processed", "clients")
OUT_DIR = os.path.join("md", "figures")
SUMMARY_MD = os.path.join("md", "client_stats_summary.md")

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Global test set info (new split)
# -----------------------------------------------------------------------------

GLOBAL_TEST_PATH = os.path.join("data", "processed", "global", "test.parquet")
global_test_size = None

if os.path.exists(GLOBAL_TEST_PATH):
    try:
        global_test_size = len(pd.read_parquet(GLOBAL_TEST_PATH))
    except Exception as e:
        print(f"[WARNING] Could not read global test set: {e}")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def safe_read_meta(meta_path):
    with open(meta_path, "r") as f:
        return json.load(f)

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
        b = b[:len(a)]
        return float((a * np.log(a / (b + 1e-12))).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

# -----------------------------------------------------------------------------
# Discover clients
# -----------------------------------------------------------------------------

client_dirs = sorted(
    [d for d in glob.glob(os.path.join(CLIENTS_DIR, "client_*")) if os.path.isdir(d)]
)
if not client_dirs:
    raise RuntimeError(f"No clients found in {CLIENTS_DIR}")

# -----------------------------------------------------------------------------
# Load per-client metadata
# -----------------------------------------------------------------------------

rows = []
all_label_ids = set()
client_label_counts = {}

for cdir in client_dirs:
    cname = os.path.basename(cdir)
    meta = safe_read_meta(os.path.join(cdir, "client_meta.json"))

    counts = {int(k): int(v) for k, v in meta.get("label_counts_total", {}).items()}
    client_label_counts[cname] = counts
    all_label_ids.update(counts.keys())

    max_frac = max(counts.values()) / max(1, sum(counts.values()))
    min_count = min(counts.values()) if counts else 0

    rows.append({
        "client": cname,
        "group_value": meta.get("group_value", ""),
        "n_total": meta["n_total"],
        "n_train": meta["n_train"],
        "n_val": meta["n_val"],
        "n_test": 0,  # test is global
        "n_classes_present": len(counts),
        "max_label_fraction": max_frac,
        "min_label_count": min_count,
    })

summary_df = pd.DataFrame(rows).sort_values("client")
summary_csv = os.path.join(OUT_DIR, "client_summary.csv")
summary_df.to_csv(summary_csv, index=False)

# -----------------------------------------------------------------------------
# Label distributions
# -----------------------------------------------------------------------------

label_ids = sorted(all_label_ids)

dist_mat = np.vstack([
    np.array([client_label_counts[c].get(l, 0) for l in label_ids], dtype=float)
    for c in summary_df["client"]
])

prob_mat = dist_mat / (dist_mat.sum(axis=1, keepdims=True) + 1e-12)

global_prob = dist_mat.sum(axis=0)
global_prob = global_prob / (global_prob.sum() + 1e-12)

# -----------------------------------------------------------------------------
# Non-IID metrics
# -----------------------------------------------------------------------------

non_iid_rows = []
for i, cname in enumerate(summary_df["client"]):
    non_iid_rows.append({
        "client": cname,
        "entropy": entropy(prob_mat[i]),
        "js_divergence_to_global": js_divergence(prob_mat[i], global_prob),
    })

non_iid_df = pd.DataFrame(non_iid_rows).merge(summary_df, on="client")
non_iid_csv = os.path.join(OUT_DIR, "client_noniid_metrics.csv")
non_iid_df.to_csv(non_iid_csv, index=False)

# -----------------------------------------------------------------------------
# PLOTS (existing)
# -----------------------------------------------------------------------------

# 1) Client sizes
plt.figure()
plt.bar(summary_df["client"], summary_df["n_total"])
plt.title("Client sizes (n_total)")
plt.ylabel("Number of samples")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "client_sizes.png"), dpi=200)
plt.close()

# 2) Max label fraction
plt.figure()
plt.bar(summary_df["client"], summary_df["max_label_fraction"])
plt.title("Imbalance per client (max label fraction)")
plt.ylabel("Max label fraction")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "client_imbalance_max_fraction.png"), dpi=200)
plt.close()

# 3) JSD to global
plt.figure()
plt.bar(non_iid_df["client"], non_iid_df["js_divergence_to_global"])
plt.title("Non-IID severity (Jensen–Shannon divergence)")
plt.ylabel("JSD (nats)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "client_jsd_to_global.png"), dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# NEW FIGURE 4 — Heatmap of label proportions (MAIN advisor figure)
# -----------------------------------------------------------------------------

plt.figure(figsize=(0.6 * len(label_ids) + 4, 0.6 * len(summary_df) + 2))
sns.heatmap(
    prob_mat,
    cmap="viridis",
    xticklabels=[f"L{l}" for l in label_ids],
    yticklabels=summary_df["client"],
    cbar_kws={"label": "Label proportion"},
)
plt.title("Label distribution per client (proportion heatmap)")
plt.xlabel("Label")
plt.ylabel("Client")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "client_label_distribution_heatmap.png"), dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# NEW FIGURE 5 — Variance of label proportions across clients
# -----------------------------------------------------------------------------

prob_long = []
for i, cname in enumerate(summary_df["client"]):
    for j, lid in enumerate(label_ids):
        prob_long.append({
            "client": cname,
            "label": f"L{lid}",
            "proportion": prob_mat[i, j],
        })

prob_long_df = pd.DataFrame(prob_long)

plt.figure(figsize=(max(8, 0.6 * len(label_ids)), 5))
sns.violinplot(
    data=prob_long_df,
    x="label",
    y="proportion",
    inner="box",
    cut=0,
)
plt.title("Variance of label proportions across clients")
plt.xlabel("Label")
plt.ylabel("Proportion across clients")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "label_proportion_variance_violin.png"), dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# Save label probability table
# -----------------------------------------------------------------------------

prob_df = pd.DataFrame(prob_mat, columns=[f"label_{l}" for l in label_ids])
prob_df.insert(0, "client", list(summary_df["client"]))
prob_df.to_csv(os.path.join(OUT_DIR, "client_label_probabilities.csv"), index=False)

# -----------------------------------------------------------------------------
# Markdown summary
# -----------------------------------------------------------------------------

md_lines = []
md_lines.append("# Client Statistics & Diagnostics\n")
md_lines.append("## What this analysis covers\n")
md_lines.append("- Per-client dataset sizes (train/val only; test is global)\n")
md_lines.append("- Label distribution imbalance within each client\n")
md_lines.append("- Non-IID severity vs global distribution\n")
md_lines.append("- Variance of label composition across clients\n\n")

if global_test_size is not None:
    md_lines.append(f"- Global test set size: {global_test_size}\n\n")

md_lines.append("## Figures\n")
md_lines.append(f"- `{OUT_DIR}/client_sizes.png`\n")
md_lines.append(f"- `{OUT_DIR}/client_imbalance_max_fraction.png`\n")
md_lines.append(f"- `{OUT_DIR}/client_jsd_to_global.png`\n")
md_lines.append(f"- `{OUT_DIR}/client_label_distribution_heatmap.png`\n")
md_lines.append(f"- `{OUT_DIR}/label_proportion_variance_violin.png`\n\n")

md_lines.append("## Summary table (per client)\n")
md_lines.append(summary_df.to_markdown(index=False))
md_lines.append("\n\n")

md_lines.append("## Non-IID metrics (per client)\n")
md_lines.append(
    non_iid_df[
        ["client", "group_value", "n_total",
         "n_classes_present", "max_label_fraction",
         "entropy", "js_divergence_to_global"]
    ].to_markdown(index=False)
)

with open(SUMMARY_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("Wrote:")
print(" -", summary_csv)
print(" -", non_iid_csv)
print(" -", SUMMARY_MD)
print(" - Figures in:", OUT_DIR)
