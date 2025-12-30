# scripts/preprocess_unlabeled.py

import os
import json
import numpy as np
import pandas as pd
import scanpy as sc

# ----------------------------
# Paths
# ----------------------------
RAW_PATH = "data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Data Contract Fields
# ----------------------------
SAMPLE_COL = "library_key"   # federated client id
X_COL = "x"
Y_COL = "y"

# ----------------------------
# Step 1: Load Data
# ----------------------------
print("Loading Xenium dataset...")
adata = sc.read_h5ad(RAW_PATH)

print(f"Cells: {adata.n_obs}")
print(f"Genes: {adata.n_vars}")

# ----------------------------
# Step 2: Expression Normalization
# ----------------------------
print("Normalizing expression (total-count + log1p)...")

# Normalize to 10k counts per cell
sc.pp.normalize_total(adata, target_sum=10_000)

# Log-transform
sc.pp.log1p(adata)

# ----------------------------
# Step 3: Build Expression Table
# ----------------------------
print("Building processed table...")

# Convert expression matrix
if not isinstance(adata.X, np.ndarray):
    expr = adata.X.toarray()
else:
    expr = adata.X

expr_df = pd.DataFrame(
    expr,
    columns=adata.var_names,
    index=adata.obs_names
)

# Metadata (NO LABELS)
meta_df = pd.DataFrame({
    "id": adata.obs_names,
    "sample_id": adata.obs[SAMPLE_COL].values,
    "x": adata.obs[X_COL].values,
    "y": adata.obs[Y_COL].values
})

# Combine metadata + expression
final_df = pd.concat(
    [meta_df.reset_index(drop=True),
     expr_df.reset_index(drop=True)],
    axis=1
)

# ----------------------------
# Step 4: Save Outputs
# ----------------------------
# Save parquet
parquet_path = os.path.join(OUT_DIR, "processed_table.parquet")
final_df.to_parquet(parquet_path, index=False)

print(f"Saved processed_table.parquet → {parquet_path}")

# Save gene list
genes_path = os.path.join(OUT_DIR, "genes.txt")
with open(genes_path, "w") as f:
    for gene in adata.var_names:
        f.write(f"{gene}\n")

print(f"Saved genes.txt ({len(adata.var_names)} genes)")

# Save metadata info
metadata = {
    "dataset": "Xenium Mouse Brain Replicates",
    "labeled": False,
    "normalization": "total_count_10k + log1p",
    "federated_clients": adata.obs[SAMPLE_COL].unique().tolist(),
    "n_cells": int(adata.n_obs),
    "n_genes": int(adata.n_vars)
}

with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("Saved metadata.json")

print("\nUnlabeled preprocessing completed successfully.")
