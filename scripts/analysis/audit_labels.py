import scanpy as sc
import pandas as pd

RAW_PATH = "data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad"

adata = sc.read_h5ad(RAW_PATH, backed="r")  # metadata only is fine
obs = adata.obs

print("Columns in adata.obs:", list(obs.columns))

# Check candidate label columns (common ones)
candidates = ["niche", "region", "cell_type", "celltype", "annotation", "cluster", "subclass"]

for col in candidates:
    if col in obs.columns:
        s = obs[col]
        n_nonnull = int(s.notna().sum())
        nunique = int(s.dropna().nunique())
        print(f"\n{col}: non-null={n_nonnull}, nunique={nunique}")
        if nunique > 1:
            print(s.dropna().value_counts().head(10))
