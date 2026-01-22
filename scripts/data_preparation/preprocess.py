import os, json
import scanpy as sc
import pandas as pd

RAW_PATH = "data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad"
OUT_DIR = "data/processed"
CLIENT_COL = "library_key"   # for federated split later
X_COL, Y_COL = "x", "y"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data...")
# Load data
adata = sc.read_h5ad(RAW_PATH)
print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

# -----------------------------
# 1) Basic filtering (safe) - skip if memory issues
# -----------------------------
print("Filtering cells (min_counts=10)...")
try:
    sc.pp.filter_cells(adata, min_counts=10, inplace=True)
    print(f"After filtering: {adata.n_obs} cells")
except (MemoryError, Exception) as e:
    print(f"[WARNING] Error during filtering ({type(e).__name__}), skipping filter step")
    print("Proceeding with all cells")

# -----------------------------
# 2) Normalize + log1p
# -----------------------------
print("Normalizing data...")
try:
    # Use inplace operations to save memory
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata, copy=False)
    print("Normalization complete")
except (MemoryError, Exception) as e:
    print(f"[ERROR] Error during normalization: {type(e).__name__}: {e}")
    print("Trying alternative approach: converting to dense array first...")
    try:
        # Last resort: convert to dense (uses more memory but avoids sparse ops)
        import gc
        gc.collect()
        if hasattr(adata.X, 'toarray'):
            print("Converting to dense array (this will use more memory)...")
            adata.X = adata.X.toarray()
        sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
        sc.pp.log1p(adata, copy=False)
        print("Normalization complete (using dense array)")
    except Exception as e2:
        print(f"[FATAL] Could not normalize data: {e2}")
        raise

# -----------------------------
# 3) PCA + neighborhood graph
# -----------------------------
sc.pp.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)

# -----------------------------
# 4) Leiden clustering (LABELS)
# -----------------------------
sc.tl.leiden(
    adata,
    resolution=0.5,
    key_added="cluster",
    flavor="igraph",
    n_iterations=2,
    directed=False
)

labels = adata.obs["cluster"].astype(str)
uniq = sorted(labels.unique())

label_map = {lab: i for i, lab in enumerate(uniq)}
with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

encoded_labels = labels.map(label_map).astype("int32")

print(f"Created {len(uniq)} clusters")

# -----------------------------
# 5) Save gene schema
# -----------------------------
gene_names = list(adata.var_names)
with open(os.path.join(OUT_DIR, "genes.txt"), "w") as f:
    f.write("\n".join(gene_names))

# -----------------------------
# 6) Build processed table
# -----------------------------
X = adata.X
try:
    import scipy.sparse as sp
    if sp.issparse(X):
        X = X.toarray()
except Exception:
    pass

expr_df = pd.DataFrame(X, columns=gene_names)

meta_df = pd.DataFrame({
    "id": adata.obs_names.astype(str),
    "sample_id": adata.obs[CLIENT_COL].astype(str).values,
    "x": adata.obs[X_COL].astype("float32").values,
    "y": adata.obs[Y_COL].astype("float32").values,
    "label": encoded_labels.values,
})

out = pd.concat([meta_df, expr_df], axis=1)
out_path = os.path.join(OUT_DIR, "processed_table.parquet")
out.to_parquet(out_path, index=False)

print("Saved:", out_path)
print("Rows:", len(out), "Genes:", len(gene_names))
