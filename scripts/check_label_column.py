import scanpy as sc
import pandas as pd

adata = sc.read_h5ad(
    "data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad",
    backed="r"
)

print(adata.obs.columns.tolist())

def check_label_column(col):
    series = adata.obs[col]
    n_total = len(series)
    n_nan = series.isna().sum()
    n_unique = series.dropna().nunique()
    
    print(f"\nColumn: {col}")
    print(f"  Total cells: {n_total}")
    print(f"  NaN values: {n_nan}")
    print(f"  Unique non-NaN values: {n_unique}")

for col in adata.obs.columns:
    check_label_column(col)


def has_labels(adata, min_classes=2):
    for col in adata.obs.columns:
        n_unique = adata.obs[col].dropna().nunique()
        if n_unique >= min_classes:
            return True, col
    return False, None

labeled, column = has_labels(adata)

if labeled:
    print(f"✅ Dataset is labeled (column: {column})")
else:
    print("❌ Dataset is UNLABELED (no usable target column)")



df = pd.read_parquet("data/processed/processed_table.parquet")

print(df.head())
print(df.columns[:10])
print(df["sample_id"].value_counts())
