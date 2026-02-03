"""
Inspect raw .h5ad obs (metadata) to find batch-relevant columns.

Advisor: use batch/donor/sample or dataset identifiers; need at least 2 batches
for hold-out evaluation. This script lists obs columns and unique value counts
so you can pick a column with multiple batches.

Usage:
  python scripts/data_preparation/inspect_raw_obs.py --raw_path data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Inspect raw h5ad obs columns and value counts")
    parser.add_argument("--raw_path", type=str, default="data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad")
    args = parser.parse_args()

    try:
        import scanpy as sc
    except ImportError:
        print("scanpy required: pip install scanpy")
        sys.exit(1)

    print(f"Loading (backed): {args.raw_path}")
    adata = sc.read_h5ad(args.raw_path, backed="r")
    print(f"  Cells: {adata.n_obs:,}, Genes: {adata.n_vars}\n")

    print("OBS columns (for batches you need a column with >= 2 unique values):")
    print("-" * 60)
    batch_candidates = ["donor_id", "donor", "condition_id", "dataset", "sample_id", "dataset_id", "library_key", "batch", "sample", "section", "replicate"]
    for col in adata.obs.columns:
        uniq = adata.obs[col].astype(str).unique()
        n = len(uniq)
        is_candidate = col.lower() in [c.lower() for c in batch_candidates]
        tag = "  <-- use for batches? (preprocess --batch_col " + col + ")" if n >= 2 and is_candidate else ""
        if n <= 20:
            vals = ", ".join(sorted(uniq)[:20])
            if n > 10:
                vals = vals + f" ... (+{n-10} more)"
            print(f"  {col}: {n} unique  [{vals}]{tag}")
        else:
            print(f"  {col}: {n} unique  (first: {adata.obs[col].iloc[0]}){tag}")

    print("\n" + "-" * 60)
    print("For partitioning you need a column with >= 2 unique values (e.g. library_key for anatomical siloing).")
    print("Preprocess with: --batch_col <column_name>")
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

if __name__ == "__main__":
    main()
