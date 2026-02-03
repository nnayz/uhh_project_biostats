import os
import scanpy as sc
import pandas as pd
from tabulate import tabulate

# Resolve path relative to project root so it works from any cwd
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# List of dataset names to process
DATASET_NAMES = [
    "10xgenomics_xenium_mouse_brain_replicates.h5ad",
    # Add more dataset filenames here, e.g.:
    # "another_dataset.h5ad",
]
RAW_PATHS = [os.path.join(_PROJECT_ROOT, "data", "raw", dataset_name) for dataset_name in DATASET_NAMES]

cols_to_check = ['library_key', 'region', 'replicate']

def process_dataset(dataset_name, raw_path):
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(
            f"Raw file not found: {raw_path}\n"
            f"Download it: python scripts/data_preparation/download_raw.py --dataset {dataset_name}"
        )
    # Load the metadata only (to save memory since these files are large)
    adata = sc.read_h5ad(raw_path, backed="r")

    # Mention the dataset name in the output
    print(f"\nLoaded dataset: {dataset_name}\n")

    # 1. List all available metadata columns
    print("Available metadata columns:", adata.obs.columns.tolist())

    # 2. Display a single concise table with value counts for each column, with lines in between each column block
    print(f"\nConcise value counts for selected columns in dataset '{dataset_name}': {cols_to_check}\n")
    summary_tables = []
    for i, col in enumerate(cols_to_check):
        if col in adata.obs.columns:
            counts = adata.obs[col].value_counts()
            df_temp = pd.DataFrame({
                "column": col,
                "value": counts.index,
                "count": counts.values
            })
            summary_tables.append(df_temp)
        else:
            # Still add a row but indicate missing column
            df_temp = pd.DataFrame({
                "column": [col],
                "value": ["<not found>"],
                "count": [None]
            })
            summary_tables.append(df_temp)
        # After each column except the last, insert a 'separator row'
        if i < len(cols_to_check) - 1:
            summary_tables.append(pd.DataFrame({
                "column": [""],
                "value": ["-----"],
                "count": [""]
            }))
    result_df = pd.concat(summary_tables, ignore_index=True)
    print(tabulate(result_df, headers="keys", tablefmt="psql", showindex=False))


def main():
    for dataset_name, raw_path in zip(DATASET_NAMES, RAW_PATHS):
        process_dataset(dataset_name, raw_path)

if __name__ == "__main__":
    main()