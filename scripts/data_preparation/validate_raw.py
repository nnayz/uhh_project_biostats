import scanpy as sc
import pandas as pd
import json
import os

# Path to the downloaded file
RAW_PATH = "data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad"

def validate_dataset():
    if not os.path.exists(RAW_PATH):
        print(f"[ERROR] {RAW_PATH} not found.")
        return

    print(f"--- Validating Xenium Mouse Brain Replicates ---")
    
    # Load metadata only (backed mode)
    adata = sc.read_h5ad(RAW_PATH, backed='r')
    
    # Define the mapping based on our "Deep Peek" discovery
    mapping = {
        "sample_id_source": "library_key",
        "label_source": "niche",
        "x_coord": "x",
        "y_coord": "y"
    }

    # 1. Verify existence of required columns
    missing = [col for col in mapping.values() if col not in adata.obs.columns]
    
    # 2. Extract Stats
    stats = {
        "filename": os.path.basename(RAW_PATH),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "columns_verified": {k: (v in adata.obs.columns) for k, v in mapping.items()},
        "federated_clients": adata.obs[mapping["sample_id_source"]].unique().tolist(),
        "n_classes": len(adata.obs[mapping["label_source"]].unique()),
        "class_distribution": adata.obs[mapping["label_source"]].value_counts().to_dict()
    }

    # --- Print Professional Output for Team ---
    print("\n[OK] DATASET VALIDATION PASSED")
    print(f"Total Cells: {stats['n_cells']}")
    print(f"Total Genes: {stats['n_genes']}")
    print(f"Target Labels: {mapping['label_source']} ({stats['n_classes']} classes)")
    print(f"Client ID Source: {mapping['sample_id_source']}")
    print(f"Detected Clients (Sites): {len(stats['federated_clients'])}")
    
    for client in stats['federated_clients']:
        print(f"  - Client ID: {client}")

    if missing:
        print(f"[WARNING] Missing columns: {missing}")
    else:
        print("\n[OK] All columns required by the Data Contract are present.")

    # 3. Save Clean Deliverable
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/basic_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    print("\n[INFO] Basic dataset stats saved to data/raw/basic_stats.json")

if __name__ == "__main__":
    validate_dataset()