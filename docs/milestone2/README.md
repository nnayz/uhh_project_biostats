# Milestone 2 — Data Preparation & Federated Partitioning (Submission)

## Goal
Prepare a clean, reproducible Milestone 2 dataset and diagnostics so the repository is ready for Milestone 3 (federated model training) without rework.

## What’s included
- Dataset acquisition and basic validation
- Preprocessing + pseudo-label creation
- Federated partitioning into 3 clients (replicates)
- Client heterogeneity / non-IID diagnostics (plots + tables)

## Repository layout (relevant paths)
- `data/raw/`
  - `10xgenomics_xenium_mouse_brain_replicates.h5ad` (downloaded)
  - `basic_stats.json` (validation summary)
- `data/processed/`
  - `processed_table.parquet` (global processed table)
  - `genes.txt` (gene schema / order)
  - `label_map.json` (pseudo-label encoding)
  - `global_metadata.json` (client summary)
  - `clients/client_XX/{train,val,test}.parquet` and `client_meta.json`
- Documentation
  - `md/data_dictionary.md` (data contract)
  - `md/preprocessing_notes.md` (pseudo-labeling + preprocessing decisions)
  - `md/partitioning_strategy.md` (client split definition)
  - `md/client_stats_summary.md` (non-IID diagnostics writeup)
- Presentation-ready figures/tables
  - `md/figures/*.png`
  - `md/figures/*.csv`

## Important decision: labels
The raw dataset contains annotation columns (e.g. `niche`, `region`), but in this benchmark file the available label columns are not usable for supervised learning (only a single unique value). For Milestone 2 we therefore generate **pseudo-labels** via Leiden clustering.

- Label column used downstream: `label` (integer)
- Origin: `adata.obs["cluster"]` from Leiden
- Mapping stored in: `data/processed/label_map.json`

Details: `md/preprocessing_notes.md`

## How to run (Milestone 2 pipeline)
Run in this order:

```bash
python scripts/download_raw.py
python scripts/validate_raw.py
python scripts/preprocess.py
python scripts/partition_clients.py
python scripts/client_stats.py
```

### Expected outputs after running
- `data/raw/basic_stats.json`
- `data/processed/processed_table.parquet`
- `data/processed/genes.txt`
- `data/processed/label_map.json`
- `data/processed/clients/client_01/` … `client_03/`
- Figures/tables under `md/figures/`

## Diagnostics outputs (used in slides)
- `md/figures/spatial_clusters.png`
- `md/figures/client_sizes.png`
- `md/figures/global_label_distribution.png`
- `md/figures/per_client_label_distribution.png`
- `md/figures/client_imbalance_max_fraction.png`
- `md/figures/client_jsd_to_global.png`
- `diagnostics/figures/batch_correction_umap_and_accuracy_train.png`
- `diagnostics/figures/batch_correction_umap_and_accuracy_test.png`

## Traceability
- Data contract: `md/data_dictionary.md`
- Preprocessing rationale: `md/preprocessing_notes.md`
- Partitioning rationale: `md/partitioning_strategy.md`
- Non-IID metrics + tables: `md/client_stats_summary.md` and `md/figures/*.csv`
