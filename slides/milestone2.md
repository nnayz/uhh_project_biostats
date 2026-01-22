# Milestone 2 — Data Preparation & Federated Partitioning

## Milestone 2 objectives
- Build a clean, reproducible dataset pipeline
- Create federated clients from replicates
- Quantify client heterogeneity (non-IID)
- Produce figures/tables ready for Milestone 3 and presentation

## Dataset
- Xenium Mouse Brain Replicates (10x Genomics)
- ~475k cells, 248 targeted genes
- Split axis (clients): `library_key` → `sample_id`

## Pipeline overview
- Download raw data
- Validate raw schema + basic stats
- Preprocess (normalize + log1p)
- Create pseudo-labels (Leiden)
- Create 3 federated clients with stratified train/val/test
- Compute client diagnostics + non-IID metrics

## Label strategy (pseudo-labels)
- Raw annotation columns are not usable as supervised labels in this benchmark file
- We generate pseudo-labels via:
  - PCA (30 comps)
  - Neighbors (15, 30 PCs)
  - Leiden (resolution 0.5)
- Mapping stored in `data/processed/label_map.json`

## Federated partitioning
- 3 clients from replicates
- Output per client:
  - `train.parquet`, `val.parquet`, `test.parquet`
  - `client_meta.json`
- Global summary: `data/processed/global_metadata.json`

## Client sizes
![](../outputs/milestone2/figures/client_sizes.png)

## Global label distribution
![](../outputs/milestone2/figures/global_label_distribution.png)

## Per-client label distribution
![](../outputs/milestone2/figures/per_client_label_distribution.png)

## Within-client imbalance
![](../outputs/milestone2/figures/client_imbalance_max_fraction.png)

## Non-IID severity (JSD to global)
![](../outputs/milestone2/figures/client_jsd_to_global.png)

## Spatial clusters (pseudo-labels)
![](../outputs/milestone2/figures/spatial_clusters.png)

## Key takeaways
- Clients are balanced in size
- Mild within-client imbalance
- Weak but measurable non-IID across replicates
- Clean baseline for Milestone 3 experiments

## Reproducibility
```bash
python scripts/data_preparation/download_raw.py
python scripts/data_preparation/validate_raw.py
python scripts/data_preparation/preprocess.py
python scripts/data_preparation/partition_clients.py
python scripts/client_stats.py
```

## Deliverables
- `docs/milestone2_readme.md`
- Figures and tables under `md/figures/`
- Slide source: `slides/milestone2.md`
