# Preprocessing Notes (Milestone 2)

## Dataset
**Xenium Mouse Brain Replicates**  
Source: 10x Genomics Xenium spatial transcriptomics dataset (h5ad format)

## Initial Label Inspection
The dataset contains the following annotation columns:
- `niche`
- `region`

However, both columns contain **only a single unique value across all cells**, making them unsuitable for supervised or federated learning experiments.

## Label Strategy: Pseudo-labels via Clustering
To enable meaningful downstream learning and federated partitioning, we derive **pseudo-labels** using unsupervised clustering.

### Clustering Method
- Dimensionality reduction: **PCA**
  - Number of components: 30
- Neighborhood graph:
  - `n_neighbors = 15`
  - `n_pcs = 30`
- Clustering algorithm: **Leiden**
  - Resolution: `0.5`
  - Backend: `igraph`
  - Result: **22 clusters**

Each cell is assigned a cluster ID, which is used as the classification label.

## Normalization
- Total-count normalization (`target_sum = 1e4`)
- Logarithmic transform (`log1p`)

Normalization is applied **globally** before any client partitioning to avoid artificial client drift.

## Feature Schema
- Number of genes: **248**
- Canonical gene order is stored in:
  - `data/processed/genes.txt`
- All downstream processing assumes this exact gene order.

## Outputs
The preprocessing step produces the following artifacts:

- `data/processed/processed_table.parquet`
  - Contains metadata (`id`, `sample_id`, `x`, `y`, `label`) and gene expression
- `data/processed/genes.txt`
  - Canonical gene list and order
- `data/processed/label_map.json`
  - Mapping from Leiden cluster IDs to integer labels

## Notes
This pseudo-labeling approach is standard practice in spatial transcriptomics when curated cell-type annotations are unavailable and is sufficient for validating federated learning pipelines in Milestone 3.
