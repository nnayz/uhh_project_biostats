# Data Preparation

Data pipeline: download, preprocess, partition for federated learning.

---

## 1. Overview

| Item | Description |
|------|-------------|
| **Dataset** | `10xgenomics_xenium_mouse_brain_replicates.h5ad` (HuggingFace SpatialCorpus) |
| **Strategy** | Anatomical Siloing — Replicate 3 held out; Replicates 1 & 2 split by Y into 3 clients |
| **Outputs** | Parquets, gene list, label map, per-client train/val splits |

---

## 2. Pipeline

```bash
# 1. Download raw h5ad
python scripts/data_preparation/download_raw.py

# 2. Preprocess (filter, normalize, cluster)
python scripts/data_preparation/preprocess.py

# 3. Partition (clients + held-out set)
python scripts/data_preparation/partition_anatomical_siloing.py

# 4. Analysis (stats, UMAPs)
python scripts/analysis/run_analysis.py
```

---

## 3. Scripts

### download_raw.py

Downloads from HuggingFace SpatialCorpus.

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `10xgenomics_xenium_mouse_brain_replicates.h5ad` | Dataset filename |
| `--output_dir` | `data/raw` | Output directory |
| `--force` | False | Re-download if exists |

**Output:** `data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad`

### preprocess.py

Loads h5ad, filters, normalizes, clusters (Leiden), outputs parquet.

| Argument | Default | Description |
|----------|---------|-------------|
| `--raw_path` | `data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad` | Input h5ad |
| `--batch_col` | Auto-detect | Column for batch/replicate ID |
| `--out_dir` | `data/processed` | Output directory |
| `--min_counts` | 10 | Minimum counts per cell |

**Steps:** Filter → Normalize (10k + log1p) → PCA (30) → Neighbors (15) → Leiden (0.5)

**Outputs:**
- `processed_table.parquet` — Full processed data
- `genes.txt` — Gene list (248)
- `label_map.json` — Cluster ID → integer
- `preprocess_config.json`

### partition_anatomical_siloing.py

Splits data into clients and held-out set.

| Argument | Default | Description |
|----------|---------|-------------|
| `--hold_out_replicate` | `"3"` | Replicate to hold out |
| `--replicate_col` | `batch_id` | Column with replicate ID |
| `--seed` | 42 | Random seed |

**Logic:**
- Held-out: Rows matching `hold_out_replicate` → `held_out_batch.parquet`
- Clients: Remaining replicates split by Y-coordinate:
  - `client_01` (Dorsal): y > 66th percentile
  - `client_02` (Mid): 33rd < y ≤ 66th percentile
  - `client_03` (Ventral): y ≤ 33rd percentile
- Each client: 80% train / 20% val (stratified by label)

**Outputs:**
```
data/processed/
├── held_out_batch.parquet
├── held_out_batch_meta.json
├── partition_config.json
└── clients/
    ├── client_01/{train,val}.parquet
    ├── client_02/{train,val}.parquet
    └── client_03/{train,val}.parquet
```

---

## 4. Data Schema

### Parquet Columns

| Column | Description |
|--------|-------------|
| `id` | Unique cell ID |
| `sample_id` | Sample identifier |
| `batch_id` | Batch/replicate (from `library_key`) |
| `x`, `y` | Spatial coordinates |
| `label` | Integer class (0 to 23) |
| Gene columns | 248 expression values |

### Files

| File | Description |
|------|-------------|
| `genes.txt` | Ordered gene names (one per line) |
| `label_map.json` | `{"0": 0, "1": 1, ...}` |
| `partition_config.json` | Strategy, percentiles, seed |

---

## 5. Directory Layout

```
data/
├── raw/
│   └── 10xgenomics_xenium_mouse_brain_replicates.h5ad
└── processed/
    ├── processed_table.parquet
    ├── genes.txt
    ├── label_map.json
    ├── preprocess_config.json
    ├── held_out_batch.parquet
    ├── held_out_batch_meta.json
    ├── partition_config.json
    ├── global_metadata.json
    └── clients/
        ├── client_01/
        │   ├── train.parquet
        │   ├── val.parquet
        │   └── client_meta.json
        ├── client_02/
        └── client_03/
```

---

## 6. Client Analysis

After partitioning, run:

```bash
python scripts/analysis/run_analysis.py
```

**Outputs in `outputs/analysis/`:**
- `client_summary.csv` — Per-client statistics
- `client_noniid_metrics.csv` — Entropy, JSD metrics
- `label_proportion_heatmap.png` — Class distribution across clients
- `client_jsd_to_global.png` — Non-IID severity
- `umap_by_client.png` — UMAP by anatomical region
- `umap_by_label.png` — UMAP by cell type
- `analysis_summary.md` — Full analysis report

---

## 7. Dataset Statistics

| Metric | Value |
|--------|-------|
| Total cells | ~450K |
| Genes | 248 |
| Classes | 24 (Leiden clusters) |
| Replicates | 3 |
| Clients | 3 (Dorsal/Mid/Ventral) |
| Held-out samples | ~158K (Replicate 3) |
