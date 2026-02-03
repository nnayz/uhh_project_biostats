# Nicheformer on Spatial Transcriptomics Data

A federated learning framework for fine-tuning **Nicheformer** on spatial single-cell transcriptomics data. Compares centralized, federated, and local training strategies for cell type classification.

---

## Project Overview

This project applies Nicheformer (a transformer-based foundation model) to **mouse brain** spatial transcriptomics data in a federated learning setting:

1. **Fine-tune Nicheformer** for cell type classification (24 Leiden clusters)
2. **Compare training strategies:** Centralized vs Federated vs Local
3. **Anatomical siloing (Non-IID):** Clients split by Y-coordinate (Dorsal/Mid/Ventral)
4. **Held-out evaluation:** Replicate 3 reserved for zero-shot evaluation

### Key Results

| Strategy | Accuracy | Macro-F1 |
|----------|----------|----------|
| Centralized | 94.45% | 94.09% |
| Federated | 92.76% | 86.84% |
| Local (mean) | 76.75% | 67.82% |

---

## Documentation

### Technical Documentation
| Document | Contents |
|----------|----------|
| [docs/data_preparation.md](docs/data_preparation.md) | Data pipeline: download, preprocess, partition |
| [docs/training.md](docs/training.md) | Training scripts, model, data loaders, configs |
| [docs/evaluation.md](docs/evaluation.md) | Evaluation scripts and process |

### Results & Analysis
| Document | Contents |
|----------|----------|
| [docs/training_summary.md](docs/training_summary.md) | Training results for all strategies |
| [docs/evaluation_results.md](docs/evaluation_results.md) | Model comparison and analysis |
| [docs/analysis_summary.md](docs/analysis_summary.md) | Dataset & client non-IID analysis |

---

## Quick Start

### 1. Setup

```bash
# Clone with submodules
git clone --recurse-submodules <repository-url>
cd Nicheformer-On-Mouse-Brain-Transcriptomics-Data

# Create environment
conda create -n niche python=3.9
conda activate niche
pip install -r requirements.txt

# Install Nicheformer
cd temp_nicheformer && pip install -e . && cd ..
```

### 2. Data Preparation

```bash
# Download and preprocess
python scripts/data_preparation/download_raw.py
python scripts/data_preparation/preprocess.py
python scripts/data_preparation/partition_anatomical_siloing.py

# Run analysis
python scripts/analysis/run_analysis.py
```

### 3. Training

```bash
# Centralized
python scripts/run_centralized.py --device cuda --epochs 10 --use_amp

# Federated
python scripts/run_federated.py --device cuda --num_rounds 10 --use_amp

# Local (all clients)
python scripts/run_local.py --client_id all --device cuda --epochs 10 --use_amp
```

### 4. Evaluation

```bash
python scripts/milestone3/evaluate_models.py --device cuda --use_amp
```

---

## Project Structure

```
├── docs/                      # Documentation
│   ├── data_preparation.md    # Data pipeline
│   ├── training.md            # Training scripts
│   ├── evaluation.md          # Evaluation process
│   ├── training_summary.md    # Training results
│   ├── evaluation_results.md  # Model comparison
│   └── analysis_summary.md    # Dataset analysis
├── scripts/
│   ├── data_preparation/      # Download, preprocess, partition
│   ├── analysis/              # Client statistics, UMAPs
│   ├── milestone3/            # Evaluation scripts
│   ├── run_centralized.py
│   ├── run_federated.py
│   └── run_local.py
├── src/                       # Core modules
│   ├── data/                  # Data loaders
│   ├── model/                 # Nicheformer wrapper
│   └── training/              # Training engine, FL client/server
├── data/
│   ├── raw/                   # Downloaded h5ad files
│   ├── processed/             # Parquets, clients, held-out set
│   └── pretrained/            # Nicheformer weights
├── results/
│   ├── centralized/
│   ├── federated/
│   ├── local/
│   └── comparison/
└── outputs/analysis/          # Data analysis outputs
```

---

## Model Architecture

**Nicheformer** is a transformer-based foundation model for spatial transcriptomics:
- **Input:** 248 genes + spatial coordinates (x, y)
- **Output:** 24 cell type classes
- **Fine-tuning modes:** `head_only`, `partial`, `full`

---

## Dataset

- **Source:** `10xgenomics_xenium_mouse_brain_replicates.h5ad` (HuggingFace SpatialCorpus)
- **Partitioning:** Anatomical siloing — 3 clients by Y-coordinate
- **Evaluation:** Held-out Replicate 3 (157,982 samples)

---

## References

- **Nicheformer:** [bioRxiv 2024](https://doi.org/10.1101/2024.04.15.589472)
- **Flower:** [flower.dev](https://flower.dev/)
- **SpatialCorpus:** [HuggingFace](https://huggingface.co/datasets/theislab/SpatialCorpus-110M)
