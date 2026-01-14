# Nicheformer on Mouse Brain Transcriptomics Data

A federated learning framework for fine-tuning the **Nicheformer** foundation model on spatial single-cell transcriptomics data from mouse brain tissue. This project implements centralized and federated training pipelines for cell type classification using the Nicheformer architecture.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Milestones Completed](#milestones-completed)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [References](#references)

---

## 🎯 Project Overview

This project applies **Nicheformer** (a transformer-based foundation model for spatial single-cell transcriptomics) to mouse brain spatial transcriptomics data in a federated learning setting. The goal is to:

1. **Fine-tune Nicheformer** for cell type classification on mouse brain data
2. **Compare centralized vs. federated training** approaches
3. **Handle non-IID data distribution** across multiple clients (replicates)
4. **Evaluate model performance** on spatial transcriptomics classification tasks

### Key Features

- ✅ **Centralized Training** - Baseline fine-tuning with all data combined
- ✅ **Federated Learning** - Distributed training using Flower framework with FedAvg
- ✅ **Nicheformer Integration** - Full integration with pretrained weights support
- ✅ **Data Pipeline** - Complete preprocessing and federated partitioning
- ✅ **Model Wrapper** - Standardized interface for fine-tuning modes (head_only, partial, full)
- ✅ **Shared Training Engine** - Reusable training and evaluation functions
- ✅ **GPU Optimizations** - AMP, parallel data loading, optimized batch sizes for maximum GPU utilization

---

## ✅ Milestones Completed

### Milestone 2: Data Preparation & Federated Partitioning ✅

**Status:** Complete

**Deliverables:**
- Dataset acquisition from HuggingFace (10x Genomics Xenium Mouse Brain Replicates)
- Data preprocessing and validation
- Pseudo-label generation via Leiden clustering (22 cell types)
- Federated partitioning into 3 clients (replicates)
- Non-IID diagnostics and visualizations
- Complete data pipeline with schema validation

**Key Outputs:**
- Processed data in `data/processed/` with train/val/test splits per client
- Gene schema (248 genes) and label mapping
- Client statistics and heterogeneity analysis
- Documentation: `docs/milestone2/README.md`

**Run Pipeline:**
```bash
python scripts/data_preparation/download_raw.py
python scripts/data_preparation/validate_raw.py
python scripts/data_preparation/preprocess.py
python scripts/data_preparation/partition_clients.py
python scripts/data_preparation/client_stats.py
```

---

### Milestone 3: Federated Model Training ✅

**Status:** Core components complete. Evaluation (Task 5) in progress.

**Deliverables:**
- Data loaders with schema validation (Data Contract API implementation)
- Nicheformer model wrapper with pretrained weights support
- Shared training engine (`train_one_epoch`, `evaluate` functions)
- Centralized baseline training script
- Federated learning orchestration (Flower + FedAvg)
- Fine-tuning configuration (head_only, partial, full modes)
- Flower client and server implementations
- Training history tracking and artifact saving

**Key Outputs:**
- Centralized training results in `results/centralized/`:
  - `model_final.pt` - Trained model weights
  - `history.json` - Training metrics per epoch
  - `metrics.csv` - Final performance metrics
  - `config.json` - Training configuration
  - `training_curves.png` - Loss and accuracy plots
- Federated training results in `results/federated/`:
  - `model_final.pt` - Aggregated model weights
  - `history.json` - Per-round metrics
  - `metrics.csv` - Final federated metrics
  - `config.json` - Federated training configuration
  - `training_curves.png` - Federated training curves
- Core infrastructure in `src/`:
  - `src/data/loaders.py` - Data loading and validation
  - `src/model/nicheformer_wrapper.py` - Model wrapper
  - `src/training/train_engine.py` - Shared training functions
  - `src/training/fl_client.py` - Flower client
  - `src/training/fl_server.py` - Flower server utilities
- Training scripts:
  - `scripts/run_centralized.py` - Centralized training
  - `scripts/run_federated.py` - Federated training
- Documentation: `docs/milestone3/` with task-specific READMEs

**What is Remaining:**
- Evaluation utilities (Task 5)
- Result visualization and comparison plots
- Per-client test set evaluation
- Confusion matrices and detailed metrics
- Milestone 3 presentation slides

---

## 🚀 Setup & Installation

### Prerequisites

- **Python:** 3.9 or newer
- **Conda/Mamba:** Recommended for environment management
- **Git:** For cloning repository and submodules
- **CUDA:** Optional, for GPU acceleration (PyTorch with CUDA support)
  - **Note:** Install CUDA-enabled PyTorch for GPU training. CPU-only PyTorch will limit performance.

### Step 1: Clone Repository

```bash
# Clone with submodules (Nicheformer is a submodule)
git clone --recurse-submodules <repository-url>
cd Nicheformer-On-Mouse-Brain-Transcriptomics-Data

# OR if already cloned without submodules:
git submodule update --init --recursive
```

### Step 2: Create Conda Environment

```bash
# Create environment
conda create -n niche python=3.9
conda activate niche

# Install core dependencies
pip install -r requirements.txt
```

### Step 3: Install Nicheformer

```bash
# Navigate to submodule directory
cd temp_nicheformer

# Install Nicheformer in editable mode
pip install -e .

# Return to project root
cd ..
```

**Note:** Nicheformer installation will automatically install its dependencies (PyTorch Lightning, torchmetrics, wandb, etc.)

### Step 4: Download Pretrained Weights (Optional but Recommended)

```bash
# Follow instructions in:
python scripts/milestone3/download_nicheformer_weights.py

# Or manually download from:
# https://data.mendeley.com/preview/87gm9hrgm8
# Place in: data/pretrained/nicheformer_pretrained.ckpt
```

### Step 5: Prepare Data (Milestone 2 Pipeline)

**Option A: Use Pre-processed Dataset (Recommended)**

Download the shared processed dataset from Google Drive:
- Link: https://drive.google.com/file/d/1cPzEAUFvLVdi0_cgKTXQl1WPJSqADda4/view?usp=sharing
- Extract to `data/processed/`

**Option B: Run Pipeline Locally**

```bash
# Run Milestone 2 pipeline
python scripts/data_preparation/download_raw.py
python scripts/data_preparation/validate_raw.py
python scripts/data_preparation/preprocess.py
python scripts/data_preparation/partition_clients.py
python scripts/data_preparation/client_stats.py
```

---

## 🏃 Quick Start

### 1. Centralized Training (GPU Optimized)

```bash
# Activate environment
conda activate niche

# Run centralized training with GPU optimizations
python scripts/run_centralized.py \
    --data_dir data/processed \
    --output_dir results/centralized \
    --device cuda \
    --epochs 10 \
    --batch_size 1024 \
    --lr 1e-4 \
    --fine_tune_mode head_only \
    --pretrained_path data/pretrained/nicheformer_pretrained.ckpt \
    --num_workers 4 \
    --use_amp
```

**GPU Optimizations:**
- `--batch_size 1024` - Increased for better GPU utilization
- `--num_workers 4` - Parallel data loading to prevent GPU starvation
- `--use_amp` - Automatic Mixed Precision for ~2x speedup
- `--device cuda` - Use GPU acceleration

### 2. Federated Training (GPU Optimized)

```bash
# Run federated training with GPU optimizations
python scripts/run_federated.py \
    --data_dir data/processed \
    --output_dir results/federated \
    --num_rounds 5 \
    --clients_per_round 3 \
    --local_epochs 2 \
    --batch_size 1024 \
    --lr 1e-4 \
    --fine_tune_mode head_only \
    --pretrained_path data/pretrained/nicheformer_pretrained.ckpt \
    --device cuda \
    --use_amp
```

**Note:** For CPU-only systems, use `--device cpu` and reduce `--batch_size` to 256.

---

## 📖 Usage Examples

### Data Loading

```python
from src.data import load_client_data, load_all_clients, load_gene_list, create_dataloader

# Load data for a specific client
train_df = load_client_data("client_01", "train", data_dir="data/processed")
genes = load_gene_list(data_dir="data/processed")
train_loader = create_dataloader(train_df, genes, batch_size=32, shuffle=True)

# Load all clients (for centralized training)
all_train_df = load_all_clients("train", data_dir="data/processed")
```

### Model Creation

```python
from src.model import create_model

# With pretrained weights
model = create_model(
    num_genes=248,
    num_labels=22,
    pretrained_path="data/pretrained/nicheformer_pretrained.ckpt",
    fine_tune_mode="head_only"  # Options: head_only, partial, full
)

# Without pretrained weights (fresh Nicheformer)
model = create_model(
    num_genes=248,
    num_labels=22,
    fine_tune_mode="head_only"
)
```

### Training

```python
from src.training import train_one_epoch, evaluate, create_optimizer, TrainingHistory
import torch

# Setup
optimizer = create_optimizer(model, learning_rate=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
history = TrainingHistory()

# Training loop
for epoch in range(num_epochs):
    train_metrics = train_one_epoch(model, train_loader, optimizer, device)
    val_metrics = evaluate(model, val_loader, device)
    
    history.add_train_metrics(train_metrics)
    history.add_val_metrics(val_metrics)
    
    print(f"Epoch {epoch+1}: Train Loss={train_metrics['loss']:.4f}, "
          f"Val Acc={val_metrics['accuracy']:.4f}")
```

---

## 🔄 Data Pipeline

### Milestone 2 Pipeline Overview

1. **Download Raw Data** (`download_raw.py`)
   - Downloads 10x Genomics Xenium Mouse Brain Replicates from HuggingFace
   - Saves to `data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad`

2. **Validate Raw Data** (`validate_raw.py`)
   - Basic statistics and validation
   - Outputs `data/raw/basic_stats.json`

3. **Preprocess** (`preprocess.py`)
   - Quality filtering
   - Normalization
   - Gene selection (248 genes)
   - Leiden clustering for pseudo-labels (22 cell types)
   - Outputs `data/processed/processed_table.parquet`

4. **Partition Clients** (`partition_clients.py`)
   - Splits data into 3 clients (replicates)
   - Creates train/val/test splits per client
   - Outputs `data/processed/clients/client_XX/{train,val,test}.parquet`

5. **Client Statistics** (`client_stats.py`)
   - Non-IID diagnostics
   - Label distribution analysis
   - Generates visualizations

### Data Schema

- **Genes:** 248 genes (canonical gene list in `data/processed/genes.txt`)
- **Labels:** 22 cell types (from Leiden clustering, mapping in `data/processed/label_map.json`)
- **Features:** Gene expression counts + optional spatial coordinates (x, y)

---

## 🧠 Model Architecture

### Nicheformer

**Nicheformer** is a transformer-based foundation model for spatial single-cell transcriptomics:

- **Architecture:** Transformer encoder with learnable positional encodings
- **Input:** Gene expression vectors (248 genes) + optional spatial coordinates
- **Output:** Cell type classification logits (22 classes)
- **Pretrained Weights:** Available from Mendeley Data (545 MB)

### Fine-Tuning Modes

1. **`head_only`**: Only classifier head trainable (backbone frozen)
   - ~267K trainable parameters
   - Fast training, good for transfer learning

2. **`partial`**: Last half of backbone + head trainable
   - ~12M trainable parameters
   - Balanced approach

3. **`full`**: All parameters trainable
   - ~25.5M trainable parameters
   - Full fine-tuning

---

## 📊 Training Outputs

Each training run generates:

- **`model_final.pt`** - Final model weights
- **`history.json`** - Training history (loss, accuracy, F1 per epoch/round)
- **`metrics.csv`** - Final metrics summary
- **`config.json`** - Training configuration
- **`training_curves.png`** - Training/validation curves

**Output Locations:**
- **Centralized:** `results/centralized/` or `outputs/milestone3/centralized/`
- **Federated:** `results/federated/` or `outputs/milestone3/federated/`

---

## 🧪 Testing

```bash
# Run comprehensive test suite
python scripts/milestone3/test_training_engine.py
```

This tests:
- Data loaders
- Model wrapper
- Training engine functions
- Full integration

---

## 📚 References

### Papers

- **Nicheformer:** Schaar, A.C., Tejada-Lapuerta, A., et al. (2024). Nicheformer: a foundation model for single-cell and spatial omics. bioRxiv. https://doi.org/10.1101/2024.04.15.589472

### Datasets

- **10x Genomics Xenium Mouse Brain Replicates:** Available on HuggingFace (`theislab/SpatialCorpus-110M`)

### Frameworks

- **Nicheformer:** https://github.com/theislab/nicheformer
- **Flower (Federated Learning):** https://flower.dev/
- **PyTorch Lightning:** https://lightning.ai/docs/pytorch/

### Pretrained Weights

- **Nicheformer Pretrained Weights:** https://data.mendeley.com/preview/87gm9hrgm8


