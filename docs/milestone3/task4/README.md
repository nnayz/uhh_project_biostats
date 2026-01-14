# Task 4: Model Integration & Shared Training Engine

**Owner:** Task 4 Team Member  
**Milestone:** 3  
**Status:** ✅ Complete

## Overview

This task implements the core infrastructure for model integration and training that all other team members rely on. It provides:

1. **Data Loaders** - Implements the Data Contract API
2. **Model Wrapper** - Wraps Nicheformer with a clean, standardized interface
3. **Training Engine** - Shared training and evaluation functions
4. **Configuration Utilities** - Fine-tuning mode management

All components follow the contracts defined in `docs/milestone3/contracts/` and are designed to be modular, reusable, and extensible for Milestone 4.

---

## What Was Done

### 1. Data Loaders (`src/data/loaders.py`)

**Purpose:** Implements the Data Contract API for loading federated client data.

**Key Functions:**
- `load_client_data(client_id, split)` - Load data for a specific client
- `load_all_clients(split)` - Load data from all clients
- `load_gene_list()` - Get canonical gene list (248 genes)
- `load_label_map()` - Get label mapping (22 labels from Leiden clustering)
- `create_dataloader(df, genes, ...)` - Create PyTorch DataLoader
- `CellDataset` - PyTorch Dataset class for spatial transcriptomics cells

**Features:**
- Schema validation (gene order, labels, NaN checks)
- Support for spatial coordinates (x, y)
- Automatic conversion to PyTorch tensors
- Ready for both centralized and federated training

**Contract Reference:** `docs/milestone3/contracts/data_contract.md`

### 2. Model Wrapper (`src/model/nicheformer_wrapper.py`)

**Purpose:** Wraps Nicheformer model to provide a clean, standardized interface.

**Key Methods (Model Contract):**
- `forward(batch) -> logits` - Forward pass through model
- `compute_loss(logits, labels) -> loss` - Compute CrossEntropyLoss
- `get_weights() -> state_dict` - Get model weights for federated aggregation
- `set_weights(state_dict)` - Set model weights from federated aggregation

**Fine-Tuning Modes:**
- `head_only` - Only classifier head is trainable (backbone frozen)
- `partial` - Last half of backbone + head trainable
- `full` - All parameters trainable

**Features:**
- Configurable fine-tuning modes
- Support for spatial coordinates in input
- Placeholder backbone (ready for actual Nicheformer integration)
- Parameter counting utilities

**Contract Reference:** `docs/milestone3/contracts/model_contract.md`

**Note:** The wrapper automatically integrates with Nicheformer when available! It will:
- Try to import and load Nicheformer if the package is installed
- Use pretrained weights if `pretrained_path` is provided
- Fall back to a placeholder backbone if Nicheformer is not available

See the "Notes" section below for installation instructions.

### 3. Training Engine (`src/training/train_engine.py`)

**Purpose:** Provides reusable training and evaluation functions used by both centralized and federated training.

**Key Functions:**
- `train_one_epoch(model, dataloader, optimizer, device)` - Train for one epoch
- `evaluate(model, dataloader, device)` - Evaluate model
- `TrainingHistory` - Track metrics across epochs/rounds
- `save_training_artifacts()` - Save all required outputs
- `create_optimizer()` - Create optimizer (Adam, AdamW, SGD)
- `create_scheduler()` - Create learning rate scheduler

**Metrics Tracked:**
- Training/validation loss
- Accuracy
- Macro-averaged F1 score

**Output Artifacts (Training Contract):**
- `history.json` - Training history
- `metrics.csv` - Final metrics
- `model_final.pt` - Final model weights
- `config.json` - Training configuration

**Contract Reference:** `docs/milestone3/contracts/training_contract.md`

### 4. Configuration (`src/config.py`)

**Purpose:** Configuration management for training parameters and fine-tuning modes.

**Key Classes:**
- `TrainingConfig` - Dataclass for all training parameters
- `FineTuningConfig` - Fine-tuning mode configuration
- `create_default_config()` - Factory function with overrides

---

## File Structure

```
src/
├── data/
│   ├── __init__.py
│   └── loaders.py          # Data Contract implementation
├── model/
│   ├── __init__.py
│   └── nicheformer_wrapper.py  # Model Contract implementation
├── training/
│   ├── __init__.py
│   └── train_engine.py    # Training Contract implementation
├── config.py              # Configuration utilities
└── README.md              # Module documentation
```

---

## How to Use

### Quick Start

```python
from src.data import load_client_data, load_gene_list, create_dataloader
from src.model import create_model
from src.training import train_one_epoch, evaluate, create_optimizer
import torch

# 1. Load data
data_dir = "data/processed"
genes = load_gene_list(data_dir)
df = load_client_data("client_01", "train", data_dir)
dataloader = create_dataloader(df, genes, batch_size=32, shuffle=True)

# 2. Create model
model = create_model(
    num_genes=248,
    num_labels=22,
    fine_tune_mode="head_only",
    include_spatial=True
)

# 3. Train
optimizer = create_optimizer(model, learning_rate=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metrics = train_one_epoch(model, dataloader, optimizer, device)
print(f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

# 4. Evaluate
val_dataloader = create_dataloader(
    load_client_data("client_01", "val", data_dir),
    genes, batch_size=32, shuffle=False
)
val_metrics = evaluate(model, val_dataloader, device)
print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
```

### Complete Training Loop Example

```python
import torch
from src.data import load_client_data, load_gene_list, create_dataloader
from src.model import create_model
from src.training import (
    train_one_epoch, evaluate, TrainingHistory,
    create_optimizer, save_training_artifacts
)
from src.config import create_default_config

# Configuration
config = create_default_config(
    num_genes=248,
    num_labels=22,
    fine_tune_mode="head_only",
    batch_size=32,
    num_epochs=10,
    learning_rate=1e-4
)

# Load data
data_dir = "data/processed"
genes = load_gene_list(data_dir)

train_df = load_client_data("client_01", "train", data_dir)
val_df = load_client_data("client_01", "val", data_dir)

train_loader = create_dataloader(train_df, genes, batch_size=config.batch_size, shuffle=True)
val_loader = create_dataloader(val_df, genes, batch_size=config.batch_size, shuffle=False)

# Create model
model = create_model(
    num_genes=config.num_genes,
    num_labels=config.num_labels,
    fine_tune_mode=config.fine_tune_mode
)

# Setup training
optimizer = create_optimizer(model, learning_rate=config.learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
history = TrainingHistory()

# Training loop
for epoch in range(config.num_epochs):
    print(f"Epoch {epoch + 1}/{config.num_epochs}")
    
    # Train
    train_metrics = train_one_epoch(model, train_loader, optimizer, device)
    history.add_train_metrics(train_metrics)
    
    # Validate
    val_metrics = evaluate(model, val_loader, device)
    history.add_val_metrics(val_metrics)
    
    print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
    print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

# Save artifacts
save_training_artifacts(
    "outputs/experiment",
    model,
    history,
    config.to_dict(),
    {'final_val_accuracy': history.val_accuracy[-1]}
)
```

### Fine-Tuning Modes

```python
# Head-only fine-tuning (only classifier head trainable)
model = create_model(
    num_genes=248,
    num_labels=22,
    fine_tune_mode="head_only"
)

# Partial fine-tuning (last half of backbone + head trainable)
model = create_model(
    num_genes=248,
    num_labels=22,
    fine_tune_mode="partial"
)

# Full fine-tuning (all parameters trainable)
model = create_model(
    num_genes=248,
    num_labels=22,
    fine_tune_mode="full"
)
```

---

## For Other Team Members

### Person 1: Centralized Baseline Owner

**What you need:**
- Use `train_one_epoch()` and `evaluate()` functions
- Load data using `load_all_clients("train")` to combine all client data
- Use the same training loop as shown above

**Example:**
```python
from src.data import load_all_clients, load_gene_list, create_dataloader
from src.training import train_one_epoch, evaluate

# Load all client data combined
train_df = load_all_clients("train", data_dir)
train_loader = create_dataloader(train_df, genes, batch_size=32, shuffle=True)

# Use the same training functions
train_metrics = train_one_epoch(model, train_loader, optimizer, device)
```

### Person 2: Federated Orchestration Owner (Flower)

**What you need:**
- Use `model.get_weights()` to get client weights for aggregation
- Use `model.set_weights()` to set global weights on clients
- Use `train_one_epoch()` for local client training

**Example Flower Client:**
```python
import flwr as fl
from src.training import train_one_epoch, create_optimizer
from src.model import create_model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_loader, val_loader):
        self.model = create_model(num_genes=248, num_labels=22)
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def fit(self, parameters, config):
        # Set global weights
        self.model.set_weights(parameters)
        
        # Train locally
        optimizer = create_optimizer(self.model, learning_rate=1e-4)
        for epoch in range(config["local_epochs"]):
            train_one_epoch(self.model, self.train_loader, optimizer, device)
        
        # Return updated weights
        return self.model.get_weights(), len(self.train_loader.dataset), {}
```

### Person 3: Data & Schema Owner

**What you need:**
- The data loaders are already implemented! You can extend validation if needed
- All data loading goes through `src/data/loaders.py`
- Schema validation is built-in

**To extend validation:**
- Modify `_validate_dataframe()` in `src/data/loaders.py`
- Add custom checks as needed

### Person 5: Evaluation & Reporting Owner

**What you need:**
- Use `evaluate()` function which returns predictions and labels
- Access metrics: `metrics['accuracy']`, `metrics['f1_macro']`, etc.
- Use `metrics['predictions']` and `metrics['labels']` for confusion matrices

**Example:**
```python
from src.training import evaluate
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Evaluate
metrics = evaluate(model, test_loader, device)

# Get predictions for confusion matrix
predictions = metrics['predictions']
labels = metrics['labels']

# Create confusion matrix
cm = confusion_matrix(labels, predictions)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()
```

---

## Testing

### Run Test Suite

```bash
python scripts/milestone3/test_training_engine.py
```

This tests:
1. **Data Loaders** - Will skip if data not found (with warning)
2. **Model Wrapper** - Uses dummy data, always runs
3. **Training Engine** - Uses dummy data, always runs
4. **Full Integration** - Uses dummy data, always runs

### Expected Output

If everything is set up correctly:
```
[PASSED] Data Loaders test passed!
[PASSED] Model Wrapper test passed!
[PASSED] Training Engine test passed!
[PASSED] Full Integration test passed!

[SUCCESS] All tests passed!
```

### Prerequisites for Testing

1. **Install Dependencies:**
   ```bash
   pip install torch pandas numpy scikit-learn
   ```

2. **Prepare Data (Optional):**
   ```bash
   python scripts/data_preparation/download_raw.py
   python scripts/data_preparation/validate_raw.py
   python scripts/data_preparation/preprocess_chunked.py  # For memory-constrained systems
   python scripts/data_preparation/partition_clients.py
   ```

**Note:** The test script works even without real data - it uses dummy data for model and training engine tests.

---

## Dependencies

Required packages (see `requirements.txt`):
- `torch>=2.0.0` - PyTorch
- `pandas>=1.5.0` - Data handling
- `numpy>=1.23.0` - Numerical operations
- `scikit-learn>=1.2.0` - Metrics (accuracy, F1)

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Design Principles

1. **Modular:** Each component is independent and can be used separately
2. **Contract-based:** Follows contracts defined in `docs/milestone3/contracts/`
3. **Reusable:** Same functions work for centralized and federated training
4. **Extensible:** Easy to extend for Milestone 4 (spatial modeling, multi-task)
5. **Well-documented:** Clear interfaces and usage examples

---

## Milestone 4 Extensions

The design supports easy extension:

### Spatial Modeling
- Add spatial features in model `forward()` without changing training loop
- Spatial coordinates are already included in data loaders

### Multi-Task Learning
- Extend `compute_loss()` to include auxiliary losses
- Training loop remains unchanged

### Personalization
- Model wrapper can be extended with client-specific heads
- Use `get_weights()` and `set_weights()` for personalized aggregation

**The training loop (`train_one_epoch`, `evaluate`) remains unchanged for all extensions.**

---



## Nicheformer Integration

### ✅ Installation Complete

Nicheformer has been successfully installed and integrated:

- **Repository:** Added as **Git submodule** at `temp_nicheformer/` pointing to https://github.com/theislab/nicheformer.git
- **Commit:** `485cadbc5caa15119adfd54228f8a8af835fcabc` (main branch)
- **Package:** Installed in editable mode in `niche` conda environment
- **Dependencies:** PyTorch Lightning, torchmetrics, wandb, and other core dependencies installed
- **Status:** Fully functional and tested

**Note:** `temp_nicheformer` is a Git submodule, not a regular directory. This means:
- It points to the official Nicheformer repository
- No code duplication in your repository
- To clone this repo with submodules: `git clone --recurse-submodules <repo-url>`
- To update submodule after cloning: `git submodule update --init --recursive`

### Git Submodule Details

**What is a Git Submodule?**

A Git submodule allows you to include one Git repository as a subdirectory of another Git repository. Instead of copying the entire Nicheformer codebase, it creates a **reference** to the official repository at a specific commit.

**Benefits:**
- ✅ **No code duplication** - Points to the official repo, doesn't copy it
- ✅ **Easy updates** - Can update to newer commits when needed
- ✅ **Smaller repository** - Your repo stays small
- ✅ **Version control** - Tracks which commit of Nicheformer you're using
- ✅ **Clean separation** - Nicheformer code stays separate from your code

**For Repository Owner:**
- The submodule is tracked in `.gitmodules` file
- You commit the submodule reference (not the code)
- To update to a newer Nicheformer version:
  ```bash
  cd temp_nicheformer
  git checkout main  # or specific commit/tag
  git pull
  cd ..
  git add temp_nicheformer
  git commit -m "Update Nicheformer submodule to latest"
  ```

**For Others (Cloning Your Repo):**
They need to initialize submodules:
```bash
# Clone your repo
git clone <your-repo-url>

# Initialize and update submodules
git submodule update --init --recursive

# Or clone with submodules in one command
git clone --recurse-submodules <your-repo-url>
```

After cloning, they still need to install Nicheformer:
```bash
conda activate niche  # or their environment
cd temp_nicheformer
pip install -e .
```

### ✅ Pretrained Weights

- **Location:** `data/pretrained/nicheformer_pretrained.ckpt` (545.71 MB)
- **Source:** Downloaded from [Mendeley Data](https://data.mendeley.com/preview/87gm9hrgm8)
- **Status:** Successfully tested and working

**Test Results:**
- ✅ Model loading: Successfully loads pretrained checkpoint
- ✅ Forward pass: Works correctly (input: [4, 250], output: [4, 22])
- ✅ Loss computation: Cross-entropy loss works
- ✅ Parameter counts: 25.5M total, 267K trainable (head_only mode)

### Model Wrapper Status

**Current Implementation:**
- ✅ **Nicheformer-only** - No placeholder fallback
- ✅ **Requires Nicheformer** - Raises clear error if not installed
- ✅ **Pretrained weights support** - Loads checkpoint with `weights_only=False` (PyTorch 2.6+ compatibility)
- ✅ **Fine-tuning modes** - head_only, partial, full all working

**Usage:**
```python
from src.model import create_model

# With pretrained weights
model = create_model(
    num_genes=248,
    num_labels=22,
    pretrained_path="data/pretrained/nicheformer_pretrained.ckpt",
    fine_tune_mode="head_only"
)

# Without pretrained weights (creates new Nicheformer)
model = create_model(
    num_genes=248,
    num_labels=22,
    fine_tune_mode="head_only"
)
```

**Note:** The wrapper requires Nicheformer to be installed. If not available, it will raise an `ImportError` with clear instructions.
- **Milestone 4 Extensions:** Milestone 4 focuses on extending the model with different objectives (neighborhood prediction, interaction modeling, contrastive learning) rather than basic integration. The model wrapper is designed to support these extensions.
- Data loaders assume Milestone 2 data structure. Ensure `data/processed/clients/` exists.
- All functions are device-agnostic (CPU/CUDA).
- The implementation is ready for use by all team members!

---

## Summary

This task provides the foundation for all training activities in Milestone 3. All team members should use these modules rather than implementing their own data loading or training logic. This ensures consistency, reusability, and makes it easier to extend for Milestone 4.

---

## Testing & Verification

Task 4 has been tested and verified by running Task 1 (centralized) and Task 2 (federated) scripts:

✅ **Centralized Test:** 1 epoch completed successfully
- Train: Loss=0.35, Accuracy=0.89, F1=0.86
- Test: Loss=0.32, Accuracy=0.90, F1=0.87

✅ **Federated Test:** 1 round completed successfully  
- Train: Loss=1.99, Accuracy=0.41
- Test: Loss=1.17, Accuracy=0.69, F1=0.46

All outputs generated correctly (model, history, metrics, plots).

**See:** `docs/milestone3/task4/TEST_RESULTS.md` for detailed test results.

---

**Questions?** Refer to the contracts in `docs/milestone3/contracts/` or check the test script `scripts/milestone3/test_training_engine.py` for usage examples.
