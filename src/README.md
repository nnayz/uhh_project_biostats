# Model Integration & Shared Training Engine

**Task 4 - Milestone 3**

This module provides the core infrastructure for model integration and training that all other team members rely on.

## Structure

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
└── README.md              # This file
```

## What's Included

### 1. Data Loaders (`src/data/loaders.py`)

Implements the **Data Contract** API:

- `load_client_data(client_id, split)` - Load data for a specific client
- `load_all_clients(split)` - Load data from all clients
- `load_gene_list()` - Get canonical gene list
- `load_label_map()` - Get label mapping
- `create_dataloader(df, genes, ...)` - Create PyTorch DataLoader
- `CellDataset` - PyTorch Dataset class for cells

**Features:**
- Schema validation
- Automatic gene order checking
- Support for spatial coordinates
- Ready for PyTorch training

### 2. Model Wrapper (`src/model/nicheformer_wrapper.py`)

Implements the **Model Contract** interface:

- `forward(batch) -> logits` - Forward pass
- `compute_loss(logits, labels) -> loss` - Loss computation
- `get_weights() -> state_dict` - Get weights for federated aggregation
- `set_weights(state_dict)` - Set weights from federated aggregation

**Features:**
- Three fine-tuning modes: `head_only`, `partial`, `full`
- Configurable spatial coordinate inclusion
- Placeholder backbone (ready for actual Nicheformer integration)
- Parameter counting utilities

### 3. Training Engine (`src/training/train_engine.py`)

Implements the **Training Contract** functions:

- `train_one_epoch(model, dataloader, optimizer, device)` - Train for one epoch
- `evaluate(model, dataloader, device)` - Evaluate model
- `TrainingHistory` - Track training metrics
- `save_training_artifacts()` - Save all required outputs
- `create_optimizer()` - Create optimizer
- `create_scheduler()` - Create learning rate scheduler

**Features:**
- Reusable by centralized and federated training
- Tracks loss, accuracy, F1-macro
- Saves history.json, metrics.csv, model_final.pt, config.json
- Device-agnostic (CPU/CUDA)
- **GPU Optimizations:** AMP support, non-blocking transfers, parallel data loading

### 4. Configuration (`src/config.py`)

Configuration management:

- `TrainingConfig` - Dataclass for all training parameters
- `FineTuningConfig` - Fine-tuning mode configuration
- `create_default_config()` - Factory function with overrides

## Usage

See `docs/milestone3/USAGE.md` for detailed usage examples.

Quick example:

```python
from src.data import load_client_data, load_gene_list, create_dataloader
from src.model import create_model
from src.training import train_one_epoch, evaluate, create_optimizer

# Load data
genes = load_gene_list()
df = load_client_data("client_01", "train")
dataloader = create_dataloader(df, genes, batch_size=32)

# Create model
model = create_model(num_genes=248, num_labels=22, fine_tune_mode="head_only")

# Train
optimizer = create_optimizer(model, learning_rate=1e-4)
metrics = train_one_epoch(model, dataloader, optimizer, torch.device("cpu"))
```

## Testing

Run the test script:

```bash
python scripts/milestone3/test_training_engine.py
```

This validates:
- Data loaders work correctly
- Model wrapper implements contract
- Training engine functions work
- Full integration

## Dependencies

Required packages:
- `torch` - PyTorch
- `pandas` - Data handling
- `numpy` - Numerical operations
- `scikit-learn` - Metrics (accuracy, F1)

## For Other Team Members

### Person 1 (Centralized Training)
Use `train_one_epoch()` and `evaluate()` with data from `load_all_clients()`.

### Person 2 (Federated/Flower)
Use `model.get_weights()` and `model.set_weights()` in Flower client code.
Use `train_one_epoch()` for local training.

### Person 3 (Data & Schema)
This module implements the data loaders! You can extend validation if needed.

### Person 5 (Evaluation)
Use `evaluate()` function. It returns predictions and labels for confusion matrices.

## Design Principles

1. **Modular**: Each component is independent and can be used separately
2. **Contract-based**: Follows contracts defined in `docs/milestone3/contracts/`
3. **Extensible**: Easy to extend for Milestone 4 (spatial modeling, multi-task)
4. **Reusable**: Same functions work for centralized and federated training
5. **Well-documented**: Clear interfaces and usage examples

## Milestone 4 Extensions

The design supports easy extension:

- **Spatial modeling**: Add spatial features in model `forward()` without changing training loop
- **Multi-task learning**: Extend `compute_loss()` to include auxiliary losses
- **Personalization**: Model wrapper can be extended with client-specific heads

The training loop (`train_one_epoch`, `evaluate`) remains unchanged.

## Notes

- The model wrapper currently uses a placeholder transformer backbone. Replace `_create_placeholder_backbone()` with actual Nicheformer loading when the library is integrated.
- Data loaders assume Milestone 2 data structure. Ensure `data/processed/clients/` exists.
- All functions are device-agnostic (CPU/CUDA).
