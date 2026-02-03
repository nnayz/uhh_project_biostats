# Source Code

Core modules for data loading, model wrapper, and training engine.

## Structure

```
src/
├── data/
│   └── loaders.py           # Data loaders API
├── model/
│   └── nicheformer_wrapper.py  # Model wrapper
├── training/
│   ├── train_engine.py      # Training functions
│   ├── fl_client.py         # Flower client
│   └── fl_server.py         # Flower server
└── config.py                # Configuration utilities
```

## Quick Reference

### Data Loaders (`src/data/loaders.py`)

```python
from src.data import load_gene_list, load_client_data, create_dataloader

genes = load_gene_list()
df = load_client_data("client_01", "train")
loader = create_dataloader(df, genes, batch_size=1024)
```

### Model (`src/model/nicheformer_wrapper.py`)

```python
from src.model import create_model

model = create_model(
    num_genes=248,
    num_labels=24,
    fine_tune_mode="head_only"
)
```

### Training (`src/training/train_engine.py`)

```python
from src.training import train_one_epoch, evaluate, create_optimizer

optimizer = create_optimizer(model, learning_rate=1e-4)
metrics = train_one_epoch(model, loader, optimizer, device)
eval_metrics = evaluate(model, test_loader, device)
```

## Documentation

See [docs/training.md](../docs/training.md) for full API documentation.
