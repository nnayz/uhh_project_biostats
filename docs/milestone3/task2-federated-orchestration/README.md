# Task 2: Federated Orchestration (Flower + FedAvg)

| Metadata | Details |
| --- | --- |
| **Owner** | Nasrul |
| **Milestone** | 3 |
| **Status** | ✅ Complete |

## 1. Overview

This document details the implementation of the **federated learning orchestration layer** using the Flower framework with the FedAvg aggregation strategy. The implementation treats model training as a black box, using the shared training engine functions (`train_one_epoch`, `evaluate`) from Task 4.

### Key Responsibilities

- **Coordinate multiple clients** - Manage data loading and local training for each client
- **Distribute global model** - Broadcast model parameters to clients each round
- **Aggregate updates using FedAvg** - Weighted averaging based on local sample counts
- **Log per-round metrics** - Track training and validation performance

### Contract Reference

- `docs/milestone3/contracts/federation_contract.md`
- `docs/milestone3/contracts/training_contract.md`
- `docs/milestone3/contracts/data_contract.md`
- `docs/milestone3/contracts/model_contract.md`

---

## 2. Setup & Installation

### Prerequisites

1. **Milestone 2 Processed Dataset**
   
   Download and extract the shared dataset:
   ```bash
   # Download from Google Drive
   # https://drive.google.com/file/d/1cPzEAUFvLVdi0_cgKTXQl1WPJSqADda4/view?usp=sharing
   
   # Extract to data/processed/
   unzip processed_data.zip -d data/
   ```

2. **Install Dependencies**
   
   ```bash
   pip install -r requirements.txt
   ```
   
   This includes the Flower framework:
   ```
   flwr>=1.5.0  # Flower for federated learning
   ```

### Verify Installation

```bash
python -c "import flwr; print(f'Flower version: {flwr.__version__}')"
```

---

## 3. File Structure

```
src/training/
├── __init__.py           # Updated exports for FL modules
├── train_engine.py       # Shared training functions (from Task 4)
├── fl_client.py          # Flower client implementation
└── fl_server.py          # Server utilities and FedAvg strategy

scripts/
└── run_federated.py      # Main entry point for federated training
```

---

## 4. Implementation Details

### `fl_client.py` - Flower Client

**Purpose:** Implements a Flower NumPyClient that performs local training on each client's data.

**Key Components:**

1. **State Dict Conversion Helpers**
   ```python
   state_dict_to_ndarrays(state_dict) -> List[np.ndarray]
   ndarrays_to_state_dict(model, ndarrays) -> Dict[str, torch.Tensor]
   ```
   - Deterministic conversion between PyTorch state_dict and NumPy arrays
   - Uses model's state_dict keys to maintain stable ordering

2. **FlowerClient Class**
   - Inherits from `fl.client.NumPyClient`
   - Loads local train/val data via Data Contract API
   - Uses Training Contract functions as a black box
   
   **Methods:**
   - `get_parameters()` - Return current model weights as NumPy arrays
   - `set_parameters()` - Load weights from NumPy arrays
   - `fit()` - Run local training for N epochs, return updated weights + metrics
   - `evaluate()` - Evaluate on local validation set, return loss + metrics

3. **Client Factory Function**
   ```python
   create_client_fn(client_ids, data_dir, local_epochs, batch_size, ...)
   ```
   - Maps Flower's numeric IDs ("0", "1", ...) to actual client folders
   - Used by `fl.simulation.start_simulation()`

### `fl_server.py` - Server Utilities

**Purpose:** Provides server-side strategy and metric aggregation utilities.

**Key Components:**

1. **Metric Aggregation Functions**
   ```python
   weighted_average(metrics) -> Metrics  # Weighted by num_examples
   aggregate_fit_metrics(results) -> Metrics
   aggregate_evaluate_metrics(results) -> Metrics
   ```

2. **Strategy Builder**
   ```python
   create_fedavg_strategy(
       initial_parameters,
       fraction_fit, fraction_evaluate,
       min_fit_clients, min_evaluate_clients,
       on_fit_config_fn, on_evaluate_config_fn
   ) -> FedAvg
   ```

3. **Round Configuration Helpers**
   ```python
   get_on_fit_config_fn(local_epochs)  # Returns config for client training
   get_on_evaluate_config_fn()         # Returns config for client evaluation
   ```

4. **RoundLogger Class**
   - Tracks per-round metrics during training
   - Converts to JSON or DataFrame for saving

### `run_federated.py` - Main Entry Point

**Purpose:** Command-line interface for running federated learning simulation.

**Key Features:**
- Discovers clients from `data_dir/clients/client_*`
- Validates data files exist before training
- Runs Flower simulation mode (single-process)
- Saves all required artifacts
- Performs final evaluation on test sets

---

## 5. Usage

### CLI Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--data_dir` | Processed data directory | `data/processed` |
| `--output_dir` | Output directory for artifacts | `results/federated` |
| `--num_rounds` | Number of federated rounds | `5` |
| `--clients_per_round` | Clients sampled per round | `2` |
| `--local_epochs` | Local training epochs per round | `1` |
| `--batch_size` | Training batch size | `1024` (GPU optimized) |
| `--lr` | Learning rate | `1e-4` |
| `--weight_decay` | Weight decay | `1e-5` |
| `--fine_tune_mode` | `head_only`, `partial`, or `full` | `head_only` |
| `--include_spatial` | Include spatial coordinates | `True` |
| `--no_spatial` | Disable spatial coordinates | `False` |
| `--pretrained_path` | Path to pretrained checkpoint | `None` |
| `--device` | `cpu` or `cuda` | `cpu` |
| `--num_workers` | Data loading workers (0=main thread, 4-8 for GPU) | `4` |
| `--use_amp` | Enable Automatic Mixed Precision (GPU) | `True` |
| `--no_amp` | Disable AMP | `False` |
| `--seed` | Random seed | `42` |
| `--verbose` / `--quiet` | Output verbosity | `verbose` |

### Execution Commands

#### 1. Smoke Test (Sanity Check)

Run a quick test with minimal rounds to verify everything works:

```bash
python scripts/run_federated.py \
  --data_dir data/processed \
  --output_dir results/federated_smoke \
  --num_rounds 2 \
  --clients_per_round 2 \
  --local_epochs 1 \
  --batch_size 1024 \
  --device cuda \
  --num_workers 4 \
  --use_amp
```

#### 2. Main Run (GPU Optimized)

```bash
python scripts/run_federated.py \
  --data_dir data/processed \
  --output_dir results/federated \
  --num_rounds 10 \
  --clients_per_round 3 \
  --local_epochs 2 \
  --batch_size 1024 \
  --lr 1e-4 \
  --fine_tune_mode head_only \
  --pretrained_path data/pretrained/nicheformer_pretrained.ckpt \
  --device cuda \
  --num_workers 4 \
  --use_amp
```

#### 3. With Pretrained Weights (GPU Optimized)

```bash
python scripts/run_federated.py \
  --data_dir data/processed \
  --output_dir results/federated_pretrained \
  --num_rounds 5 \
  --clients_per_round 3 \
  --local_epochs 2 \
  --batch_size 1024 \
  --lr 1e-4 \
  --fine_tune_mode head_only \
  --pretrained_path data/pretrained/nicheformer_pretrained.ckpt \
  --device cuda \
  --num_workers 4 \
  --use_amp
```

**GPU Optimizations:**
- `--batch_size 1024` - Increased for better GPU utilization (4x default)
- `--num_workers 4` - Parallel data loading to prevent GPU starvation
- `--use_amp` - Automatic Mixed Precision for ~2x speedup
- Expected GPU utilization: 80-95%

**For CPU-only systems:**
- Use `--device cpu`
- Reduce `--batch_size` to 256 or 128
- Set `--num_workers 0` (or omit)
- Use `--no_amp` (AMP is CUDA-only)

---

## 6. Outputs

After a successful run, `--output_dir` will contain:

| File | Description |
| --- | --- |
| `config.json` | All hyperparameters and configuration |
| `history.json` | Round-level metrics (raw Flower history) |
| `metrics.csv` | Tabular metrics per round |
| `model_final.pt` | Final aggregated global model weights |
| `eval_summary.json` | Per-client and global test evaluation |
| `plots/loss_curve.png` | Training/validation loss over rounds |
| `plots/accuracy_curve.png` | Accuracy over rounds |
| `plots/f1_macro_curve.png` | F1 macro score over rounds |

### Example `metrics.csv`

```csv
round,train_loss,train_accuracy,train_f1_macro,val_loss,val_accuracy,val_f1_macro
1,2.3456,0.2345,0.1234,2.4567,0.2234,0.1123
2,1.8765,0.3456,0.2345,1.9876,0.3345,0.2234
3,1.4567,0.4567,0.3456,1.5678,0.4456,0.3345
```

### Example `eval_summary.json`

```json
{
  "global_test": {
    "loss": 1.2345,
    "accuracy": 0.5678,
    "f1_macro": 0.4567,
    "total_samples": 15000
  },
  "per_client_test": {
    "client_01": {
      "loss": 1.1234,
      "accuracy": 0.5890,
      "f1_macro": 0.4780,
      "num_samples": 5000
    },
    "client_02": { ... },
    "client_03": { ... }
  },
  "clients": ["client_01", "client_02", "client_03"]
}
```

---

## 7. Architecture & Design

### FedAvg Algorithm

The implementation follows the standard FedAvg (Federated Averaging) algorithm:

```
1. Server initializes global model parameters θ₀
2. For each round r = 1, 2, ..., R:
   a. Server selects K clients (or all clients)
   b. Server broadcasts θᵣ₋₁ to selected clients
   c. Each client k:
      - Sets local model to θᵣ₋₁
      - Trains for E local epochs on local data
      - Returns updated weights θₖ and sample count nₖ
   d. Server aggregates:
      θᵣ = Σₖ (nₖ / Σₖ nₖ) * θₖ
3. Final global model θᵣ is saved
```

### Key Design Decisions

1. **Simulation Mode**
   - Uses `fl.simulation.start_simulation()` for single-process execution
   - All clients run sequentially in the same process
   - No network communication overhead

2. **Parameter Weighting**
   - FedAvg weights client updates by `num_examples` (local sample count)
   - Clients with more data have proportionally more influence

3. **Black Box Training**
   - Client calls `train_one_epoch()` and `evaluate()` from training engine
   - No custom training loops beyond orchestration

4. **Custom Strategy for Parameter Saving**
   - `FedAvgWithParameterSaving` extends `FedAvg` to capture final parameters
   - Flower's default history doesn't expose aggregated parameters

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Flower Server                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  FedAvgWithParameterSaving Strategy                         │   │
│  │  - initial_parameters (from initial model)                   │   │
│  │  - aggregate_fit_metrics / aggregate_evaluate_metrics        │   │
│  │  - Saves aggregated_parameters after each round              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Client 0    │     │  Client 1    │     │  Client 2    │
│  (client_01) │     │  (client_02) │     │  (client_03) │
├──────────────┤     ├──────────────┤     ├──────────────┤
│ FlowerClient │     │ FlowerClient │     │ FlowerClient │
│ - load_data  │     │ - load_data  │     │ - load_data  │
│ - fit()      │     │ - fit()      │     │ - fit()      │
│ - evaluate() │     │ - evaluate() │     │ - evaluate() │
└──────────────┘     └──────────────┘     └──────────────┘
       │                     │                     │
       │  train_one_epoch()  │                     │
       │  evaluate()         │                     │
       │  (from train_engine)│                     │
       └─────────────────────┴─────────────────────┘
```

---

## 8. Integration with Other Tasks

### Using Shared Components

The federated implementation uses the following from Task 4:

```python
# Data loading (Data Contract)
from src.data import load_client_data, load_gene_list, load_label_map, create_dataloader

# Model (Model Contract)
from src.model import create_model
# model.get_weights() / model.set_weights() for parameter exchange

# Training (Training Contract)
from src.training import train_one_epoch, evaluate, create_optimizer, save_model
```

### For Evaluation Task (Task 5)

The federated runner produces artifacts ready for evaluation:

```python
# Load final federated model
import torch
from src.model import create_model

model = create_model(num_genes=248, num_labels=22)
model.load_state_dict(torch.load("results/federated/model_final.pt"))

# Use evaluation functions
from src.training import evaluate
metrics = evaluate(model, test_loader, device)
```

### Comparison with Centralized (Task 1)

Both runners produce comparable outputs:

| Metric | Centralized | Federated |
| --- | --- | --- |
| Output format | Same | Same |
| `metrics.csv` | Per-epoch | Per-round |
| `model_final.pt` | Final pooled model | Final aggregated model |
| Evaluation | Pooled test + per-client | Per-client + weighted global |

---

## 9. Extending for Milestone 4

### Adding FedProx

Replace the strategy in `run_federated.py`:

```python
from flwr.server.strategy import FedProx

strategy = FedProx(
    initial_parameters=initial_parameters,
    proximal_mu=0.1,  # Proximal term coefficient
    ...
)
```

### Personalized Aggregation

Extend `FedAvgWithParameterSaving` to save per-client heads:

```python
class PersonalizedFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_heads = {}  # Store personalized heads
    
    def aggregate_fit(self, server_round, results, failures):
        # Custom aggregation logic
        ...
```

### Client-Specific Heads

Modify `FlowerClient` to maintain local head:

```python
def fit(self, parameters, config):
    # Set only backbone parameters from global model
    self.set_backbone_parameters(parameters)
    # Keep local head unchanged
    ...
```

---

## 10. Troubleshooting

### Common Issues

1. **"No client directories found"**
   ```
   ValueError: No client directories found in data/processed/clients
   ```
   **Solution:** Download and extract the Milestone 2 processed dataset.

2. **"Missing required data files"**
   ```
   FileNotFoundError: Missing required data files:
   data/processed/clients/client_01/train.parquet
   ```
   **Solution:** Ensure all client folders contain `train.parquet`, `val.parquet`, `test.parquet`.

3. **"Could not extract final parameters"**
   ```
   Warning: Could not extract final parameters from strategy.
   ```
   **Cause:** No training rounds completed successfully.
   **Solution:** Check client logs for training errors.

4. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution:** 
   - Reduce `--batch_size` (try 512, 256, or 128)
   - Disable AMP with `--no_amp` (uses more memory but slower)
   - Reduce `--num_workers` to 2 or 0
   - Use `--device cpu` as fallback

5. **Low GPU Utilization**
   ```
   GPU usage: ~20% (expected: 80-95%)
   ```
   **Solution:**
   - Increase `--batch_size` (if memory allows)
   - Increase `--num_workers` to 6 or 8
   - Ensure `--use_amp` is enabled
   - Check for data loading bottlenecks

### Debug Mode

Run with verbose output:

```bash
python scripts/run_federated.py --num_rounds 1 --verbose 2>&1 | tee debug.log
```

---

## 11. Summary

This task provides the complete federated learning orchestration layer for Milestone 3:

| Component | File | Purpose |
| --- | --- | --- |
| Flower Client | `src/training/fl_client.py` | Local training on client data |
| Server Utilities | `src/training/fl_server.py` | FedAvg strategy and aggregation |
| Main Script | `scripts/run_federated.py` | CLI entry point for experiments |

**Key Features:**
- ✅ Flower simulation mode (single-process, no network)
- ✅ FedAvg with sample-weighted aggregation
- ✅ Deterministic state_dict ↔ NumPy conversion
- ✅ Per-round metric logging
- ✅ Final model saving and evaluation
- ✅ Training curves and plots

**Questions?** Refer to the contracts in `docs/milestone3/contracts/` or the implementation files directly.
