# Training

Training scripts and infrastructure for centralized, federated, and local training.

---

## 1. Training Strategies

| Strategy | Data | Evaluation |
|----------|------|------------|
| **Centralized** | All clients pooled | Held-out set |
| **Federated** | Each client trains locally; FedAvg aggregation | Held-out set |
| **Local** | Single client only | Held-out set |

All models are evaluated on the **same held-out set** (`held_out_batch.parquet`).

---

## 2. Scripts

### run_centralized.py

Trains one model on pooled data from all clients.

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/processed` | Processed data directory |
| `--output_dir` | `results/centralized` | Output directory |
| `--epochs` | 10 | Training epochs |
| `--batch_size` | 1024 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--fine_tune_mode` | `head_only` | `head_only` / `partial` / `full` |
| `--pretrained_path` | None | Nicheformer checkpoint |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--use_amp` | True | Automatic Mixed Precision |

```bash
python scripts/run_centralized.py --device cuda --epochs 10 --use_amp
```

### run_federated.py

Federated training with Flower (FedAvg simulation).

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_rounds` | 5 | Federated rounds |
| `--clients_per_round` | 3 | Clients per round |
| `--local_epochs` | 1 | Epochs per client per round |
| `--batch_size` | 1024 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--device` | `cpu` | `cpu` or `cuda` |

```bash
python scripts/run_federated.py --device cuda --num_rounds 10 --local_epochs 3
```

### run_local.py

Trains on a single client's data.

| Argument | Default | Description |
|----------|---------|-------------|
| `--client_id` | Required | `client_01`, `client_02`, `client_03`, or `all` |
| `--epochs` | 10 | Training epochs |
| `--output_dir` | `results/local/local_{client_id}` | Output directory |

```bash
# Single client
python scripts/run_local.py --client_id client_01 --device cuda

# All clients
python scripts/run_local.py --client_id all --device cuda
```

---

## 3. Output Structure

Each training run produces:

```
results/{strategy}/
├── model_final.pt        # Final model weights
├── config.json           # Training configuration
├── history.json          # Per-epoch/round metrics
├── metrics.csv           # Final evaluation metrics
├── eval_summary.json     # Evaluation on held-out set
└── plots/
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── f1_macro_curve.png
    └── confusion_matrix.png
```

Federated also includes:
- `per_client_metrics.csv` — Per-round, per-client metrics
- `per_client_curves.png` — Training curves by client

---

## 4. Data Loaders

API in `src/data/loaders.py`:

| Function | Description |
|----------|-------------|
| `load_gene_list(data_dir)` | Gene names (248) |
| `load_label_map(data_dir)` | Label mapping (24 classes) |
| `load_client_data(client_id, split)` | Client's train/val DataFrame |
| `load_all_clients(split)` | All clients concatenated |
| `load_global_test()` | Held-out set |
| `create_dataloader(df, genes, ...)` | PyTorch DataLoader |

---

## 5. Model

`src/model/nicheformer_wrapper.py`:

- **Input:** 248 genes + 2 spatial (x, y) = 250 features
- **Output:** 24 cell type logits
- **Fine-tuning modes:**
  - `head_only` — Classifier only (~267K params)
  - `partial` — Last half + head (~12M params)
  - `full` — All parameters (~25.5M params)
- **Methods:** `forward()`, `get_weights()`, `set_weights()`, `compute_loss()`

---

## 6. Training Engine

`src/training/train_engine.py`:

| Function | Description |
|----------|-------------|
| `train_one_epoch()` | One training epoch |
| `evaluate()` | Evaluation with metrics |
| `TrainingHistory` | Tracks metrics per epoch |
| `save_training_artifacts()` | Save model, history, config |
| `create_optimizer()` | Adam optimizer |
| `create_scheduler()` | Cosine annealing |

---

## 7. Federated Learning

Flower framework with FedAvg:

- **Client:** `src/training/fl_client.py`
- **Server:** `src/training/fl_server.py`
- **Mode:** Simulation (single process, sequential clients)

---

## 8. Hyperparameters

| Parameter | Centralized/Local | Federated |
|-----------|-------------------|-----------|
| Epochs | 10 | 3 per round × 10 rounds |
| Batch size | 1024 | 512 |
| Learning rate | 1e-4 | 1e-4 |
| Optimizer | Adam | Adam |
| Weight decay | 1e-5 | 1e-5 |
| AMP | Enabled | Enabled |

---

## 9. Checklist

Before training:

- [ ] Data prepared: `data/processed/clients/client_XX/{train,val}.parquet`
- [ ] Held-out set: `data/processed/held_out_batch.parquet`
- [ ] Genes & labels: `data/processed/genes.txt`, `label_map.json`
- [ ] Optional: `data/pretrained/nicheformer_pretrained.ckpt`

Test with:
```bash
python scripts/milestone3/test_training_engine.py
```
