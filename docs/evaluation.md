# Evaluation

Scripts and process for evaluating and comparing trained models.

---

## 1. Overview

| Script | Purpose |
|--------|---------|
| `scripts/milestone3/evaluate_models.py` | Load trained models, evaluate on held-out set, generate comparison |
| `scripts/milestone3/visualize_evaluation.py` | Generate UMAPs, additional visualizations |
| `scripts/analysis/run_analysis.py` | Dataset analysis (run after partitioning, before training) |

**Evaluation principle:** All models evaluated on the **same held-out test set** (`held_out_batch.parquet` = Replicate 3).

---

## 2. evaluate_models.py

Main evaluation and comparison script.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--centralized_dir` | `results/centralized` | Centralized model directory |
| `--federated_dir` | `results/federated` | Federated model directory |
| `--local_results_dir` | `results/local` | Local models directory |
| `--data_dir` | `data/processed` | Processed data directory |
| `--output_dir` | `results/comparison` | Output for comparison |
| `--no_local` | False | Exclude local models |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--batch_size` | 1024 | Batch size |
| `--use_amp` | False | Automatic Mixed Precision |

### Usage

```bash
# Full comparison (all strategies)
python scripts/milestone3/evaluate_models.py \
    --device cuda \
    --batch_size 1024 \
    --use_amp

# Without local models
python scripts/milestone3/evaluate_models.py --no_local
```

### Outputs

| File | Description |
|------|-------------|
| `evaluation_summary.json` | Full metrics (JSON format) |
| `evaluation_summary.md` | Human-readable analysis |
| `overall_metrics_comparison.png` | Bar chart: Accuracy & F1 |
| `per_client_comparison.png` | Per-client accuracy & F1 |
| `confusion_matrix_centralized.png` | Centralized confusion matrix |
| `confusion_matrix_federated.png` | Federated confusion matrix |

---

## 3. visualize_evaluation.py

Additional visualizations using processed data and results.

```bash
python scripts/milestone3/visualize_evaluation.py
```

**Outputs:**
- UMAP colored by batch ID
- UMAP colored by cell type
- Scatter of evaluation performance

---

## 4. Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Accuracy** | Overall correctness | General performance |
| **Macro-F1** | Class-balanced F1 | Minority class performance |
| **Loss** | Cross-entropy loss | Training convergence |

### Interpreting Gaps

| Gap | Meaning |
|-----|---------|
| Small accuracy, large F1 | Struggles with minority classes |
| Large accuracy gap | Fundamental performance issue |
| High loss | Poor confidence/calibration |

---

## 5. Evaluation Workflow

```
1. Train models
   ├── scripts/run_centralized.py → results/centralized/
   ├── scripts/run_federated.py → results/federated/
   └── scripts/run_local.py → results/local/

2. Evaluate and compare
   └── scripts/milestone3/evaluate_models.py → results/comparison/

3. (Optional) Additional visualizations
   └── scripts/milestone3/visualize_evaluation.py
```

---

## 6. Output Directory Structure

```
results/
├── centralized/
│   ├── model_final.pt
│   ├── config.json
│   ├── history.json
│   ├── metrics.csv
│   ├── eval_summary.json
│   └── plots/
├── federated/
│   ├── model_final.pt
│   ├── config.json
│   ├── history.json
│   ├── per_client_metrics.csv
│   └── plots/
├── local/
│   ├── local_client_01/
│   ├── local_client_02/
│   └── local_client_03/
└── comparison/
    ├── evaluation_summary.json
    ├── evaluation_summary.md
    ├── overall_metrics_comparison.png
    ├── per_client_comparison.png
    └── confusion_matrix_*.png
```

---

## 7. Per-Run Outputs

Each training script automatically evaluates on the held-out set and saves:

| File | Description |
|------|-------------|
| `eval_summary.json` | Test metrics (accuracy, F1, loss) |
| `metrics.csv` | Metrics in CSV format |
| `plots/confusion_matrix.png` | Predictions vs actual |

The comparison script (`evaluate_models.py`) consolidates these into a unified comparison.

---

## 8. Results Documentation

For detailed evaluation results, see:
- [docs/evaluation_results.md](evaluation_results.md) — Full comparison analysis
- [docs/training_summary.md](training_summary.md) — Training results by strategy
- [docs/analysis_summary.md](analysis_summary.md) — Dataset & client analysis
