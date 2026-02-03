# Training Summary

Results and analysis for all three training strategies: **Centralized**, **Federated**, and **Local**.

---

## Overview

All models are evaluated on the **same held-out test set** (Replicate 3, 157,982 samples).

| Strategy | Test Accuracy | Test Macro-F1 | Test Loss |
|----------|---------------|---------------|-----------|
| **Centralized** | **94.45%** | **94.09%** | **0.154** |
| **Federated** | 92.76% | 86.84% | 0.204 |
| **Local (mean)** | 76.75% | 67.82% | 1.736 |

---

## 1. Centralized Training

**Best performance** — pooling all training data yields optimal results.

### Configuration

| Parameter | Value |
|-----------|-------|
| Training samples | 316,563 (all clients pooled) |
| Epochs | 10 |
| Batch size | 1024 |
| Learning rate | 0.0001 |
| Fine-tune mode | head_only |
| Pretrained weights | No |

### Results

| Metric | Validation | Test (held-out) |
|--------|------------|-----------------|
| Accuracy | 94.02% | 94.45% |
| Macro-F1 | 93.67% | 94.09% |
| Loss | 0.167 | 0.154 |

### Convergence

- Smooth convergence with no overfitting
- Converged by epoch 6-7
- Final validation loss: 0.167

### Why Centralized Works Best

- Sees **all 316K samples** with the full global distribution
- Can learn **all 24 cell types** with representative samples
- No distribution shift between training and evaluation

---

## 2. Federated Training

**Competitive alternative** — only 1.69% below centralized in accuracy.

### Configuration

| Parameter | Value |
|-----------|-------|
| Training samples | ~105K per client per round |
| Rounds | 10 |
| Local epochs per round | 3 |
| Clients per round | 3 |
| Batch size | 512 |
| Learning rate | 0.0001 |
| Fine-tune mode | head_only |
| Pretrained weights | Yes (Nicheformer) |
| Aggregation | FedAvg |

### Results

| Metric | Validation | Test (held-out) |
|--------|------------|-----------------|
| Accuracy | — | 92.76% |
| Macro-F1 | — | 86.84% |
| Loss | 0.216 | 0.204 |

### Convergence

- Started at loss 1.176 (round 1)
- Decreased steadily to 0.216 (round 10)
- Each round: 3 local epochs per client before aggregation

### Gap Analysis vs Centralized

| Metric | Centralized | Federated | Gap |
|--------|-------------|-----------|-----|
| Accuracy | 94.45% | 92.76% | -1.69% |
| Macro-F1 | 94.09% | 86.84% | **-7.25%** |
| Loss | 0.154 | 0.204 | +0.050 |

**Why the F1 gap is larger than the accuracy gap:**

- **Accuracy** = overall correctness (dominated by majority classes)
- **Macro-F1** = equal weight to all classes, including minorities
- Federated struggles with **rare cell types**:
  - Each client sees a non-IID subset where some classes are under-represented
  - FedAvg helps but doesn't fully recover class balance
  - Minority classes may be majority in another client, creating conflicting gradients

### Why Federated Still Performs Well

- **Knowledge sharing** across clients via aggregation
- Pretrained Nicheformer weights provide a strong starting point
- Multiple rounds allow the model to see all data distributions

---

## 3. Local Training

**Significantly underperforms** — each client only sees one anatomical region.

### Configuration

| Parameter | Value |
|-----------|-------|
| Training samples | ~84K (one client only) |
| Epochs | 10 |
| Batch size | 1024 |
| Learning rate | 0.0008 |
| Fine-tune mode | head_only |
| Pretrained weights | No |

### Per-Client Results

| Client | Region | Test Accuracy | Test Macro-F1 | Test Loss |
|--------|--------|---------------|---------------|-----------|
| client_01 | dorsal | 61.98% | 54.19% | 3.334 |
| client_02 | mid | 90.19% | 83.24% | 0.564 |
| client_03 | ventral | 78.08% | 66.02% | 1.309 |
| **Mean** | — | **76.75%** | **67.82%** | **1.736** |

### Client Analysis

**client_01 (dorsal) — Worst performer (61.98%):**
- Has all 24 classes in training but poor generalization
- Dorsal region's cell type mix differs from the held-out set
- JSD to global: 0.152 (lowest, but still significant)

**client_02 (mid) — Best local performer (90.19%):**
- Approaches federated performance
- Mid region likely has more overlap with held-out set's distribution
- JSD to global: 0.273 (highest, yet performs best)

**client_03 (ventral) — Middle performer (78.08%):**
- Missing one class (23 vs 24)
- Limited ability to recognize all cell types
- JSD to global: 0.252

### Per-Client Comparison: Federated vs Local

| Client | Federated Acc | Local Acc | Gap |
|--------|---------------|-----------|-----|
| client_01 | 92.76% | 61.98% | **+30.78%** |
| client_02 | 92.76% | 90.19% | +2.57% |
| client_03 | 92.76% | 78.08% | +14.68% |

**Key insight:** The federated model is **consistent** (92.76% for all), while local models **vary wildly** (61.98% to 90.19%).

### Why Local Fails

- Each local model only sees ~105K samples from **one anatomical region**
- **Missing cell types**: Some labels have <0.1% representation in certain regions
- Cannot generalize to cell types it has never (or rarely) seen
- Example: Model trained on dorsal struggles with Label 12 (0.02% in dorsal, 21% in ventral)

---

## Strategy Comparison

### What Each Strategy Sees

| Strategy | Training Data | Notes |
|----------|---------------|-------|
| **Centralized** | All 316,563 samples pooled | Best data efficiency; sees global distribution |
| **Federated** | Each client separately, then aggregates | Non-IID per round; aggregation aids generalization |
| **Local** | Single client only (~105K) | Only one anatomical region; may miss cell types |

### Performance Ranking

```
Centralized (94.45%) > Federated (92.76%) > Local Mean (76.75%)
     ↑                       ↑                    ↑
Best overall          Close to centralized   High variance
                      (-1.69% accuracy)      (61.98% - 90.19%)
```

---

## Conclusions

1. **Centralized is the gold standard** when data can be pooled — 94.45% accuracy with excellent class balance (94.09% F1)

2. **Federated is a strong alternative** when data cannot be centralized — only 1.69% accuracy drop, though 7.25% F1 drop indicates room for improvement on minority classes

3. **Local training is insufficient** for this dataset — the non-IID nature means each client sees a biased distribution, leading to poor generalization

4. **Federated learning's value proposition:** achieves near-centralized performance while respecting data locality constraints

---

## Output Files

### Centralized (`results/centralized/`)
- `model_final.pt` — Final model weights
- `config.json` — Training configuration
- `history.json` — Per-epoch metrics
- `eval_summary.json` — Test set evaluation
- `plots/` — Training curves, confusion matrix

### Federated (`results/federated/`)
- `model_final.pt` — Aggregated model weights
- `config.json` — Training configuration
- `history.json` — Per-round metrics
- `per_client_metrics.csv` — Per-client, per-round metrics
- `plots/` — Training curves, per-client curves

### Local (`results/local/local_client_XX/`)
- Per-client: `model_final.pt`, `config.json`, `history.json`, `eval_summary.json`, `plots/`
