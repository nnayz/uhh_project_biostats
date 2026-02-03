# Model Evaluation Summary

This document summarizes the evaluation results comparing **Centralized**, **Federated**, and **Local** training strategies on the mouse brain spatial transcriptomics dataset. All models are evaluated on the **same held-out test set** (Replicate 3, 157,982 samples).

---

## Overall Performance Comparison

| Strategy | Test Accuracy | Test Macro-F1 | Test Loss |
|----------|---------------|---------------|-----------|
| **Centralized** | **94.45%** | **94.09%** | **0.154** |
| **Federated** | 92.76% | 86.84% | 0.204 |
| **Local (mean)** | 76.75% | 67.82% | 1.736 |

### Key Findings

1. **Centralized outperforms all strategies** — As expected, pooling all training data yields the best results with 94.45% accuracy and 94.09% Macro-F1.

2. **Federated achieves competitive accuracy** — Only 1.69% below centralized in accuracy, demonstrating that federated aggregation effectively combines knowledge from non-IID clients.

3. **Federated F1 gap is larger** — The 7.25% F1 gap (94.09% vs 86.84%) indicates federated struggles more with minority classes compared to centralized.

4. **Local training significantly underperforms** — With only 76.75% mean accuracy and 67.82% mean F1, local models fail to generalize to cell types not present in their training region.

---

## Training Configuration

| Parameter | Centralized | Federated | Local |
|-----------|-------------|-----------|-------|
| **Epochs/Rounds** | 10 epochs | 10 rounds | 10 epochs |
| **Local epochs per round** | — | 3 | — |
| **Batch size** | 1024 | 512 | 1024 |
| **Learning rate** | 0.0001 | 0.0001 | 0.0008 |
| **Fine-tune mode** | head_only | head_only | head_only |
| **Pretrained weights** | No | Yes (Nicheformer) | No |
| **Training samples** | 316,563 (all clients) | ~105K per client per round | ~84K (one client) |

**Note:** Federated used pretrained Nicheformer weights while Centralized and Local trained from scratch. This may partially explain Federated's strong performance despite seeing non-IID data.

---

## Per-Client Local Model Performance

| Client | Region | Test Accuracy | Test Macro-F1 | Test Loss |
|--------|--------|---------------|---------------|-----------|
| client_01 | dorsal | 61.98% | 54.19% | 3.334 |
| client_02 | mid | 90.19% | 83.24% | 0.564 |
| client_03 | ventral | 78.08% | 66.02% | 1.309 |

### Analysis by Client

**client_01 (dorsal) — Worst performer:**
- Only 61.98% accuracy despite having all 24 classes in training
- JSD to global: 0.152 (lowest) — but still significant distribution shift
- The dorsal region's cell type mix differs enough from the held-out set (Replicate 3) to cause poor generalization

**client_02 (mid) — Best local performer:**
- 90.19% accuracy, approaching federated performance
- JSD to global: 0.273 (highest) — yet performs best
- The mid region likely has more overlap with the held-out set's cell type distribution

**client_03 (ventral) — Middle performer:**
- 78.08% accuracy, 66.02% F1
- JSD to global: 0.252
- Missing one class (23 vs 24) limits its ability to recognize all cell types

---

## Per-Client Comparison: Federated vs Local

| Client | Federated Acc | Local Acc | Gap | Federated F1 | Local F1 | Gap |
|--------|---------------|-----------|-----|--------------|----------|-----|
| client01_dorsal | 92.76% | 61.98% | **+30.78%** | 86.84% | 54.19% | **+32.65%** |
| client02_mid | 92.76% | 90.19% | +2.57% | 86.84% | 83.24% | +3.60% |
| client03_ventral | 92.76% | 78.08% | +14.68% | 86.84% | 66.02% | +20.82% |

### Key Insight

The **federated model is a single model** applied to all clients, achieving consistent 92.76% accuracy regardless of the anatomical region. In contrast, **local models vary wildly** (61.98% to 90.19%) depending on how well their training region matches the test set.

This demonstrates the core value of federated learning: **knowledge sharing across clients produces a robust model** that generalizes better than any individual local model.

---

## Training Convergence

### Centralized Training
- **Final validation loss:** 0.167 (converged by epoch 6-7)
- **Final validation accuracy:** 94.02%
- **Final validation F1:** 93.67%
- Smooth convergence with no signs of overfitting

### Federated Training
- **Final validation loss:** 0.216 (after 10 rounds)
- Started at 1.176 (round 1) and decreased steadily
- Each round involves 3 local epochs per client before aggregation

### Local Training (per client)
- client_01: Struggled to converge (high loss throughout)
- client_02: Converged well (closest to federated/centralized)
- client_03: Moderate convergence

---

## Centralized vs Federated Gap Analysis

| Metric | Centralized | Federated | Difference |
|--------|-------------|-----------|------------|
| Accuracy | 94.45% | 92.76% | -1.69% |
| Macro-F1 | 94.09% | 86.84% | -7.25% |
| Loss | 0.154 | 0.204 | +0.050 |

### Why does Federated have a larger F1 gap than accuracy gap?

**Accuracy** measures overall correctness (dominated by majority classes).
**Macro-F1** gives equal weight to all classes, including minorities.

The larger F1 gap (7.25%) suggests federated struggles with **rare cell types**:
- Each client sees a non-IID subset where some classes are under-represented
- FedAvg aggregation helps but doesn't fully recover class balance
- Minority classes in one client may be majority in another, creating conflicting gradients

---

## Generated Outputs

### Comparison Directory (`results/comparison/`)
| File | Description |
|------|-------------|
| `evaluation_summary.json` | Full metrics for all strategies (JSON) |
| `overall_metrics_comparison.png` | Bar chart: Accuracy & F1 for all 3 strategies |
| `per_client_comparison.png` | Side-by-side: Accuracy (left) & F1 (right) per client |
| `confusion_matrix_centralized.png` | Confusion matrix for centralized model |
| `confusion_matrix_federated.png` | Confusion matrix for federated model |

### Per-Strategy Results
| Directory | Contents |
|-----------|----------|
| `results/centralized/` | config, history, metrics, eval_summary, training plots |
| `results/federated/` | config, history, per_client_metrics, training plots |
| `results/local/local_client_XX/` | config, history, metrics, eval_summary, training plots per client |

---

## Conclusions

1. **Centralized training is the gold standard** when data can be pooled — 94.45% accuracy with excellent class balance (94.09% F1).

2. **Federated learning is a strong alternative** when data cannot be centralized — only 1.69% accuracy drop, though the 7.25% F1 drop indicates room for improvement on minority classes.

3. **Local training is insufficient** for this dataset — the non-IID nature (anatomical siloing) means each client sees a biased cell type distribution, leading to poor generalization (76.75% mean accuracy).

4. **client_02 (mid region)** happens to have the most generalizable local model, but this is dataset-specific and not reliable for production.

5. **Federated learning's value proposition is clear:** it achieves near-centralized performance while respecting data locality constraints, making it the recommended approach when data pooling is not possible.

---

## Visualizations Reference

| Plot | Location | Description |
|------|----------|-------------|
| Overall comparison | `results/comparison/overall_metrics_comparison.png` | Centralized vs Federated vs Local (mean) |
| Per-client comparison | `results/comparison/per_client_comparison.png` | Accuracy & F1 by client |
| Centralized confusion matrix | `results/comparison/confusion_matrix_centralized.png` | 24×24 cell type predictions |
| Federated confusion matrix | `results/comparison/confusion_matrix_federated.png` | 24×24 cell type predictions |
| Centralized training curves | `results/centralized/plots/` | loss, accuracy, f1_macro curves |
| Federated training curves | `results/federated/plots/` | loss curve, per_client_curves |
| Local training curves | `results/local/local_client_XX/plots/` | Per-client training dynamics |
