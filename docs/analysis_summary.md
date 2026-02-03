# Dataset & Client Analysis Summary

This analysis examines the **mouse brain spatial transcriptomics dataset** after anatomical siloing — a federated learning data partitioning strategy where clients are defined by anatomical region (Y-coordinate tertiles: dorsal, mid, ventral) and evaluation uses a completely held-out replicate.

---

## Dataset Overview

| Component | Description | Samples |
|-----------|-------------|---------|
| **Held-out set** | Replicate 3 (evaluation only) | 157,982 |
| **Client 01 (Dorsal)** | Upper Y-coordinate tertile from Replicates 1 & 2 | 105,511 |
| **Client 02 (Mid)** | Middle Y-coordinate tertile from Replicates 1 & 2 | 105,541 |
| **Client 03 (Ventral)** | Lower Y-coordinate tertile from Replicates 1 & 2 | 105,511 |
| **Total training data** | All 3 clients combined | 316,563 |
| **Number of cell types (labels)** | Leiden clusters | 24 |

---

## Training Strategy Comparison

| Strategy | What it sees during training | Notes |
|----------|------------------------------|-------|
| **Centralized** | All 316,563 samples pooled | Best data efficiency; sees global label distribution |
| **Federated** | Each client separately, then aggregates | Sees non-IID data per round; aggregation helps generalization |
| **Local** | Single client only (e.g., 105,511 samples) | Sees only one anatomical region; may miss cell types |

All strategies are evaluated on the **same held-out set** (Replicate 3, 157,982 samples).

---

## Client Statistics

| Client | Region | Total | Train (80%) | Val (20%) | Classes | Max Label Fraction |
|--------|--------|-------|-------------|-----------|---------|-------------------|
| client_01 | dorsal | 105,511 | 84,399 | 21,112 | 24 | 22.08% |
| client_02 | mid | 105,541 | 84,425 | 21,116 | 23 | 20.63% |
| client_03 | ventral | 105,511 | 84,396 | 21,115 | 23 | 20.52% |

**Key observations:**
- Clients are **balanced in size** (~105K samples each)
- Client 01 (dorsal) sees all 24 classes; clients 02 and 03 see only 23 classes (missing one)
- Max label fraction ~20-22% indicates moderate within-client class imbalance

---

## Non-IID Analysis

| Client | Region | Entropy | JSD to Global | Interpretation |
|--------|--------|---------|---------------|----------------|
| client_01 | dorsal | 2.555 | 0.152 | Moderately different from global |
| client_02 | mid | 2.786 | 0.273 | **Most different from global** |
| client_03 | ventral | 2.520 | 0.252 | Significantly different from global |

**Jensen-Shannon Divergence (JSD)** measures how different each client's label distribution is from the pooled global distribution:
- **client_02 (mid)** has the highest JSD (0.273) — its cell type mix differs most from the global average
- **client_01 (dorsal)** has the lowest JSD (0.152) — closest to global distribution
- Higher JSD = more non-IID, which explains why local training on a single client may underperform

**Entropy** indicates within-client label diversity:
- client_02 (mid) has highest entropy (2.786) — most diverse label distribution
- client_03 (ventral) has lowest entropy (2.520) — more concentrated in certain labels

---

## Label Distribution Insights

From `label_proportion_heatmap.png` and `client_label_probabilities.csv`:

### Dominant labels per client:
- **client_01 (dorsal)**: Label 0 (22%), Labels 18/21 (9% each)
- **client_02 (mid)**: Label 1 (21%), Labels 17/23 (10% each)
- **client_03 (ventral)**: Label 12 (21%), Label 19 (18%)

### Labels that vary significantly across clients:
- **Label 0**: 22% in dorsal, 4% in mid, 0.1% in ventral — highly region-specific
- **Label 1**: 0.005% in dorsal, 21% in mid, 5% in ventral
- **Label 12**: 0.02% in dorsal, 3% in mid, 21% in ventral
- **Label 19**: 0.01% in dorsal, 0.6% in mid, 18% in ventral

This **anatomical heterogeneity** is expected in brain tissue — different regions contain different cell type compositions.

---

## Implications for Training Strategies

### Why Centralized performs best:
- Sees all 316K samples with the full global distribution
- Can learn all 24 cell types with representative samples

### Why Federated may slightly underperform Centralized:
- Each round sees only one client's non-IID distribution
- FedAvg aggregation helps but doesn't fully recover global distribution
- Still benefits from seeing all clients over multiple rounds

### Why Local performs worst:
- Each local model only sees ~105K samples from ONE anatomical region
- **Missing cell types**: Some labels have <0.1% representation in certain regions
- Cannot generalize to cell types it has never (or rarely) seen
- Example: A model trained only on dorsal (client_01) will struggle with Label 12 (0.02% in dorsal but 21% in ventral)

---

## Generated Outputs

### CSV Files
| File | Description |
|------|-------------|
| `client_summary.csv` | Per-client statistics (sizes, splits, class counts) |
| `client_noniid_metrics.csv` | Entropy and JSD metrics per client |
| `client_label_probabilities.csv` | Full label proportion matrix (clients × labels) |

### Plots
| Plot | What it shows |
|------|---------------|
| `split_overview.png` | Held-out (157K) vs clients (~105K each) |
| `client_sizes.png` | Total samples per client (balanced) |
| `train_val_per_client.png` | 80/20 train/val split per client |
| `label_proportion_heatmap.png` | Non-IID visualization: label proportions vary by client |
| `global_label_distribution.png` | Pooled label distribution (what centralized sees) |
| `client_jsd_to_global.png` | JSD metric per client (higher = more non-IID) |
| `client_imbalance_max_fraction.png` | Within-client class imbalance |
| `umap_by_client.png` | UMAP colored by anatomical region |
| `umap_by_label.png` | UMAP colored by cell type (Leiden cluster) |

---

## Summary

This dataset exhibits **moderate non-IID characteristics** due to anatomical siloing:
- **Balanced client sizes** (~105K each) means federated aggregation weights are equal
- **Significant label heterogeneity** (JSD 0.15–0.27) explains performance gaps between strategies
- **Missing classes in some clients** (23 vs 24) directly impacts local training
- The **held-out set** (Replicate 3) provides a fair, unbiased evaluation for all strategies

The analysis confirms that **federated learning is a reasonable middle ground** between centralized (best performance) and local (worst performance) when data cannot be pooled.
