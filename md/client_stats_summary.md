# Client Statistics & Diagnostics

## What this analysis covers

- Per-client dataset sizes (train/val only; test is global)

- Label distribution imbalance within each client

- Non-IID severity vs global distribution

- Variance of label composition across clients


- Global test set size: 94918


## Figures

- `md/figures/client_sizes.png`

- `md/figures/client_imbalance_max_fraction.png`

- `md/figures/client_jsd_to_global.png`

- `md/figures/client_label_distribution_heatmap.png`

- `md/figures/label_proportion_variance_violin.png`


## Summary table (per client)

| client    | group_value   |   n_total |   n_train |   n_val |   n_test |   n_classes_present |   max_label_fraction |   min_label_count |
|:----------|:--------------|----------:|----------:|--------:|---------:|--------------------:|---------------------:|------------------:|
| client_01 | replicate 1   |    129392 |    116443 |   12949 |        0 |                  23 |            0.0895341 |               483 |
| client_02 | replicate 2   |    123639 |    111266 |   12373 |        0 |                  23 |            0.0888878 |               442 |
| client_03 | replicate 3   |    126596 |    113925 |   12671 |        0 |                  23 |            0.0878543 |               427 |



## Non-IID metrics (per client)

| client    | group_value   |   n_total |   n_classes_present |   max_label_fraction |   entropy |   js_divergence_to_global |
|:----------|:--------------|----------:|--------------------:|---------------------:|----------:|--------------------------:|
| client_01 | replicate 1   |    129392 |                  23 |            0.0895341 |   2.93694 |               0.000204407 |
| client_02 | replicate 2   |    123639 |                  23 |            0.0888878 |   2.93973 |               2.60314e-05 |
| client_03 | replicate 3   |    126596 |                  23 |            0.0878543 |   2.94192 |               0.00021147  |



## Global test: cell-type distribution (UMAP)

UMAP of the global test set (no batch correction), colored by cell type/label. The right panel summarizes cell-type proportions for the plotted test set.


- `diagnostics/figures/test_cell_distribution_umap.png`


How to generate:

```bash

python scripts/generate_test_cell_distribution_figure.py \

  --data_path data/processed \

  --out_path diagnostics/figures/test_cell_distribution_umap.png \

  --seed 42 \

  --max_points 200000

```
