# Client Statistics & Diagnostics

## What this analysis covers

- Per-client dataset sizes (train/val only; test is global)

- Label distribution imbalance within each client

- Non-IID severity vs global distribution

- Variance of label composition across clients


- Global test set size: 94920


## Figures

- `md/figures/client_sizes.png`

- `md/figures/client_imbalance_max_fraction.png`

- `md/figures/client_jsd_to_global.png`

- `md/figures/client_label_distribution_heatmap.png`

- `md/figures/label_proportion_variance_violin.png`


## Summary table (per client)

| client    | group_value   |   n_total |   n_train |   n_val |   n_test |   n_classes_present |   max_label_fraction |   min_label_count |
|:----------|:--------------|----------:|----------:|--------:|---------:|--------------------:|---------------------:|------------------:|
| client_01 | replicate 1   |    129583 |    116614 |   12969 |        0 |                  23 |            0.0886768 |              1632 |
| client_02 | replicate 2   |    123703 |    111324 |   12379 |        0 |                  23 |            0.0885508 |              1550 |
| client_03 | replicate 3   |    126339 |    113693 |   12646 |        0 |                  23 |            0.0857613 |              1542 |



## Non-IID metrics (per client)

| client    | group_value   |   n_total |   n_classes_present |   max_label_fraction |   entropy |   js_divergence_to_global |
|:----------|:--------------|----------:|--------------------:|---------------------:|----------:|--------------------------:|
| client_01 | replicate 1   |    129583 |                  23 |            0.0886768 |   3.00733 |               0.000210246 |
| client_02 | replicate 2   |    123703 |                  23 |            0.0885508 |   3.01039 |               2.42716e-05 |
| client_03 | replicate 3   |    126339 |                  23 |            0.0857613 |   3.01576 |               0.000231982 |