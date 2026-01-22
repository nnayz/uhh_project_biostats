# Client Statistics & Diagnostics (Milestone 2)

## Overview
This document summarizes the statistical properties of the federated clients created in Milestone 2 and quantifies the degree of data heterogeneity (non-IID behavior) across clients.

The goal of this analysis is to verify that:
- federated clients are well-formed,
- data is reasonably balanced,
- non-IID characteristics are present and measurable,
- the dataset is suitable for downstream federated learning experiments.

---

## Federated Setup Summary

- Number of clients: **3**
- Client definition: biological replicates (`library_key`)
  - `client_01` → replicate 1  
  - `client_02` → replicate 2  
  - `client_03` → replicate 3
- Labels: **22 pseudo-labels** derived via Leiden clustering
- Data type: spatial transcriptomics (Xenium mouse brain)

Each client contains mutually exclusive samples and is split into train/validation/test subsets.

---

## Client Size Statistics

Client dataset sizes are comparable across all sites:

- Each client contains approximately **150k–160k samples**
- No client dominates the global dataset
- This prevents bias in federated aggregation due to extreme size imbalance

This balanced setup ensures that observed federated learning behavior is driven primarily by data heterogeneity rather than sample count disparity.

(See: `md/figures/client_sizes.png`)

---

## Label Distribution & Imbalance

To assess within-client label skew, we compute the **maximum label fraction** per client, defined as:

> max(label count) / total samples in client

Results:
- Maximum label fraction is approximately **10–11%** across all clients
- Values are similar across clients

Interpretation:
- No single label dominates any client
- Label distributions are relatively well spread
- Within-client class imbalance is **mild**

This level of imbalance is realistic for biological data and does not pose issues for supervised or federated training.

(See: `md/figures/client_imbalance_max_fraction.png`)

---

## Non-IID Severity Analysis

To quantify inter-client heterogeneity, we measure **Jensen–Shannon divergence (JSD)** between each client’s label distribution and the global label distribution.

### Metric
- Jensen–Shannon divergence (in nats)
- Symmetric, bounded, and interpretable
- Lower values indicate more similarity to the global distribution

### Results
- JSD values are on the order of **10⁻⁴**
- `client_02` is closest to the global distribution
- `client_01` and `client_03` show slightly higher divergence

Interpretation:
- The federated setup exhibits **weak but measurable non-IID behavior**
- Differences arise naturally from replicate-specific variation
- No extreme or artificially induced heterogeneity is present

This represents a **realistic and stable federated learning scenario**, suitable for validating training pipelines before exploring more severe non-IID settings.

(See: `md/figures/client_jsd_to_global.png`)

---

## Summary Table (Per Client)

The following statistics are computed for each client:
- total number of samples
- train/validation/test split sizes
- number of labels present
- maximum label fraction
- entropy of label distribution
- Jensen–Shannon divergence to global distribution

Detailed tables are available in:
- `md/figures/client_summary.csv`
- `md/figures/client_noniid_metrics.csv`
- `md/figures/client_label_probabilities.csv`

---

## Key Takeaways

- Federated clients are **balanced in size**
- Label distributions show **mild within-client imbalance**
- Inter-client heterogeneity is **present but not extreme**
- The dataset is **well-suited for federated learning experiments**
- This setup provides a strong baseline for Milestone 3

---

## Notes for Downstream Experiments

- Results in Milestone 3 should be interpreted in the context of **mild non-IID data**
- Strong performance differences across clients are not expected at this stage
- More aggressive non-IID scenarios (e.g., spatial splits or label filtering) can be explored in later milestones if needed

---

## Conclusion

The client diagnostics confirm that the data preparation and federated partitioning steps were successful. The resulting federated dataset is clean, reproducible, and ready for federated model training and evaluation in Milestone 3.
