# Data Contract – Milestone 3

## Purpose
This document defines the single source of truth for how data is accessed,
validated, and consumed across Milestones 3 and 4.

All training, federated, and evaluation code must use this interface.

---

## Input Sources (From Milestone 2)

Per client:
- data/processed/clients/client_XX/train.parquet
- data/processed/clients/client_XX/val.parquet
- data/processed/clients/client_XX/test.parquet

Global:
- genes.txt
- label_map.json

---

## Required Parquet Schema

Each row corresponds to one cell.

| Column | Type | Description |
|------|------|-------------|
| cell_id | string | Unique identifier for the cell |
| client_id | string | Federated client identifier |
| x | float | Spatial x-coordinate (µm) |
| y | float | Spatial y-coordinate (µm) |
| label | int | Pseudo-label (Leiden cluster ID) |
| gene_* | float | Normalized, log-transformed gene expression |

Notes:
- There must be exactly G gene columns.
- Gene order must match genes.txt exactly.

---

## Loader API (Required)

The following functions must be implemented and used everywhere:

- load_client_data(client_id, split)
- load_all_clients(split)
- load_gene_list()
- load_label_map()

Each loader must return tensors ready for PyTorch training.

---

## Validation Rules
- Assert gene column order matches genes.txt
- Assert labels are integers in [0, num_labels)
- Assert no NaN or infinite values
- Assert train/val/test splits are non-empty

---

## Milestone 4 Extension
Spatial neighborhoods, graphs, or adjacency matrices must be derived
from this schema and must not replace it.
