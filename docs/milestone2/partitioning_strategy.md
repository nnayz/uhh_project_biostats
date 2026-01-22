# Federated Partitioning Strategy (Milestone 2)

## Objective
The goal of this step is to simulate **multiple independent data sites (clients)** for federated learning experiments.

## Client Definition
Clients are defined using the column:
- `sample_id` (derived from `library_key`)

This corresponds to three experimental replicates:
- replicate 1
- replicate 2
- replicate 3

Each replicate is treated as an independent client, simulating data from separate institutions.

## Number of Clients
- Total clients: **3**
  - `client_01`
  - `client_02`
  - `client_03`

## Within-Client Data Splits
For each client, data is split as follows:
- **Training:** 80%
- **Validation:** 10%
- **Test:** 10%

Splits are **stratified by label** (Leiden cluster) to preserve the local label distribution within each client.

## Leakage Prevention
- No samples are shared across clients
- Train/validation/test splits are mutually exclusive
- All clients share the same gene schema (`genes.txt`)

## Output Structure
Each client directory follows the same structure:

data/processed/clients/client_XX/
├── train.parquet
├── val.parquet
├── test.parquet
└── client_meta.json


Additionally:
- `data/processed/global_metadata.json` summarizes all clients

## Metadata Contents
Each `client_meta.json` includes:
- Client name
- Split axis
- Sample counts (total, train, val, test)
- Label distribution for the full client dataset

## Notes
This partitioning strategy produces realistic **non-IID data distributions** across clients and is suitable for evaluating federated learning behavior in Milestone 3.
