# Training Contract

## Purpose
Defines shared training and evaluation logic used in both centralized
and federated settings.

---

## Required Functions

- train_one_epoch(model, dataloader, optimizer)
- evaluate(model, dataloader)

These functions must be reused everywhere.

---

## Required Metrics
- Training loss
- Accuracy
- Macro-F1 score

---

## Output Artifacts
Each run must save:
- history.json
- metrics.csv
- model_final.pt
- config.yaml

---

## Milestone 4 Extension
Multi-task objectives or auxiliary spatial losses must be added
inside compute_loss only.
