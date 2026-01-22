# Model Contract – NicheFormer Integration

## Purpose
Defines how NicheFormer (or any future model) integrates with centralized
and federated training pipelines.

Training logic must not depend on model internals.

---

## Required Model Interface

Every model wrapper must implement:

- forward(batch) -> logits
- compute_loss(logits, labels) -> scalar
- get_weights() -> state_dict
- set_weights(state_dict)

---

## Baseline Task (Milestone 3)
- Pseudo-label classification (predict Leiden cluster ID)

Loss:
- CrossEntropyLoss

---

## Fine-Tuning Modes
The following modes must be configurable:

- Head-only fine-tuning (backbone frozen)
- Partial backbone fine-tuning
- Full fine-tuning

---

## Milestone 4 Extension
The classifier head may be replaced with:
- neighborhood prediction
- interaction modeling
- contrastive or spatial objectives

The training loop must remain unchanged.
