# Federation Contract – Flower + FedAvg

## Framework
Flower (Python-based federated learning framework)

---

## Algorithm
FedAvg (baseline)

Global weights are computed as a weighted average of client updates,
where weights are proportional to the number of local training samples.

---

## Server Responsibilities
- Initialize global model weights
- Dispatch weights to clients
- Aggregate updates using FedAvg
- Log round-level metrics

---

## Client Responsibilities
- Load local train/val data
- Train locally for N epochs
- Return updated weights and metrics

---

## Configuration Parameters
- num_clients
- clients_per_round
- num_rounds
- local_epochs
- batch_size
- learning_rate
- random_seed

---

## Milestone 4 Extension
- Replace FedAvg with FedProx, Scaffold, or personalized aggregation
- Add client-specific heads without modifying server logic
