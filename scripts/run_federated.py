"""
Federated Training Runner (GLOBAL TEST SPLIT – FINAL)

Key properties:
- Clients only contain train + val
- A single GLOBAL test set is used for final evaluation
- Compatible with Flower FedAvg
- Evaluates the TRUE federated global model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import flwr as fl
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    create_dataloader,
    load_client_data,
    load_gene_list,
    load_label_map,
)
from src.model import create_model
from src.training import train_one_epoch, evaluate
from src.config import TrainingConfig


# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#

def list_clients(data_dir: str) -> List[str]:
    clients_dir = Path(data_dir) / "clients"
    return sorted(
        p.name for p in clients_dir.iterdir()
        if p.is_dir() and p.name.startswith("client_")
    )


def validate_data(data_dir: str, clients: List[str]) -> None:
    for c in clients:
        cdir = Path(data_dir) / "clients" / c
        for split in ["train", "val"]:
            if not (cdir / f"{split}.parquet").exists():
                raise FileNotFoundError(f"Missing {split} for {c}")

    if not (Path(data_dir) / "global" / "test.parquet").exists():
        raise FileNotFoundError("Global test set missing")


# -----------------------------------------------------------------------------#
# Flower Client
# -----------------------------------------------------------------------------#

class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        lr,
        local_epochs,
        use_amp,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.use_amp = use_amp

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data.copy_(torch.from_numpy(new_p).to(self.device))

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        for _ in range(self.local_epochs):
            train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.device,
                use_amp=self.use_amp,
                scaler=scaler,
                verbose=False,
            )

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = evaluate(
            self.model,
            self.val_loader,
            self.device,
            use_amp=self.use_amp,
            verbose=False,
        )

        return (
            float(metrics["loss"]),
            len(self.val_loader.dataset),
            {
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
            },
        )


# -----------------------------------------------------------------------------#
# Global Test Evaluation (SERVER SIDE)
# -----------------------------------------------------------------------------#

def get_global_evaluate_fn(
    data_dir,
    genes,
    num_genes,
    num_labels,
    batch_size,
    include_spatial,
    device,
    num_workers,
    use_amp,
    args,
):
    def evaluate_fn(server_round, parameters, config):
        model = create_model(
            num_genes=num_genes,
            num_labels=num_labels,
            pretrained_path=args.pretrained_path,
            fine_tune_mode=args.fine_tune_mode,
            include_spatial=include_spatial,
        ).to(device)

        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(model.state_dict().keys(), parameters)
        }
        model.load_state_dict(state_dict, strict=True)

        test_df = pd.read_parquet(
            Path(data_dir) / "global" / "test.parquet"
        )

        loader = create_dataloader(
            test_df,
            genes,
            batch_size=batch_size,
            shuffle=False,
            include_spatial=include_spatial,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        metrics = evaluate(
            model,
            loader,
            device,
            use_amp=use_amp,
            verbose=True,
        )

        return metrics["loss"], metrics

    return evaluate_fn


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="results/federated")
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--clients_per_round", type=int, default=3)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fine_tune_mode", default="head_only")
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--include_spatial", action="store_true", default=True)
    parser.add_argument("--no_spatial", action="store_true")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    include_spatial = args.include_spatial and not args.no_spatial
    device = torch.device(args.device)
    use_amp = args.use_amp and device.type == "cuda"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    clients = list_clients(args.data_dir)
    validate_data(args.data_dir, clients)

    genes = load_gene_list(args.data_dir)
    num_genes = len(genes)
    num_labels = len(load_label_map(args.data_dir))

    def client_fn(cid: str):
        cname = clients[int(cid)]

        train_df = load_client_data(cname, "train", args.data_dir)
        val_df = load_client_data(cname, "val", args.data_dir)

        train_loader = create_dataloader(
            train_df, genes, args.batch_size, True,
            include_spatial, args.num_workers, device.type == "cuda"
        )
        val_loader = create_dataloader(
            val_df, genes, args.batch_size, False,
            include_spatial, args.num_workers, device.type == "cuda"
        )

        model = create_model(
            num_genes, num_labels,
            args.pretrained_path,
            args.fine_tune_mode,
            include_spatial,
        ).to(device)

        return FlowerClient(
            model, train_loader, val_loader,
            device, args.lr, args.local_epochs, use_amp
        )

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.clients_per_round / len(clients),
        min_fit_clients=args.clients_per_round,
        min_available_clients=len(clients),
        evaluate_fn=get_global_evaluate_fn(
            args.data_dir, genes, num_genes, num_labels,
            args.batch_size, include_spatial,
            device, args.num_workers, use_amp, args
        ),
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(clients),
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )

    print("✔ Federated training complete")


if __name__ == "__main__":
    main()
