"""
Federated Learning Runner (UPDATED for GLOBAL TEST SPLIT)

Key design:
- Clients provide TRAIN + VAL only
- A single GLOBAL test set is used for final evaluation
- No per-client test evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy

from src.data import (
    create_dataloader,
    load_gene_list,
    load_label_map,
)
from src.model import create_model
from src.training import (
    evaluate,
    save_model,
)
from src.training.fl_client import (
    create_client_fn,
    state_dict_to_ndarrays,
    ndarrays_to_state_dict,
)
from src.training.fl_server import (
    get_on_fit_config_fn,
    get_on_evaluate_config_fn,
    aggregate_fit_metrics,
    aggregate_evaluate_metrics,
)

# ---------------------------------------------------------------------
# Custom FedAvg Strategy (to extract final parameters)
# ---------------------------------------------------------------------

class FedAvgWithParameterSaving(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregated_parameters: Optional[Parameters] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            parameters, _ = aggregated
            self.aggregated_parameters = parameters
        return aggregated


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _list_clients(data_dir: str) -> List[str]:
    clients_dir = Path(data_dir) / "clients"
    if not clients_dir.exists():
        raise FileNotFoundError(f"Clients directory not found: {clients_dir}")
    clients = sorted(
        p.name for p in clients_dir.iterdir()
        if p.is_dir() and p.name.startswith("client_")
    )
    if not clients:
        raise ValueError("No client directories found.")
    return clients


def _validate_data_exists(data_dir: str, clients: List[str]) -> None:
    required_files = ["train.parquet", "val.parquet"]
    missing = []
    for client in clients:
        cdir = Path(data_dir) / "clients" / client
        for f in required_files:
            if not (cdir / f).exists():
                missing.append(str(cdir / f))
    if missing:
        raise FileNotFoundError(
            "Missing required client files:\n" + "\n".join(missing)
        )


def _save_training_curves(history: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rounds = history["round"].values

    for metric, title, fname in [
        ("train_loss", "Training Loss", "train_loss.png"),
        ("train_accuracy", "Training Accuracy", "train_accuracy.png"),
        ("train_f1_macro", "Training F1 Macro", "train_f1_macro.png"),
        ("val_loss", "Validation Loss", "val_loss.png"),
        ("val_accuracy", "Validation Accuracy", "val_accuracy.png"),
        ("val_f1_macro", "Validation F1 Macro", "val_f1_macro.png"),
    ]:
        if metric in history.columns:
            plt.figure()
            plt.plot(rounds, history[metric], marker="o")
            plt.xlabel("Round")
            plt.ylabel(metric)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / fname, dpi=150)
            plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Learning (GLOBAL test)")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/federated")
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--clients_per_round", type=int, default=2)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--fine_tune_mode", type=str, default="head_only",
                        choices=["head_only", "partial", "full"])
    parser.add_argument("--include_spatial", action="store_true", default=True)
    parser.add_argument("--no_spatial", action="store_true")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()
    include_spatial = args.include_spatial and not args.no_spatial

    import platform
    if args.num_workers is None:
        args.num_workers = 0 if platform.system() == "Windows" else 4

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = args.data_dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Federated Learning (FedAvg) — GLOBAL Test Protocol")
    print("=" * 60)

    # Discover clients
    clients = _list_clients(data_dir)
    _validate_data_exists(data_dir, clients)
    print(f"Clients: {clients}")

    # Load metadata
    genes = load_gene_list(data_dir)
    label_map = load_label_map(data_dir)
    num_genes = len(genes)
    num_labels = len(label_map)

    # Load GLOBAL test
    global_test_path = Path(data_dir) / "global" / "test.parquet"
    if not global_test_path.exists():
        raise FileNotFoundError(
            f"Global test set not found: {global_test_path}"
        )
    global_test_df = pd.read_parquet(global_test_path)

    global_test_loader = create_dataloader(
        global_test_df,
        genes,
        batch_size=args.batch_size,
        shuffle=False,
        include_spatial=include_spatial,
        num_workers=args.num_workers,
    )

    # Initial model
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    model = create_model(
        num_genes=num_genes,
        num_labels=num_labels,
        pretrained_path=args.pretrained_path,
        fine_tune_mode=args.fine_tune_mode,
        include_spatial=include_spatial,
    )
    model.to(device)

    initial_weights = state_dict_to_ndarrays(model.get_weights())
    initial_parameters = ndarrays_to_parameters(initial_weights)

    use_amp = args.use_amp and not args.no_amp and device.type == "cuda"

    client_fn = create_client_fn(
        client_ids=clients,
        data_dir=data_dir,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        fine_tune_mode=args.fine_tune_mode,
        include_spatial=include_spatial,
        pretrained_path=args.pretrained_path,
        num_workers=args.num_workers,
        verbose=True,
        use_amp=use_amp,
    )

    strategy = FedAvgWithParameterSaving(
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min(args.clients_per_round, len(clients)),
        min_evaluate_clients=min(args.clients_per_round, len(clients)),
        min_available_clients=len(clients),
        on_fit_config_fn=get_on_fit_config_fn(args.local_epochs),
        on_evaluate_config_fn=get_on_evaluate_config_fn(),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    print("\nStarting Flower simulation...\n")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0 if args.device == "cpu" else 0.5},
    )

    # Extract round metrics
    round_metrics = []
    for r in range(1, args.num_rounds + 1):
        row = {"round": r}
        if history.metrics_distributed_fit:
            for k, vals in history.metrics_distributed_fit.items():
                for rr, v in vals:
                    if rr == r:
                        row[f"train_{k}"] = v
        if history.metrics_distributed:
            for k, vals in history.metrics_distributed.items():
                for rr, v in vals:
                    if rr == r:
                        row[f"val_{k}"] = v
        if history.losses_distributed:
            for rr, v in history.losses_distributed:
                if rr == r:
                    row["val_loss"] = v
        round_metrics.append(row)

    metrics_df = pd.DataFrame(round_metrics)

    # Load final model
    final_weights = parameters_to_ndarrays(strategy.aggregated_parameters)
    final_state = ndarrays_to_state_dict(model, final_weights)
    model.set_weights(final_state)

    # GLOBAL test evaluation
    print("\nFinal Evaluation on GLOBAL Test Set")
    global_test_metrics = evaluate(
        model,
        global_test_loader,
        device,
        verbose=True,
        use_amp=use_amp,
    )

    eval_summary = {
        "global_test": {
            "loss": float(global_test_metrics["loss"]),
            "accuracy": float(global_test_metrics["accuracy"]),
            "f1_macro": float(global_test_metrics["f1_macro"]),
            "num_samples": len(global_test_df),
        },
        "clients": clients,
    }

    # Save artifacts
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    with open(out_dir / "history.json", "w") as f:
        json.dump({"rounds": round_metrics}, f, indent=2)

    save_model(model, str(out_dir / "model_final.pt"))

    config = vars(args)
    config.update({
        "num_clients": len(clients),
        "num_genes": num_genes,
        "num_labels": num_labels,
    })
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    _save_training_curves(metrics_df, out_dir)

    print("\n✔ Federated training complete")
    print(f"✔ Results saved to: {out_dir}")
    print(f"✔ Global Test Accuracy: {eval_summary['global_test']['accuracy']:.4f}")
    print(f"✔ Global Test F1 Macro: {eval_summary['global_test']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
