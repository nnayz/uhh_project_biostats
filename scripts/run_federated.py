"""\
Federated Learning Runner (Milestone 3 - Task 2)

Runs a federated fine-tuning experiment using Flower simulation with FedAvg.

Uses the shared:
  - Data Contract (src/data)
  - Model Contract (src/model)
  - Training Contract (src/training)
  - Federation Contract (this file + fl_client.py + fl_server.py)

Outputs (Evaluation + Training + Federation Contracts):
  results/federated/
    history.json       # Round-level metrics
    metrics.csv        # Tabular metrics per round
    config.json        # Training configuration
    model_final.pt     # Final global model weights
    eval_summary.json  # Final evaluation results
    plots/*.png        # Training curves (optional)

Example:
  python scripts/run_federated.py --num_rounds 5 --clients_per_round 3 --local_epochs 1 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import flwr as fl
from flwr.common import ndarrays_to_parameters

from src.data import (
    create_dataloader,
    load_client_data,
    load_gene_list,
    load_label_map,
)
from src.model import create_model
from src.training import (
    TrainingHistory,
    evaluate,
    save_model,
)
from src.training.fl_client import (
    FlowerClient,
    create_client_fn,
    state_dict_to_ndarrays,
    ndarrays_to_state_dict,
)
from src.training.fl_server import (
    create_fedavg_strategy,
    get_on_fit_config_fn,
    get_on_evaluate_config_fn,
    RoundLogger,
    weighted_average,
)

from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy


class FedAvgWithParameterSaving(FedAvg):
    """
    FedAvg strategy that saves the aggregated parameters after each round.
    
    This allows us to extract the final model parameters after simulation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregated_parameters: Optional[Parameters] = None
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ):
        """Aggregate fit results and save the parameters."""
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            parameters, metrics = aggregated
            self.aggregated_parameters = parameters
        return aggregated


def _list_clients(data_dir: str) -> List[str]:
    """
    Discover client IDs by listing data_dir/clients/client_*.
    
    Args:
        data_dir: Path to processed data directory
        
    Returns:
        Sorted list of client folder names
    """
    clients_dir = Path(data_dir) / "clients"
    if not clients_dir.exists():
        raise FileNotFoundError(
            f"Clients directory not found: {clients_dir}. "
            "Make sure Milestone 2 outputs exist at the expected location."
        )
    
    clients = sorted([
        p.name for p in clients_dir.iterdir()
        if p.is_dir() and p.name.startswith("client_")
    ])
    
    if not clients:
        raise ValueError(
            f"No client directories found in {clients_dir}. "
            "Expected directories like 'client_01', 'client_02', etc."
        )
    
    return clients


def _validate_data_exists(data_dir: str, clients: List[str]) -> None:
    """
    Validate that required data files exist for all clients.
    
    Args:
        data_dir: Path to processed data directory
        clients: List of client folder names
    """
    required_files = ["train.parquet", "val.parquet", "test.parquet"]
    missing = []
    
    for client in clients:
        client_dir = Path(data_dir) / "clients" / client
        for filename in required_files:
            filepath = client_dir / filename
            if not filepath.exists():
                missing.append(str(filepath))
    
    if missing:
        raise FileNotFoundError(
            f"Missing required data files:\n" +
            "\n".join(missing[:10]) +
            ("\n..." if len(missing) > 10 else "") +
            "\n\nPlease download the processed dataset from the shared Drive link "
            "(see README.md) and extract it to data/processed/."
        )


def _save_training_curves(history: pd.DataFrame, out_dir: Path) -> None:
    """
    Save training and validation curves as plots.
    
    Args:
        history: DataFrame with round-level metrics
        out_dir: Output directory for plots
    """
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    rounds = history["round"].values
    
    # Loss curve
    if "train_loss" in history.columns or "val_loss" in history.columns:
        plt.figure(figsize=(8, 6))
        if "train_loss" in history.columns:
            plt.plot(rounds, history["train_loss"], label="train_loss", marker="o")
        if "val_loss" in history.columns:
            plt.plot(rounds, history["val_loss"], label="val_loss", marker="s")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.title("Federated Training - Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "loss_curve.png", dpi=150)
        plt.close()
    
    # Accuracy curve
    if "train_accuracy" in history.columns or "val_accuracy" in history.columns:
        plt.figure(figsize=(8, 6))
        if "train_accuracy" in history.columns:
            plt.plot(rounds, history["train_accuracy"], label="train_acc", marker="o")
        if "val_accuracy" in history.columns:
            plt.plot(rounds, history["val_accuracy"], label="val_acc", marker="s")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title("Federated Training - Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "accuracy_curve.png", dpi=150)
        plt.close()
    
    # F1 Macro curve
    if "train_f1_macro" in history.columns or "val_f1_macro" in history.columns:
        plt.figure(figsize=(8, 6))
        if "train_f1_macro" in history.columns:
            plt.plot(rounds, history["train_f1_macro"], label="train_f1", marker="o")
        if "val_f1_macro" in history.columns:
            plt.plot(rounds, history["val_f1_macro"], label="val_f1", marker="s")
        plt.xlabel("Round")
        plt.ylabel("F1 Macro")
        plt.title("Federated Training - F1 Macro Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "f1_macro_curve.png", dpi=150)
        plt.close()


def _run_final_evaluation(
    model: torch.nn.Module,
    clients: List[str],
    genes: List[str],
    data_dir: str,
    batch_size: int,
    include_spatial: bool,
    device: torch.device,
    num_workers: int = 0,
    use_amp: bool = False,
) -> Dict:
    """
    Run final evaluation of the global model on all clients' test sets.
    
    Args:
        model: Trained global model
        clients: List of client IDs
        genes: Gene list for data loading
        data_dir: Path to processed data
        batch_size: Batch size for evaluation
        include_spatial: Whether to include spatial features
        device: Device to run on
        num_workers: Number of data loading workers
        
    Returns:
        Dictionary with per-client and aggregated test metrics
    """
    per_client_test = {}
    all_losses = []
    all_accuracies = []
    all_f1s = []
    total_samples = 0
    
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Sets")
    print("=" * 60)
    
    for client_id in clients:
        try:
            test_df = load_client_data(
                client_id, "test",
                data_dir=data_dir,
                validate=True
            )
            test_loader = create_dataloader(
                test_df, genes,
                batch_size=batch_size,
                shuffle=False,
                include_spatial=include_spatial,
                num_workers=num_workers,
            )
            
            print(f"\n{client_id} test ({len(test_df)} samples):")
            metrics = evaluate(model, test_loader, device, verbose=True, use_amp=use_amp)
            
            per_client_test[client_id] = {
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
                "num_samples": len(test_df),
            }
            
            # Accumulate for weighted average
            n = len(test_df)
            all_losses.append(metrics["loss"] * n)
            all_accuracies.append(metrics["accuracy"] * n)
            all_f1s.append(metrics["f1_macro"] * n)
            total_samples += n
            
        except FileNotFoundError as e:
            print(f"Warning: Could not load test data for {client_id}: {e}")
            continue
    
    # Compute weighted average across clients
    if total_samples > 0:
        global_test = {
            "loss": sum(all_losses) / total_samples,
            "accuracy": sum(all_accuracies) / total_samples,
            "f1_macro": sum(all_f1s) / total_samples,
            "total_samples": total_samples,
        }
    else:
        global_test = {"loss": None, "accuracy": None, "f1_macro": None, "total_samples": 0}
    
    print("\n" + "-" * 40)
    print(f"Global Test (weighted avg): Loss={global_test['loss']:.4f}, "
          f"Acc={global_test['accuracy']:.4f}, F1={global_test['f1_macro']:.4f}")
    
    return {
        "global_test": global_test,
        "per_client_test": per_client_test,
        "clients": clients,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated learning with Flower (FedAvg)"
    )
    
    # Data and output
    parser.add_argument(
        "--data_dir", type=str, default="data/processed",
        help="Processed data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/federated",
        help="Output directory for artifacts"
    )
    
    # Federated settings
    parser.add_argument(
        "--num_rounds", type=int, default=5,
        help="Number of federated rounds"
    )
    parser.add_argument(
        "--clients_per_round", type=int, default=2,
        help="Number of clients to sample per round"
    )
    parser.add_argument(
        "--local_epochs", type=int, default=1,
        help="Number of local training epochs per round"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=1024,
        help="Batch size for training (increased for GPU utilization)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5,
        help="Weight decay"
    )
    
    # Model settings
    parser.add_argument(
        "--fine_tune_mode", type=str, default="head_only",
        choices=["head_only", "partial", "full"],
        help="Fine-tuning mode"
    )
    parser.add_argument(
        "--include_spatial", action="store_true", default=True,
        help="Include spatial coordinates"
    )
    parser.add_argument(
        "--no_spatial", action="store_true",
        help="Disable spatial coordinates"
    )
    parser.add_argument(
        "--pretrained_path", type=str, default=None,
        help="Path to pretrained Nicheformer checkpoint"
    )
    
    # Other settings
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Training device"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of data loading workers (0=main thread, 4-8 recommended for GPU). Auto-set to 0 on Windows with Ray."
    )
    parser.add_argument(
        "--use_amp", action="store_true", default=True,
        help="Use Automatic Mixed Precision (AMP) for faster training"
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable AMP"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed progress"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    # Process arguments
    include_spatial = args.include_spatial and not args.no_spatial
    verbose = args.verbose and not args.quiet
    
    # Windows + Ray + multiprocessing DataLoader = incompatible
    # Auto-set num_workers=0 on Windows when using Ray/Flower
    import platform
    if args.num_workers is None:
        if platform.system() == 'Windows':
            args.num_workers = 0
            if verbose:
                print("[INFO] Windows detected: Setting num_workers=0 (multiprocessing not compatible with Ray on Windows)")
        else:
            args.num_workers = 4  # Default for Linux/Mac
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup paths
    data_dir = args.data_dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover clients
    print("=" * 60)
    print("Federated Learning with Flower (FedAvg)")
    print("=" * 60)
    
    clients = _list_clients(data_dir)
    print(f"\nDiscovered {len(clients)} clients: {clients}")
    
    # Validate data exists
    _validate_data_exists(data_dir, clients)
    
    # Load gene list and label map
    genes = load_gene_list(data_dir)
    label_map = load_label_map(data_dir)
    num_genes = len(genes)
    num_labels = len(label_map)
    
    print(f"Genes: {num_genes}, Labels: {num_labels}")
    print(f"Rounds: {args.num_rounds}, Clients/round: {args.clients_per_round}, Local epochs: {args.local_epochs}")
    print(f"Device: {args.device}, Fine-tune mode: {args.fine_tune_mode}")
    
    # Create initial model to get initial parameters
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    
    initial_model = create_model(
        num_genes=num_genes,
        num_labels=num_labels,
        pretrained_path=args.pretrained_path,
        fine_tune_mode=args.fine_tune_mode,
        include_spatial=include_spatial,
    )
    initial_model.to(device)
    
    if hasattr(initial_model, "count_parameters"):
        total_params, trainable_params = initial_model.count_parameters()
        print(f"Model params: total={total_params:,}, trainable={trainable_params:,}")
    
    # Get initial parameters
    initial_weights = state_dict_to_ndarrays(initial_model.get_weights())
    initial_parameters = ndarrays_to_parameters(initial_weights)
    
    # Setup AMP for GPU training
    use_amp = args.use_amp and not args.no_amp and device.type == 'cuda'
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) for faster GPU training")
    
    # Create client factory
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
        verbose=verbose,
        use_amp=use_amp,
    )
    
    # Create FedAvg strategy with parameter saving
    from src.training.fl_server import aggregate_fit_metrics, aggregate_evaluate_metrics
    
    strategy = FedAvgWithParameterSaving(
        initial_parameters=initial_parameters,
        fraction_fit=1.0,  # We'll control via min_fit_clients
        fraction_evaluate=1.0,
        min_fit_clients=min(args.clients_per_round, len(clients)),
        min_evaluate_clients=min(args.clients_per_round, len(clients)),
        min_available_clients=len(clients),
        on_fit_config_fn=get_on_fit_config_fn(args.local_epochs),
        on_evaluate_config_fn=get_on_evaluate_config_fn(),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )
    
    # Run Flower simulation
    print("\n" + "=" * 60)
    print("Starting Federated Training Simulation")
    print("=" * 60 + "\n")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0 if args.device == "cpu" else 0.5},
    )
    
    # Extract metrics from history
    print("\n" + "=" * 60)
    print("Extracting Round Metrics")
    print("=" * 60)
    
    round_metrics = []
    
    # History object has:
    # - losses_distributed: List of (round, loss) tuples
    # - losses_centralized: List of (round, loss) tuples  
    # - metrics_distributed: Dict[str, List[(round, value)]]
    # - metrics_distributed_fit: Dict[str, List[(round, value)]]
    # - metrics_centralized: Dict[str, List[(round, value)]]
    
    for r in range(1, args.num_rounds + 1):
        round_data = {"round": r}
        
        # Get distributed fit metrics (training)
        if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
            fit_metrics = history.metrics_distributed_fit
            for metric_name, values in fit_metrics.items():
                for round_num, value in values:
                    if round_num == r:
                        if metric_name == "loss":
                            round_data["train_loss"] = value
                        elif metric_name == "accuracy":
                            round_data["train_accuracy"] = value
                        elif metric_name == "f1_macro":
                            round_data["train_f1_macro"] = value
        
        # Get distributed evaluate metrics (validation)
        if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
            eval_metrics = history.metrics_distributed
            for metric_name, values in eval_metrics.items():
                for round_num, value in values:
                    if round_num == r:
                        if metric_name == "accuracy":
                            round_data["val_accuracy"] = value
                        elif metric_name == "f1_macro":
                            round_data["val_f1_macro"] = value
        
        # Get distributed losses (validation loss)
        if hasattr(history, 'losses_distributed') and history.losses_distributed:
            for round_num, loss in history.losses_distributed:
                if round_num == r:
                    round_data["val_loss"] = loss
        
        round_metrics.append(round_data)
        
        # Print round summary
        parts = [f"Round {r}:"]
        if "train_loss" in round_data:
            parts.append(f"train_loss={round_data['train_loss']:.4f}")
        if "train_accuracy" in round_data:
            parts.append(f"train_acc={round_data['train_accuracy']:.4f}")
        if "val_loss" in round_data:
            parts.append(f"val_loss={round_data['val_loss']:.4f}")
        if "val_accuracy" in round_data:
            parts.append(f"val_acc={round_data['val_accuracy']:.4f}")
        print(" | ".join(parts))
    
    metrics_df = pd.DataFrame(round_metrics)
    
    # Get final model parameters from strategy
    print("\n" + "=" * 60)
    print("Extracting Final Model")
    print("=" * 60)
    
    # Get the aggregated parameters from our custom strategy
    if strategy.aggregated_parameters is not None:
        final_weights = parameters_to_ndarrays(strategy.aggregated_parameters)
        print("Successfully extracted final aggregated parameters from strategy.")
    else:
        # Fallback: use initial weights
        print("Warning: Could not extract final parameters from strategy.")
        print("Using initial parameters...")
        final_weights = initial_weights
    
    # Load final weights into model
    final_state_dict = ndarrays_to_state_dict(initial_model, final_weights)
    initial_model.set_weights(final_state_dict)
    
    # Run final evaluation
    eval_summary = _run_final_evaluation(
        model=initial_model,
        clients=clients,
        genes=genes,
        data_dir=data_dir,
        batch_size=args.batch_size,
        include_spatial=include_spatial,
        device=device,
        num_workers=args.num_workers,
        use_amp=use_amp,
    )
    
    # Save artifacts
    print("\n" + "=" * 60)
    print("Saving Artifacts")
    print("=" * 60)
    
    # Save config
    config = {
        "data_dir": data_dir,
        "output_dir": str(out_dir),
        "num_rounds": args.num_rounds,
        "clients_per_round": args.clients_per_round,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "fine_tune_mode": args.fine_tune_mode,
        "include_spatial": include_spatial,
        "pretrained_path": args.pretrained_path,
        "device": args.device,
        "seed": args.seed,
        "num_genes": num_genes,
        "num_labels": num_labels,
        "clients": clients,
        "num_clients": len(clients),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json")
    
    # Save history (round metrics)
    history_dict = {
        "rounds": round_metrics,
        "losses_distributed": list(history.losses_distributed) if hasattr(history, 'losses_distributed') else [],
        "losses_centralized": list(history.losses_centralized) if hasattr(history, 'losses_centralized') else [],
    }
    with open(out_dir / "history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"  Saved history.json")
    
    # Save metrics CSV
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    print(f"  Saved metrics.csv")
    
    # Save final model
    model_path = out_dir / "model_final.pt"
    save_model(initial_model, str(model_path))
    print(f"  Saved model_final.pt")
    
    # Save evaluation summary
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"  Saved eval_summary.json")
    
    # Save plots
    if not metrics_df.empty:
        _save_training_curves(metrics_df, out_dir)
        print(f"  Saved plots/")
    
    print("\n" + "=" * 60)
    print(f"Federated training complete!")
    print(f"Artifacts saved to: {out_dir}")
    print("=" * 60)
    
    # Print final summary
    if eval_summary["global_test"]["accuracy"] is not None:
        print(f"\nFinal Global Test Results:")
        print(f"  Accuracy: {eval_summary['global_test']['accuracy']:.4f}")
        print(f"  F1 Macro: {eval_summary['global_test']['f1_macro']:.4f}")
        print(f"  Loss: {eval_summary['global_test']['loss']:.4f}")


if __name__ == "__main__":
    main()
