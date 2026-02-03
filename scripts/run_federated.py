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
    load_global_test,
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
    EvaluateRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy


def _client_id_from_proxy(proxy: ClientProxy, client_ids: List[str]) -> str:
    """Map Flower cid (0,1,2) to actual client_id (client_01, ...)."""
    cid = getattr(proxy, "cid", None)
    if cid is None:
        cid = getattr(proxy, "id", cid)
    if client_ids and cid is not None:
        try:
            idx = int(cid)
            if 0 <= idx < len(client_ids):
                return client_ids[idx]
        except (ValueError, TypeError):
            pass
    return f"client_{cid}"


class FedAvgWithParameterSaving(FedAvg):
    """
    FedAvg strategy that saves aggregated parameters and per-client metrics each round.
    Per-client metrics support analysis of non-IID effects and which client drags the average.
    """

    def __init__(self, *args, client_ids: Optional[List[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregated_parameters: Optional[Parameters] = None
        self.client_ids: List[str] = client_ids or []
        self.per_client_fit_metrics: List[Dict] = []
        self.per_client_eval_metrics: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ):
        """Record per-client fit metrics, then aggregate."""
        for proxy, fit_res in results:
            client_id = _client_id_from_proxy(proxy, self.client_ids)
            metrics = {}
            if hasattr(fit_res, "metrics") and fit_res.metrics:
                metrics = dict(fit_res.metrics)
            elif hasattr(fit_res, "status") and getattr(fit_res.status, "metrics", None):
                metrics = dict(fit_res.status.metrics)
            self.per_client_fit_metrics.append({
                "round": server_round,
                "client_id": client_id,
                "num_examples": getattr(fit_res, "num_examples", 0),
                **metrics,
            })
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            parameters, _ = aggregated
            self.aggregated_parameters = parameters
        return aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, "EvaluateRes"]],
        failures: List,
    ):
        """Record per-client evaluation metrics, then aggregate."""
        for proxy, eval_res in results:
            client_id = _client_id_from_proxy(proxy, self.client_ids)
            metrics = {}
            if hasattr(eval_res, "metrics") and eval_res.metrics:
                metrics = dict(eval_res.metrics)
            elif hasattr(eval_res, "status") and getattr(eval_res.status, "metrics", None):
                metrics = dict(eval_res.status.metrics)
            self.per_client_eval_metrics.append({
                "round": server_round,
                "client_id": client_id,
                "loss": getattr(eval_res, "loss", None),
                "num_examples": getattr(eval_res, "num_examples", 0),
                **metrics,
            })
        return super().aggregate_evaluate(server_round, results, failures)


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
    Per-client test sets are not used; evaluation is on held_out_batch.parquet.
    """
    required_files = ["train.parquet", "val.parquet"]
    missing = []

    for client in clients:
        client_dir = Path(data_dir) / "clients" / client
        for filename in required_files:
            filepath = client_dir / filename
            if not filepath.exists():
                missing.append(str(filepath))

    if missing:
        raise FileNotFoundError(
            f"Missing required data files:\n"
            + "\n".join(missing[:10])
            + ("\n..." if len(missing) > 10 else "")
            + "\n\nRun data preparation first: partition_anatomical_siloing.py (see data_preparation.md)."
        )


def _load_client_labels(data_dir: Path, clients: List[str]) -> Dict[str, str]:
    """
    Load combined client labels (e.g. client01_dorsal, client02_mid, client03_ventral)
    from client_meta.json so graphs show both client id and brain region.
    """
    out: Dict[str, str] = {}
    for cid in clients:
        meta_path = data_dir / "clients" / cid / "client_meta.json"
        region = cid
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                region = meta.get("group_value") or meta.get("batch_id") or meta.get("anatomical_region") or cid
                region = str(region)
            except Exception:
                pass
        # client_01 -> client01_dorsal (no underscore between client and number)
        short_id = cid.replace("client_", "client") if cid.startswith("client_") else cid
        out[cid] = f"{short_id}_{region}"
    return out


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


def _save_per_client_metrics_and_plots(
    per_client_fit: List[Dict],
    per_client_eval: List[Dict],
    out_dir: Path,
    client_id_to_label: Optional[Dict[str, str]] = None,
) -> None:
    """
    Save per-client fit/eval metrics (JSON, CSV) and per-client curves for analysis
    and transparency (non-IID effects, which client drags the average).
    client_id_to_label: optional map client_01 -> "dorsal", etc., for readable graph legends.
    """
    if not per_client_fit and not per_client_eval:
        return
    client_id_to_label = client_id_to_label or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save raw JSON
    with open(out_dir / "per_client_fit_metrics.json", "w") as f:
        json.dump(per_client_fit, f, indent=2)
    with open(out_dir / "per_client_eval_metrics.json", "w") as f:
        json.dump(per_client_eval, f, indent=2)

    # Build merged per-client table: round, client_id, client_label (if available), train_*, val_*
    fit_df = pd.DataFrame(per_client_fit) if per_client_fit else pd.DataFrame()
    eval_df = pd.DataFrame(per_client_eval) if per_client_eval else pd.DataFrame()
    if fit_df.empty and eval_df.empty:
        return
    # Standardize column names: fit has loss/accuracy/f1_macro -> train_*; eval -> val_*
    if not fit_df.empty:
        fit_df = fit_df.rename(columns={
            "loss": "train_loss", "accuracy": "train_accuracy", "f1_macro": "train_f1_macro"
        })
    if not eval_df.empty:
        eval_df = eval_df.rename(columns={
            "loss": "val_loss", "accuracy": "val_accuracy", "f1_macro": "val_f1_macro"
        })
    merge_df = None
    if not fit_df.empty and not eval_df.empty:
        eval_cols = ["round", "client_id"] + [c for c in ["val_loss", "val_accuracy", "val_f1_macro"] if c in eval_df.columns]
        merge_df = fit_df.merge(eval_df[eval_cols], on=["round", "client_id"], how="outer")
    elif not fit_df.empty:
        merge_df = fit_df.copy()
    else:
        merge_df = eval_df.copy()
    if merge_df is not None and not merge_df.empty:
        merge_df = merge_df.sort_values(["round", "client_id"]).reset_index(drop=True)
        # Add human-readable label (e.g. dorsal, mid, ventral) for graphs and CSV
        merge_df["client_label"] = merge_df["client_id"].map(
            lambda cid: client_id_to_label.get(cid, cid)
        )
        # Reorder so client_label is next to client_id
        cols = ["round", "client_id", "client_label"] + [c for c in merge_df.columns if c not in ("round", "client_id", "client_label")]
        merge_df = merge_df[[c for c in cols if c in merge_df.columns]]
        merge_df.to_csv(out_dir / "per_client_metrics.csv", index=False)

    # Per-client curves: one line per client; legend uses brain region (client_label)
    clients = sorted(merge_df["client_id"].unique().tolist()) if merge_df is not None and not merge_df.empty else []
    rounds = sorted(merge_df["round"].unique().tolist()) if merge_df is not None and not merge_df.empty else []
    if not clients or not rounds:
        return
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(clients), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (metric, title) in zip(
        axes.flat,
        [
            ("train_loss", "Per-client train loss"),
            ("train_accuracy", "Per-client train accuracy"),
            ("val_loss", "Per-client val loss"),
            ("val_accuracy", "Per-client val accuracy"),
        ],
    ):
        if metric not in merge_df.columns:
            ax.set_visible(False)
            continue
        for i, cid in enumerate(clients):
            sub = merge_df[merge_df["client_id"] == cid].sort_values("round")
            if sub.empty:
                continue
            label = client_id_to_label.get(cid, cid)
            ax.plot(sub["round"], sub[metric], label=label, color=colors[i % len(colors)], marker="o", markersize=4)
        ax.set_xlabel("Round")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "per_client_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved per_client_fit_metrics.json, per_client_eval_metrics.json, per_client_metrics.csv, plots/per_client_curves.png")


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
    Run final evaluation of the global model on the global test set.
    
    Args:
        model: Trained global model
        clients: List of client IDs (for metadata)
        genes: Gene list for data loading
        data_dir: Path to processed data
        batch_size: Batch size for evaluation
        include_spatial: Whether to include spatial features
        device: Device to run on
        num_workers: Number of data loading workers
        
    Returns:
        Dictionary with global test metrics
    """
    print("\n" + "=" * 60)
    print("Final Evaluation on Global Test Set")
    print("=" * 60)
    
    # Load global test set (shared across all clients)
    try:
        test_df = load_global_test(data_dir=data_dir, validate=True)
    except FileNotFoundError as e:
        print(f"Error: Could not load global test set: {e}")
        return {
            "global_test": {"loss": None, "accuracy": None, "f1_macro": None, "total_samples": 0},
            "clients": clients,
        }
    
    test_loader = create_dataloader(
        test_df, genes,
        batch_size=batch_size,
        shuffle=False,
        include_spatial=include_spatial,
        num_workers=num_workers,
    )
    
    print(f"\nGlobal test set ({len(test_df):,} samples):")
    metrics = evaluate(model, test_loader, device, verbose=True, use_amp=use_amp)
    
    global_test = {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "f1_macro": float(metrics["f1_macro"]),
        "total_samples": len(test_df),
    }
    
    print("\n" + "-" * 40)
    print(f"Global Test: Loss={global_test['loss']:.4f}, "
          f"Acc={global_test['accuracy']:.4f}, F1={global_test['f1_macro']:.4f}")
    
    return {
        "global_test": global_test,
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
        help="Batch size for training. Reduce (e.g. 256 or 512) if OOM in Ray client actors."
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
        client_ids=clients,
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

    # Per-client metrics and curves (for non-IID analysis and transparency)
    # Use brain-region labels (dorsal, mid, ventral) from client_meta.json for graph legends
    client_id_to_label = _load_client_labels(Path(data_dir), clients)
    _save_per_client_metrics_and_plots(
        getattr(strategy, "per_client_fit_metrics", []),
        getattr(strategy, "per_client_eval_metrics", []),
        out_dir,
        client_id_to_label=client_id_to_label,
    )

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
