"""
Federated Learning Server - Strategy and Aggregation Utilities

Implements the Federation Contract for server-side federated learning.
The server:
  - Initializes global model weights
  - Dispatches weights to clients
  - Aggregates updates using FedAvg
  - Logs round-level metrics

See training.md (project root) for federated training summary.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import flwr as fl
from flwr.common import (
    FitRes,
    EvaluateRes,
    Metrics,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


# ---------------------------------------------------------------------------
# Metric Aggregation Functions
# ---------------------------------------------------------------------------

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Compute weighted average of metrics across clients.
    
    Weighting is proportional to number of samples (num_examples).
    This is used by FedAvg for proper aggregation.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples from each client
        
    Returns:
        Dictionary of aggregated metrics
    """
    if not metrics:
        return {}
    
    # Collect all metric keys
    all_keys = set()
    for _, m in metrics:
        all_keys.update(m.keys())
    
    # Compute weighted average for each metric
    aggregated = {}
    total_examples = sum(num for num, _ in metrics)
    
    for key in all_keys:
        weighted_sum = 0.0
        valid_examples = 0
        
        for num_examples, m in metrics:
            if key in m:
                weighted_sum += num_examples * float(m[key])
                valid_examples += num_examples
        
        if valid_examples > 0:
            aggregated[key] = weighted_sum / valid_examples
    
    return aggregated


def aggregate_fit_metrics(
    results: List[Tuple[ClientProxy, FitRes]]
) -> Metrics:
    """
    Aggregate training metrics from fit results.
    
    This is a fit_metrics_aggregation_fn for FedAvg.
    
    Args:
        results: List of (client_proxy, fit_result) tuples
        
    Returns:
        Aggregated training metrics
    """
    metrics_list = []
    for _, fit_res in results:
        # Handle both old and new Flower API
        if hasattr(fit_res, 'metrics') and fit_res.metrics:
            metrics_list.append((fit_res.num_examples, fit_res.metrics))
        elif hasattr(fit_res, 'status') and hasattr(fit_res.status, 'metrics'):
            # New API: metrics might be in status
            if fit_res.status.metrics:
                metrics_list.append((fit_res.num_examples, fit_res.status.metrics))
    return weighted_average(metrics_list) if metrics_list else {}


def aggregate_evaluate_metrics(
    results: List[Tuple[ClientProxy, EvaluateRes]]
) -> Metrics:
    """
    Aggregate evaluation metrics from evaluate results.
    
    This is an evaluate_metrics_aggregation_fn for FedAvg.
    
    Args:
        results: List of (client_proxy, evaluate_result) tuples
        
    Returns:
        Aggregated evaluation metrics
    """
    metrics_list = []
    for _, eval_res in results:
        # Handle both old and new Flower API
        if hasattr(eval_res, 'metrics') and eval_res.metrics:
            metrics_list.append((eval_res.num_examples, eval_res.metrics))
        elif hasattr(eval_res, 'status') and hasattr(eval_res.status, 'metrics'):
            # New API: metrics might be in status
            if eval_res.status.metrics:
                metrics_list.append((eval_res.num_examples, eval_res.status.metrics))
    return weighted_average(metrics_list) if metrics_list else {}


# ---------------------------------------------------------------------------
# Strategy Builder
# ---------------------------------------------------------------------------

def create_fedavg_strategy(
    initial_parameters: Optional[Parameters] = None,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
    on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
) -> FedAvg:
    """
    Create a FedAvg strategy with proper metric aggregation.
    
    FedAvg weights client updates proportionally to the number of local
    training samples (handled automatically when clients return num_examples).
    
    Args:
        initial_parameters: Initial global model parameters
        fraction_fit: Fraction of clients to sample for training
        fraction_evaluate: Fraction of clients to sample for evaluation
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum number of available clients
        on_fit_config_fn: Function to configure clients for training
        on_evaluate_config_fn: Function to configure clients for evaluation
        
    Returns:
        Configured FedAvg strategy
    """
    return FedAvg(
        initial_parameters=initial_parameters,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )


def get_on_fit_config_fn(local_epochs: int) -> Callable[[int], Dict[str, Scalar]]:
    """
    Create a configuration function for client training.
    
    Args:
        local_epochs: Number of local training epochs
        
    Returns:
        Configuration function that returns training config for each round
    """
    def on_fit_config(server_round: int) -> Dict[str, Scalar]:
        return {
            "local_epochs": local_epochs,
            "server_round": server_round,
        }
    return on_fit_config


def get_on_evaluate_config_fn() -> Callable[[int], Dict[str, Scalar]]:
    """
    Create a configuration function for client evaluation.
    
    Returns:
        Configuration function that returns evaluation config for each round
    """
    def on_evaluate_config(server_round: int) -> Dict[str, Scalar]:
        return {
            "server_round": server_round,
        }
    return on_evaluate_config


# ---------------------------------------------------------------------------
# Round Logging Helpers
# ---------------------------------------------------------------------------

class RoundLogger:
    """
    Logger for tracking per-round metrics during federated training.
    """
    
    def __init__(self):
        self.rounds: List[Dict] = []
    
    def log_round(
        self,
        round_num: int,
        fit_metrics: Optional[Metrics] = None,
        eval_metrics: Optional[Metrics] = None,
        eval_loss: Optional[float] = None,
    ) -> None:
        """
        Log metrics for a training round.
        
        Args:
            round_num: Round number (1-indexed)
            fit_metrics: Aggregated training metrics
            eval_metrics: Aggregated evaluation metrics
            eval_loss: Aggregated evaluation loss
        """
        round_data = {"round": round_num}
        
        if fit_metrics:
            round_data["train_loss"] = fit_metrics.get("loss")
            round_data["train_accuracy"] = fit_metrics.get("accuracy")
            round_data["train_f1_macro"] = fit_metrics.get("f1_macro")
        
        if eval_metrics:
            round_data["val_accuracy"] = eval_metrics.get("accuracy")
            round_data["val_f1_macro"] = eval_metrics.get("f1_macro")
        
        if eval_loss is not None:
            round_data["val_loss"] = eval_loss
        
        self.rounds.append(round_data)
        
        # Print round summary
        parts = [f"Round {round_num}:"]
        if fit_metrics:
            parts.append(f"train_loss={fit_metrics.get('loss', 'N/A'):.4f}")
            parts.append(f"train_acc={fit_metrics.get('accuracy', 'N/A'):.4f}")
        if eval_loss is not None:
            parts.append(f"val_loss={eval_loss:.4f}")
        if eval_metrics:
            parts.append(f"val_acc={eval_metrics.get('accuracy', 'N/A'):.4f}")
        print(" | ".join(parts))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {"rounds": self.rounds}
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.rounds)
