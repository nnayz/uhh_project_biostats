"""
Comprehensive Model Evaluation & Comparison (Centralized, Federated, Local).

- Overall comparison: all three strategies on global test (accuracy, F1-macro, loss).
  Plots and summary include Centralized, Federated, and Local (mean across clients).
- Per-client comparison: Federated vs Local only (one federated model vs each client's
  local model on the global test set). Centralized is a single model and not compared
  per client.

Default: includes local training (--no_local to exclude).
Usage:
    python scripts/milestone3/evaluate_models.py
    python scripts/milestone3/evaluate_models.py --no_local
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    create_dataloader,
    load_client_data,
    load_gene_list,
    load_label_map,
    load_global_test,
)
from src.model import create_model
from src.training import evaluate


def _load_client_labels(data_dir: str, clients: List[str]) -> Dict[str, str]:
    """Load client01_dorsal-style labels from client_meta.json for plot axes."""
    out: Dict[str, str] = {}
    for cid in clients:
        meta_path = Path(data_dir) / "clients" / cid / "client_meta.json"
        region = cid
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                region = meta.get("group_value") or meta.get("batch_id") or meta.get("anatomical_region") or cid
                region = str(region)
            except Exception:
                pass
        short_id = cid.replace("client_", "client") if cid.startswith("client_") else cid
        out[cid] = f"{short_id}_{region}"
    return out


def load_model(
    model_path: str,
    config_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    print(f"\nLoading model from {model_path}...")
    
    # Load config to get model parameters
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model with same architecture
    model = create_model(
        num_genes=config['num_genes'],
        num_labels=config['num_labels'],
        pretrained_path=config.get('pretrained_path'),
        fine_tune_mode=config.get('fine_tune_mode', 'head_only'),
        include_spatial=config.get('include_spatial', True),
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"  Model loaded: {config['num_genes']} genes, {config['num_labels']} labels")
    return model


def evaluate_model_comprehensive(
    model: torch.nn.Module,
    clients: List[str],
    genes: List[str],
    data_dir: str,
    batch_size: int,
    include_spatial: bool,
    device: torch.device,
    num_workers: int = 0,
    use_amp: bool = False,
    per_client_eval: bool = True,
) -> Dict:
    """
    Comprehensive evaluation on global test set, optionally with per-client breakdown.
    
    Returns:
        Dictionary with:
        - global_test: metrics on global test set
        - per_client_test: metrics per client (if per_client_eval=True)
        - all_predictions: predictions for confusion matrix
        - all_labels: labels for confusion matrix
    """
    print("\n" + "=" * 60)
    print("Comprehensive Model Evaluation")
    print("=" * 60)
    
    # Load global test set (shared across all clients)
    try:
        test_df = load_global_test(data_dir=data_dir, validate=True)
        print(f"  Loaded global test set: {len(test_df):,} samples")
    except FileNotFoundError as e:
        raise ValueError(f"Could not load global test set: {e}")
    
    test_loader = create_dataloader(
        test_df, genes,
        batch_size=batch_size,
        shuffle=False,
        include_spatial=include_spatial,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    
    print(f"\nEvaluating on global test set ({len(test_df):,} samples)...")
    test_metrics = evaluate(model, test_loader, device, verbose=True, use_amp=use_amp)
    
    result = {
        "global_test": {
            "loss": float(test_metrics["loss"]),
            "accuracy": float(test_metrics["accuracy"]),
            "f1_macro": float(test_metrics["f1_macro"]),
            "total_samples": len(test_df),
        },
        "clients": clients,
        "predictions": test_metrics["predictions"],
        "labels": test_metrics["labels"],
    }
    
    # Per-client evaluation on global test set (filter by client's samples)
    if per_client_eval and 'sample_id' in test_df.columns:
        per_client_test = {}
        print(f"\nPer-client evaluation on global test set:")
        for client_id in clients:
            # Get the original client identifier from sample_id
            # Map client_id (client_01) to sample_id value (replicate 1)
            client_meta_path = Path(data_dir) / "clients" / client_id / "client_meta.json"
            if client_meta_path.exists():
                with open(client_meta_path, 'r') as f:
                    client_meta = json.load(f)
                group_value = client_meta.get('group_value', '')
                
                # Filter test set by this client's samples
                client_test_df = test_df[test_df['sample_id'] == group_value]
                
                if len(client_test_df) > 0:
                    client_test_loader = create_dataloader(
                        client_test_df, genes,
                        batch_size=batch_size,
                        shuffle=False,
                        include_spatial=include_spatial,
                        num_workers=num_workers,
                        pin_memory=(device.type == 'cuda'),
                    )
                    client_metrics = evaluate(model, client_test_loader, device, verbose=False, use_amp=use_amp)
                    per_client_test[client_id] = {
                        "loss": float(client_metrics["loss"]),
                        "accuracy": float(client_metrics["accuracy"]),
                        "f1_macro": float(client_metrics["f1_macro"]),
                        "num_samples": len(client_test_df),
                    }
                    print(f"  {client_id}: {len(client_test_df):,} samples, Acc={client_metrics['accuracy']:.4f}")
        
        # With held-out-batch strategy, eval set has one batch only; no client matches. Record held-out batch as single "client".
        if not per_client_test and "sample_id" in test_df.columns and test_df["sample_id"].nunique() == 1:
            held_out_id = test_df["sample_id"].iloc[0]
            result["per_client_test"] = {
                "held_out_batch": {
                    "loss": result["global_test"]["loss"],
                    "accuracy": result["global_test"]["accuracy"],
                    "f1_macro": result["global_test"]["f1_macro"],
                    "num_samples": result["global_test"]["total_samples"],
                }
            }
            print(f"  held_out_batch ({held_out_id}): {result['global_test']['total_samples']:,} samples (eval set)")
        else:
            result["per_client_test"] = per_client_test
    
    return result


def plot_confusion_matrix(
    labels: List[int],
    predictions: List[int],
    num_labels: int,
    output_path: Path,
    title: str = "Confusion Matrix",
    show_numbers: bool = False,
):
    """Create confusion matrix as color heatmap (viridis colormap for better visibility)."""
    cm = confusion_matrix(labels, predictions, labels=list(range(num_labels)))
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    cm_normalized = cm.astype('float') / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Count', fontsize=11)

    if show_numbers:
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=7)

    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix: {output_path}")
    plt.close()


def evaluate_local_models(
    clients: List[str],
    genes: List[str],
    data_dir: str,
    batch_size: int,
    include_spatial: bool,
    device: torch.device,
    num_workers: int = 0,
    use_amp: bool = False,
    local_results_dir: Path = None,
) -> Dict[str, Dict]:
    """
    Evaluate all local training models (one per client).
    
    Returns:
        Dictionary mapping client_id to evaluation results
    """
    if local_results_dir is None:
        local_results_dir = PROJECT_ROOT / "results"
    
    local_evals = {}
    
    print("\n" + "=" * 60)
    print("Evaluating Local Training Models")
    print("=" * 60)
    
    for client_id in clients:
        local_dir = local_results_dir / f"local_{client_id}"
        model_path = local_dir / "model_final.pt"
        config_path = local_dir / "config.json"
        
        if not model_path.exists() or not config_path.exists():
            print(f"  ⚠ Skipping {client_id}: model not found at {local_dir}")
            continue
        
        print(f"\n  Evaluating {client_id}...")
        try:
            model = load_model(str(model_path), str(config_path), device)
            eval_result = evaluate_model_comprehensive(
                model,
                [client_id],  # Only this client for metadata
                genes,
                data_dir,
                batch_size,
                include_spatial,
                device,
                num_workers,
                use_amp,
                per_client_eval=False,  # Local models don't need per-client breakdown
            )
            local_evals[client_id] = eval_result
            print(f"    ✓ {client_id}: Acc={eval_result['global_test']['accuracy']:.4f}, F1={eval_result['global_test']['f1_macro']:.4f}")
        except Exception as e:
            print(f"    ✗ Error evaluating {client_id}: {e}")
            continue
    
    return local_evals


def plot_per_client_comparison(
    centralized_eval: Dict,
    federated_per_client: Dict[str, Dict],
    output_path: Path,
    local_evals: Optional[Dict[str, Dict]] = None,
    client_id_to_label: Optional[Dict[str, str]] = None,
    federated_global_accuracy: Optional[float] = None,
    federated_global_f1: Optional[float] = None,
    clients: Optional[List[str]] = None,
):
    """Combined per-client comparison: Centralized, Federated, Local. Accuracy (left) and Macro-F1 (right) on the same scale."""
    if clients is None:
        clients = sorted(set(list(federated_per_client.keys()) + (list(local_evals.keys()) if local_evals else [])))
    
    # Centralized is a single model - same value for all clients
    cent_acc = centralized_eval['global_test']['accuracy']
    cent_f1 = centralized_eval['global_test']['f1_macro']
    cent_accs = [cent_acc] * len(clients)
    cent_f1s = [cent_f1] * len(clients)
    
    # Get federated values (use global if per-client not available)
    fed_accs = [federated_per_client.get(c, {}).get('accuracy', federated_global_accuracy) for c in clients]
    fed_f1s = [federated_per_client.get(c, {}).get('f1_macro', federated_global_f1) for c in clients]
    
    # Get local values (None if not evaluated)
    local_accs = [local_evals.get(c, {}).get('global_test', {}).get('accuracy') if (local_evals and c in local_evals) else None for c in clients]
    local_f1s = [local_evals.get(c, {}).get('global_test', {}).get('f1_macro') if (local_evals and c in local_evals) else None for c in clients]
    
    labels = [client_id_to_label.get(c, c) for c in clients] if client_id_to_label else clients
    
    # Compute y-axis range based on all values (with margin)
    all_vals = cent_accs + cent_f1s + fed_accs + fed_f1s + [v for v in local_accs if v is not None] + [v for v in local_f1s if v is not None]
    y_min = max(0, min(all_vals) - 0.05)
    y_max = min(1, max(all_vals) + 0.05)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x = np.arange(len(clients))
    width = 0.25  # Narrower bars for 3 strategies
    
    # Left: Accuracy
    ax = axes[0]
    ax.bar(x - width, cent_accs, width, label='Centralized', color='#3498db', alpha=0.85)
    ax.bar(x, fed_accs, width, label='Federated', color='#e74c3c', alpha=0.85)
    for i in range(len(clients)):
        if local_accs[i] is not None:
            ax.bar(x[i] + width, local_accs[i], width, color='#2ecc71', alpha=0.85, label='Local' if i == 0 else '')
    # Add value labels
    for i in range(len(clients)):
        ax.text(x[i] - width, cent_accs[i] + 0.01, f'{cent_accs[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(x[i], fed_accs[i] + 0.01, f'{fed_accs[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        if local_accs[i] is not None:
            ax.text(x[i] + width, local_accs[i] + 0.01, f'{local_accs[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Client', fontsize=12)
    ax.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([y_min, y_max])
    
    # Right: Macro-F1
    ax = axes[1]
    ax.bar(x - width, cent_f1s, width, label='Centralized', color='#3498db', alpha=0.85)
    ax.bar(x, fed_f1s, width, label='Federated', color='#e74c3c', alpha=0.85)
    for i in range(len(clients)):
        if local_f1s[i] is not None:
            ax.bar(x[i] + width, local_f1s[i], width, color='#2ecc71', alpha=0.85, label='Local' if i == 0 else '')
    # Add value labels
    for i in range(len(clients)):
        ax.text(x[i] - width, cent_f1s[i] + 0.01, f'{cent_f1s[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(x[i], fed_f1s[i] + 0.01, f'{fed_f1s[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        if local_f1s[i] is not None:
            ax.text(x[i] + width, local_f1s[i] + 0.01, f'{local_f1s[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xlabel('Client', fontsize=12)
    ax.set_title('Macro-F1', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Per-Client Comparison: Centralized vs Federated vs Local', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_overall_metrics_comparison(
    centralized_eval: Dict,
    federated_eval: Dict,
    local_evals: Optional[Dict[str, Dict]],
    output_path: Path,
):
    """Single clear chart: Accuracy and Macro-F1 for Centralized, Federated, Local (mean)."""
    metrics_names = ['Accuracy', 'Macro-F1']
    centralized_vals = [
        centralized_eval['global_test']['accuracy'],
        centralized_eval['global_test']['f1_macro'],
    ]
    federated_vals = [
        federated_eval['global_test']['accuracy'],
        federated_eval['global_test']['f1_macro'],
    ]
    if local_evals:
        local_vals = [
            np.mean([local_evals[c]['global_test']['accuracy'] for c in local_evals]),
            np.mean([local_evals[c]['global_test']['f1_macro'] for c in local_evals]),
        ]
    else:
        local_vals = None
    
    # Compute y-axis range based on all values
    all_vals = centralized_vals + federated_vals + (local_vals if local_vals else [])
    y_min = max(0, min(all_vals) - 0.05)
    y_max = min(1, max(all_vals) + 0.05)
    
    x = np.arange(len(metrics_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 6))
    
    bars1 = ax.bar(x - width, centralized_vals, width, label='Centralized', color='#3498db', alpha=0.9)
    bars2 = ax.bar(x, federated_vals, width, label='Federated', color='#e74c3c', alpha=0.9)
    if local_vals is not None:
        bars3 = ax.bar(x + width, local_vals, width, label='Local (mean)', color='#2ecc71', alpha=0.9)
    
    # Add value labels on bars
    for i, v in enumerate(centralized_vals):
        ax.text(x[i] - width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for i, v in enumerate(federated_vals):
        ax.text(x[i], v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    if local_vals is not None:
        for i, v in enumerate(local_vals):
            ax.text(x[i] + width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Test Performance: Centralized vs Federated vs Local', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([y_min, y_max])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Task 5: Comprehensive Model Evaluation")
    parser.add_argument(
        "--centralized_dir",
        type=str,
        default="results/centralized",
        help="Directory containing centralized model and results",
    )
    parser.add_argument(
        "--federated_dir",
        type=str,
        default="results/federated",
        help="Directory containing federated model and results",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparison",
        help="Output directory for comparison plots",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers (0 for Windows compatibility)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use Automatic Mixed Precision",
    )
    parser.add_argument(
        "--local_results_dir",
        type=str,
        default="results/local",
        help="Directory containing local training results (e.g. results/local/ with local_client_01, local_client_02 inside)",
    )
    parser.add_argument(
        "--no_local",
        action="store_true",
        help="Exclude local training from comparison (default: include centralized, federated, and local)",
    )
    
    args = parser.parse_args()
    include_local = not args.no_local
    
    device = torch.device(args.device)
    centralized_dir = Path(args.centralized_dir)
    federated_dir = Path(args.federated_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Task 5: Comprehensive Model Evaluation & Comparison")
    print("=" * 60)
    
    # Load configurations
    print("\nLoading configurations...")
    with open(centralized_dir / "config.json", 'r') as f:
        centralized_config = json.load(f)
    with open(federated_dir / "config.json", 'r') as f:
        federated_config = json.load(f)
    
    # Load histories
    print("\nLoading training histories...")
    with open(centralized_dir / "history.json", 'r') as f:
        centralized_history = json.load(f)
    with open(federated_dir / "history.json", 'r') as f:
        federated_history = json.load(f)
    
    # Load models
    centralized_model = load_model(
        centralized_dir / "model_final.pt",
        centralized_dir / "config.json",
        device,
    )
    federated_model = load_model(
        federated_dir / "model_final.pt",
        federated_dir / "config.json",
        device,
    )
    
    # Load data info
    genes = load_gene_list(args.data_dir)
    clients = federated_config.get('clients', ['client_01', 'client_02', 'client_03'])
    num_labels = centralized_config['num_labels']
    include_spatial = centralized_config.get('include_spatial', True)
    
    # Evaluate models
    print("\n" + "=" * 60)
    print("Evaluating Centralized Model")
    print("=" * 60)
    centralized_eval = evaluate_model_comprehensive(
        centralized_model,
        clients,
        genes,
        args.data_dir,
        args.batch_size,
        include_spatial,
        device,
        args.num_workers,
        args.use_amp,
        per_client_eval=True,  # Enable per-client breakdown
    )
    
    print("\n" + "=" * 60)
    print("Evaluating Federated Model")
    print("=" * 60)
    federated_eval = evaluate_model_comprehensive(
        federated_model,
        clients,
        genes,
        args.data_dir,
        args.batch_size,
        include_spatial,
        device,
        args.num_workers,
        args.use_amp,
        per_client_eval=True,  # Enable per-client breakdown
    )
    
    # Evaluate local models (default: include for full comparison)
    local_evals = {}
    if include_local:
        local_results_dir = Path(args.local_results_dir)
        local_evals = evaluate_local_models(
            clients,
            genes,
            args.data_dir,
            args.batch_size,
            include_spatial,
            device,
            args.num_workers,
            args.use_amp,
            local_results_dir,
        )
    
    # Add final metrics to history for plotting
    centralized_history['final_accuracy'] = centralized_eval['global_test']['accuracy']
    centralized_history['final_f1_macro'] = centralized_eval['global_test']['f1_macro']
    federated_history['final_accuracy'] = federated_eval['global_test']['accuracy']
    federated_history['final_f1_macro'] = federated_eval['global_test']['f1_macro']
    
    # Generate missing federated plots
    print("\n" + "=" * 60)
    print("Generating Missing Federated Plots")
    print("=" * 60)
    
    federated_plots_dir = federated_dir / "plots"
    federated_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Accuracy curve (from history if available, or use final eval)
    if 'rounds' in federated_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        rounds = [r['round'] for r in federated_history['rounds']]
        val_losses = [r.get('val_loss', 0) for r in federated_history['rounds']]
        ax.plot(rounds, val_losses, 'r-o', linewidth=2, markersize=8)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title('Federated Training: Validation Loss', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(federated_plots_dir / "loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {federated_plots_dir / 'loss_curve.png'}")
    
    # Confusion matrix for federated
    plot_confusion_matrix(
        federated_eval['labels'],
        federated_eval['predictions'],
        num_labels,
        federated_plots_dir / "confusion_matrix.png",
        title="Federated Model: Confusion Matrix",
    )
    
    # Per-client accuracy for federated
    if 'per_client_test' in federated_eval:
        clients_list = sorted(federated_eval['per_client_test'].keys())
        accs = [federated_eval['per_client_test'][c]['accuracy'] for c in clients_list]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(clients_list, accs, color='red', alpha=0.7)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('Client', fontsize=12)
        ax.set_title('Federated Model: Per-Client Test Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylim([0.9, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        for i, (client, acc) in enumerate(zip(clients_list, accs)):
            ax.text(i, acc + 0.001, f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(federated_plots_dir / "per_client_accuracy.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {federated_plots_dir / 'per_client_accuracy.png'}")
    
    # Generate comparison plots
    print("\n" + "=" * 60)
    print("Generating Comparison Plots")
    print("=" * 60)
    
    # Main comparison: Accuracy and Macro-F1 only (Centralized, Federated, Local mean)
    plot_overall_metrics_comparison(
        centralized_eval,
        federated_eval,
        local_evals if local_evals else None,
        output_dir / "overall_metrics_comparison.png",
    )
    
    client_id_to_label = _load_client_labels(args.data_dir, clients)
    fed_pc = federated_eval.get('per_client_test', {})
    fed_global_acc = federated_eval['global_test']['accuracy']
    fed_global_f1 = federated_eval['global_test']['f1_macro']
    local_evals_opt = local_evals if local_evals else None

    # Per-client comparison: Centralized, Federated, Local — Accuracy (left) and Macro-F1 (right)
    plot_per_client_comparison(
        centralized_eval,
        fed_pc,
        output_dir / "per_client_comparison.png",
        local_evals=local_evals_opt,
        client_id_to_label=client_id_to_label,
        federated_global_accuracy=fed_global_acc,
        federated_global_f1=fed_global_f1,
        clients=clients,
    )
    
    # Confusion matrices (color heatmap only, no numbers)
    plot_confusion_matrix(
        centralized_eval['labels'],
        centralized_eval['predictions'],
        num_labels,
        output_dir / "confusion_matrix_centralized.png",
        title="Centralized Model: Confusion Matrix",
    )
    
    plot_confusion_matrix(
        federated_eval['labels'],
        federated_eval['predictions'],
        num_labels,
        output_dir / "confusion_matrix_federated.png",
        title="Federated Model: Confusion Matrix",
    )
    
    # Save evaluation summary (convert numpy types to native Python types for JSON)
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    # Build per-client metrics summary
    per_client_metrics = {}
    centralized_per_client = centralized_eval.get('per_client_test', {})
    federated_per_client = federated_eval.get('per_client_test', {})
    
    for client_id in clients:
        per_client_metrics[client_id] = {
            "centralized": centralized_per_client.get(client_id, centralized_eval.get('global_test', {})),
            "federated": federated_per_client.get(client_id, federated_eval.get('global_test', {})),
        }
        if client_id in local_evals:
            per_client_metrics[client_id]["local"] = local_evals[client_id].get('global_test', {})
    
    # Overall comparison: all three strategies (centralized, federated, local)
    comparison = {
        "overall": {
            "centralized": {
                "accuracy": float(centralized_eval['global_test']['accuracy']),
                "f1_macro": float(centralized_eval['global_test']['f1_macro']),
                "loss": float(centralized_eval['global_test']['loss']),
            },
            "federated": {
                "accuracy": float(federated_eval['global_test']['accuracy']),
                "f1_macro": float(federated_eval['global_test']['f1_macro']),
                "loss": float(federated_eval['global_test']['loss']),
            },
        },
        "accuracy_diff_centralized_minus_federated": float(centralized_eval['global_test']['accuracy'] - federated_eval['global_test']['accuracy']),
        "f1_diff_centralized_minus_federated": float(centralized_eval['global_test']['f1_macro'] - federated_eval['global_test']['f1_macro']),
        "loss_diff_centralized_minus_federated": float(centralized_eval['global_test']['loss'] - federated_eval['global_test']['loss']),
    }
    if local_evals:
        local_accs = [local_evals[c]['global_test']['accuracy'] for c in local_evals]
        local_f1s = [local_evals[c]['global_test']['f1_macro'] for c in local_evals]
        local_losses = [local_evals[c]['global_test']['loss'] for c in local_evals]
        comparison["overall"]["local_mean"] = {
            "accuracy": float(np.mean(local_accs)),
            "f1_macro": float(np.mean(local_f1s)),
            "loss": float(np.mean(local_losses)),
        }
    eval_summary = {
        "centralized": convert_to_native(centralized_eval),
        "federated": convert_to_native(federated_eval),
        "local": convert_to_native(local_evals) if local_evals else {},
        "per_client_metrics": convert_to_native(per_client_metrics),
        "comparison": comparison,
    }
    
    with open(output_dir / "evaluation_summary.json", 'w') as f:
        json.dump(eval_summary, f, indent=2)
    
    print(f"\n✓ Saved evaluation summary: {output_dir / 'evaluation_summary.json'}")
    
    # Print summary: overall (all three strategies)
    print("\n" + "=" * 60)
    print("Evaluation Summary (overall: Centralized, Federated, Local)")
    print("=" * 60)
    print(f"\nCentralized Model:")
    print(f"  Test Accuracy: {centralized_eval['global_test']['accuracy']:.4f}")
    print(f"  Test F1-Macro: {centralized_eval['global_test']['f1_macro']:.4f}")
    print(f"  Test Loss: {centralized_eval['global_test']['loss']:.4f}")
    
    print(f"\nFederated Model:")
    print(f"  Test Accuracy: {federated_eval['global_test']['accuracy']:.4f}")
    print(f"  Test F1-Macro: {federated_eval['global_test']['f1_macro']:.4f}")
    print(f"  Test Loss: {federated_eval['global_test']['loss']:.4f}")
    
    if local_evals:
        local_accs = [local_evals[c]['global_test']['accuracy'] for c in local_evals]
        local_f1s = [local_evals[c]['global_test']['f1_macro'] for c in local_evals]
        print(f"\nLocal (mean across clients):")
        print(f"  Test Accuracy: {np.mean(local_accs):.4f}")
        print(f"  Test F1-Macro: {np.mean(local_f1s):.4f}")
    
    comp = eval_summary['comparison']
    print(f"\nComparison (Centralized - Federated):")
    print(f"  Accuracy Difference: {comp['accuracy_diff_centralized_minus_federated']:.4f}")
    print(f"  F1-Macro Difference: {comp['f1_diff_centralized_minus_federated']:.4f}")
    print(f"  Loss Difference: {comp['loss_diff_centralized_minus_federated']:.4f}")
    
    # Per-client: Federated vs Local only
    fed_pc = federated_eval.get('per_client_test', {})
    print(f"\nPer-Client (Federated vs Local, on global test set):")
    print(f"{'Client':<20} {'Federated':<12} {'Local':<12}")
    print("-" * 44)
    fed_global_acc = federated_eval['global_test']['accuracy']
    for client_id in clients:
        f_metrics = fed_pc.get(client_id, {})
        f_acc = f_metrics.get('accuracy', fed_global_acc)
        l_acc = local_evals[client_id].get('global_test', {}).get('accuracy', 0) if local_evals and client_id in local_evals else None
        label = client_id_to_label.get(client_id, client_id)
        if l_acc is not None:
            print(f"{label:<20} {f_acc:<12.4f} {l_acc:<12.4f}")
        else:
            print(f"{label:<20} {f_acc:<12.4f} —")
    
    print("\n" + "=" * 60)
    print("✅ Task 5 Evaluation Complete!")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
