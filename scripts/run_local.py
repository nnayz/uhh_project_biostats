"""\
Local Training Runner (Milestone 4 - Task 3)

Trains a centralized model on a single client's local data only.
This simulates what would happen if each client trained independently
without federated learning.

Outputs (Evaluation + Training Contracts):
  results/local_client_XX/
    history.json
    metrics.csv
    eval_summary.json
    model_final.pt
    config.yaml
    plots/*.png

Examples:
  python scripts/run_local.py --client_id client_01 --data_dir data/processed --device cpu --epochs 10
  python scripts/run_local.py --client_id all --data_dir data/processed --device cuda --epochs 10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
    create_optimizer,
    create_scheduler,
    evaluate,
    save_training_artifacts,
    train_one_epoch,
)
from src.config import TrainingConfig


def _list_clients(data_dir: str) -> List[str]:
    """Discover client IDs (client_01, client_02, ...) from data_dir/clients/."""
    clients_dir = Path(data_dir) / "clients"
    if not clients_dir.exists():
        return []
    return sorted(
        p.name for p in clients_dir.iterdir()
        if p.is_dir() and p.name.startswith("client_")
    )


def _load_client_label(data_dir: str, client_id: str) -> str:
    """Load combined label (e.g. client01_dorsal) from client_meta.json."""
    meta_path = Path(data_dir) / "clients" / client_id / "client_meta.json"
    if not meta_path.exists():
        short = client_id.replace("client_", "client") if client_id.startswith("client_") else client_id
        return f"{short}_?"
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        region = meta.get("group_value") or meta.get("batch_id") or meta.get("anatomical_region") or "?"
        short = client_id.replace("client_", "client") if client_id.startswith("client_") else client_id
        return f"{short}_{region}"
    except Exception:
        short = client_id.replace("client_", "client") if client_id.startswith("client_") else client_id
        return f"{short}_?"


def _save_training_curves(history: TrainingHistory, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    h = history.to_dict()
    epochs = np.arange(1, len(h["train"]["loss"]) + 1)

    plt.figure()
    plt.plot(epochs, h["train"]["loss"], label="train_loss")
    plt.plot(epochs, h["val"]["loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    out_png2 = out_png.with_name("accuracy_curve.png")
    plt.figure()
    plt.plot(epochs, h["train"]["accuracy"], label="train_acc")
    plt.plot(epochs, h["val"]["accuracy"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png2)
    plt.close()

    out_png3 = out_png.with_name("f1_macro_curve.png")
    plt.figure()
    plt.plot(epochs, h["train"]["f1_macro"], label="train_f1_macro")
    plt.plot(epochs, h["val"]["f1_macro"], label="val_f1_macro")
    plt.xlabel("epoch")
    plt.ylabel("f1_macro")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png3)
    plt.close()


def _save_confusion_matrix(y_true: List[int], y_pred: List[int], num_labels: int, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(include_values=False, xticks_rotation="vertical", ax=ax)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _run_one_client(
    client_id: str,
    data_dir: str,
    args: argparse.Namespace,
    genes: List[str],
    num_genes: int,
    num_labels: int,
    test_df: pd.DataFrame,
    include_spatial: bool,
    base_output_dir: Path,
    out_dir_override: Optional[Path] = None,
) -> None:
    """
    Train one model on a single client's data only, evaluate on held-out set, save artifacts.
    Model is created fresh each call; no shared state with other clients.
    """
    # Reset seed per client for reproducibility (same init if same pretrained)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = out_dir_override if out_dir_override is not None else base_output_dir / f"local_{client_id}"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    client_label = _load_client_label(data_dir, client_id)

    # Load only this client's train and val
    train_df = load_client_data(client_id, "train", data_dir=data_dir, validate=True)
    val_df = load_client_data(client_id, "val", data_dir=data_dir, validate=True)

    if args.max_train is not None and len(train_df) > args.max_train:
        train_df = train_df.sample(n=args.max_train, random_state=args.seed).reset_index(drop=True)
    if args.max_val is not None and len(val_df) > args.max_val:
        val_df = val_df.sample(n=args.max_val, random_state=args.seed).reset_index(drop=True)
    test_df_use = test_df
    if args.max_test is not None and len(test_df_use) > args.max_test:
        test_df_use = test_df_use.sample(n=args.max_test, random_state=args.seed).reset_index(drop=True)

    train_loader = create_dataloader(
        train_df, genes, batch_size=args.batch_size, shuffle=True,
        include_spatial=include_spatial, num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = create_dataloader(
        val_df, genes, batch_size=args.batch_size, shuffle=False,
        include_spatial=include_spatial, num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    test_loader = create_dataloader(
        test_df_use, genes, batch_size=args.batch_size, shuffle=False,
        include_spatial=include_spatial, num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    # Fresh model for this client (no leakage from other clients)
    model = create_model(
        num_genes=num_genes, num_labels=num_labels,
        pretrained_path=args.pretrained_path,
        fine_tune_mode=args.fine_tune_mode,
        include_spatial=include_spatial,
    )
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(device)

    use_amp = args.use_amp and not args.no_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    optimizer = create_optimizer(model, learning_rate=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, scheduler_type="cosine", num_epochs=args.epochs)

    history = TrainingHistory()

    print(f"\n{'='*60}")
    print(f"Local training on {client_label} ({client_id}) data only")
    print(f"{'='*60}")
    print(f"Train/Val/Test sizes: {len(train_df):,} / {len(val_df):,} / {len(test_df_use):,}")
    print(f"Device: {device}, Batch size: {args.batch_size}, Workers: {args.num_workers}")
    if hasattr(model, "count_parameters"):
        total_params, trainable_params = model.count_parameters()
        print(f"Model params: total={total_params:,}, trainable={trainable_params:,}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, verbose=True, use_amp=use_amp, scaler=scaler)
        val_metrics = evaluate(model, val_loader, device, verbose=True, use_amp=use_amp)
        history.add_train_metrics(train_metrics)
        history.add_val_metrics(val_metrics)
        if scheduler is not None:
            scheduler.step()

    print("\nFinal evaluation on global test set")
    global_test_metrics = evaluate(model, test_loader, device, verbose=True, use_amp=use_amp)

    _save_training_curves(history, plots_dir / "loss_curve.png")
    _save_confusion_matrix(global_test_metrics["labels"], global_test_metrics["predictions"], num_labels, plots_dir / "confusion_matrix.png")

    eval_summary = {
        "client_id": client_id,
        "client_label": client_label,
        "global_test": {
            "loss": global_test_metrics["loss"],
            "accuracy": global_test_metrics["accuracy"],
            "f1_macro": global_test_metrics["f1_macro"],
            "total_samples": len(test_df_use),
        },
        "train_samples": len(train_df),
        "val_samples": len(val_df),
    }
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    rows = [{"split": "global_test", **eval_summary["global_test"]}]
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    config = TrainingConfig(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_spatial=include_spatial,
        num_genes=num_genes,
        num_labels=num_labels,
        fine_tune_mode=args.fine_tune_mode,
        pretrained_path=args.pretrained_path,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=str(device),
        output_dir=str(out_dir),
        experiment_name=f"local_{client_id}",
    )
    config_dict = asdict(config)
    config_dict["client_label"] = client_label

    save_training_artifacts(
        output_dir=str(out_dir),
        model=model,
        history=history,
        config=config_dict,
        metrics=None,
    )

    print(f"Saved artifacts to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local training (single client's data only; use --client_id all for all clients)")
    parser.add_argument("--client_id", type=str, default="all", help="Client ID (e.g. client_01) or 'all' to train all clients in one run")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto: results/ or results/local_{client_id}); with --client_id all, writes to output_dir/local_client_XX")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Training device")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--fine_tune_mode", type=str, default="head_only", choices=["head_only", "partial", "full"])
    parser.add_argument("--include_spatial", action="store_true", default=True, help="Include spatial coords")
    parser.add_argument("--no_spatial", action="store_true", help="Disable spatial coords")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Optional pretrained Nicheformer checkpoint")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of data loading workers (0=main thread, 4-8 recommended for GPU). Auto-set to 0 on Windows.")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use Automatic Mixed Precision (AMP) for faster training")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train", type=int, default=None, help="Optional cap for training rows (debug)")
    parser.add_argument("--max_val", type=int, default=None, help="Optional cap for val rows (debug)")
    parser.add_argument("--max_test", type=int, default=None, help="Optional cap for test rows (debug)")
    args = parser.parse_args()

    # Auto-set num_workers on Windows (multiprocessing issues)
    import platform
    if args.num_workers is None:
        if platform.system() == "Windows":
            args.num_workers = 0
            print("[INFO] Windows detected: Setting num_workers=0 (multiprocessing can cause issues on Windows)")
        else:
            args.num_workers = 4

    include_spatial = args.include_spatial and not args.no_spatial
    data_dir = args.data_dir

    # Discover and validate clients
    available = _list_clients(data_dir)
    if not available:
        raise FileNotFoundError(
            f"No clients found in {Path(data_dir) / 'clients'}. Run partition_anatomical_siloing.py first."
        )

    if args.client_id == "all":
        clients_to_run = available
        print(f"Running local training for all {len(clients_to_run)} clients: {clients_to_run}")
    else:
        if args.client_id not in available:
            raise FileNotFoundError(
                f"Client '{args.client_id}' not found. Available clients: {available}. "
                f"Use --client_id one of {available} or --client_id all."
            )
        clients_to_run = [args.client_id]

    # Base output directory: each client writes to base_output_dir/local_client_XX
    # Default: results/local/ (so outputs go to results/local/local_client_01, etc.)
    # Single client + --output_dir: write to that path via out_dir_override
    base_output_dir = Path(args.output_dir) if args.output_dir else Path("results") / "local"

    # Load shared data once (genes, num_labels, held-out test set)
    genes = load_gene_list(data_dir)
    num_genes = len(genes)
    num_labels = len(load_label_map(data_dir))
    test_df = load_global_test(data_dir=data_dir, validate=True)
    if args.max_test is not None and len(test_df) > args.max_test:
        test_df = test_df.sample(n=args.max_test, random_state=args.seed).reset_index(drop=True)

    if use_amp := (args.use_amp and not args.no_amp):
        print("Using Automatic Mixed Precision (AMP) for faster GPU training")

    for client_id in clients_to_run:
        out_dir_override = Path(args.output_dir) if args.output_dir and len(clients_to_run) == 1 else None
        _run_one_client(
            client_id=client_id,
            data_dir=data_dir,
            args=args,
            genes=genes,
            num_genes=num_genes,
            num_labels=num_labels,
            test_df=test_df,
            include_spatial=include_spatial,
            base_output_dir=base_output_dir,
            out_dir_override=out_dir_override,
        )

    if len(clients_to_run) > 1:
        print(f"\nDone. Local training completed for {len(clients_to_run)} clients. Outputs: {base_output_dir}/local_client_XX/")


if __name__ == "__main__":
    main()
