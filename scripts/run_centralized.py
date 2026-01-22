"""
Centralized Baseline Training Runner (UPDATED for GLOBAL TEST SPLIT)

Key changes vs old version:
- Clients provide TRAIN + VAL only
- A single GLOBAL test set is used for evaluation
- No per-client test files are assumed
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    create_dataloader,
    load_all_clients,
    load_gene_list,
    load_label_map,
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _list_clients(data_dir: str) -> List[str]:
    clients_dir = Path(data_dir) / "clients"
    if not clients_dir.exists():
        return []
    return sorted(
        p.name for p in clients_dir.iterdir()
        if p.is_dir() and p.name.startswith("client_")
    )


def _save_training_curves(history: TrainingHistory, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    h = history.to_dict()
    epochs = np.arange(1, len(h["train"]["loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, h["train"]["loss"], label="train_loss")
    plt.plot(epochs, h["val"]["loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, h["train"]["accuracy"], label="train_acc")
    plt.plot(epochs, h["val"]["accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png")
    plt.close()

    # F1
    plt.figure()
    plt.plot(epochs, h["train"]["f1_macro"], label="train_f1")
    plt.plot(epochs, h["val"]["f1_macro"], label="val_f1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Macro")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "f1_macro_curve.png")
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Centralized training (global test)")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results/centralized")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--fine_tune_mode", type=str, default="head_only",
                        choices=["head_only", "partial", "full"])
    parser.add_argument("--include_spatial", action="store_true", default=True)
    parser.add_argument("--no_spatial", action="store_true")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    include_spatial = args.include_spatial and not args.no_spatial

    # Num workers safety
    import platform
    if args.num_workers is None:
        args.num_workers = 0 if platform.system() == "Windows" else 4

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = args.data_dir
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Discover clients
    # -------------------------------------------------------------------------
    clients = _list_clients(data_dir)
    if not clients:
        raise FileNotFoundError("No clients found under data/processed/clients")

    print(f"Centralized training on pooled clients: {clients}")

    # -------------------------------------------------------------------------
    # Load data (TRAIN + VAL from clients)
    # -------------------------------------------------------------------------
    genes = load_gene_list(data_dir)
    num_genes = len(genes)
    num_labels = len(load_label_map(data_dir))

    train_df = load_all_clients("train", data_dir=data_dir, validate=True)
    val_df = load_all_clients("val", data_dir=data_dir, validate=True)

    # -------------------------------------------------------------------------
    # Load GLOBAL test set
    # -------------------------------------------------------------------------
    global_test_path = Path(data_dir) / "global" / "test.parquet"
    if not global_test_path.exists():
        raise FileNotFoundError(
            f"Global test set not found: {global_test_path}. "
            "Run updated partition_clients.py first."
        )

    test_df = pd.read_parquet(global_test_path)

    print(f"Train / Val / Global-Test sizes: "
          f"{len(train_df):,} / {len(val_df):,} / {len(test_df):,}")

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------
    train_loader = create_dataloader(
        train_df, genes,
        batch_size=args.batch_size,
        shuffle=True,
        include_spatial=include_spatial,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    val_loader = create_dataloader(
        val_df, genes,
        batch_size=args.batch_size,
        shuffle=False,
        include_spatial=include_spatial,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    test_loader = create_dataloader(
        test_df, genes,
        batch_size=args.batch_size,
        shuffle=False,
        include_spatial=include_spatial,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = create_model(
        num_genes=num_genes,
        num_labels=num_labels,
        pretrained_path=args.pretrained_path,
        fine_tune_mode=args.fine_tune_mode,
        include_spatial=include_spatial,
    )

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    model.to(device)

    use_amp = args.use_amp and not args.no_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    optimizer = create_optimizer(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        scheduler_type="cosine",
        num_epochs=args.epochs,
    )

    history = TrainingHistory()

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model, train_loader, optimizer,
            device, verbose=True,
            use_amp=use_amp, scaler=scaler
        )
        val_metrics = evaluate(
            model, val_loader,
            device, verbose=True,
            use_amp=use_amp
        )
        history.add_train_metrics(train_metrics)
        history.add_val_metrics(val_metrics)
        scheduler.step()

    # -------------------------------------------------------------------------
    # Final evaluation on GLOBAL TEST
    # -------------------------------------------------------------------------
    print("\nFinal evaluation on GLOBAL test set")
    global_test_metrics = evaluate(
        model, test_loader,
        device, verbose=True,
        use_amp=use_amp
    )

    # -------------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------------
    _save_training_curves(history, plots_dir)

    eval_summary = {
        "global_test": {
            "loss": global_test_metrics["loss"],
            "accuracy": global_test_metrics["accuracy"],
            "f1_macro": global_test_metrics["f1_macro"],
        },
        "clients": clients,
    }

    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

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
        experiment_name="centralized",
    )

    save_training_artifacts(
        output_dir=str(out_dir),
        model=model,
        history=history,
        config=asdict(config),
        metrics=None,
    )

    print(f"\n✔ Centralized training complete")
    print(f"✔ Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
