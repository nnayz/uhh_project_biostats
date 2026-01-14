"""\
Centralized Baseline Training Runner (Milestone 3 - Task 1)

Runs a pooled (non-federated) fine-tuning experiment using the shared:
  - Data Contract (src/data)
  - Model Contract (src/model)
  - Training Contract (src/training)

Outputs (Evaluation + Training Contracts):
  results/centralized/
    history.json
    metrics.csv
    eval_summary.json
    model_final.pt
    config.yaml
    plots/*.png

Example:
  python scripts/run_centralized.py --data_dir data/processed --device cpu --epochs 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

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
    load_all_clients,
    load_client_data,
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


def _list_clients(data_dir: str) -> List[str]:
    clients_dir = Path(data_dir) / "clients"
    if not clients_dir.exists():
        return []
    return sorted([p.name for p in clients_dir.iterdir() if p.is_dir() and p.name.startswith("client_")])


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


def _save_per_client_accuracy(per_client: Dict[str, Dict[str, float]], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    clients = list(per_client.keys())
    accs = [per_client[c]["accuracy"] for c in clients]
    plt.figure()
    plt.bar(clients, accs)
    plt.ylabel("accuracy")
    plt.title("Per-client test accuracy")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Centralized baseline training (pooled clients)")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--output_dir", type=str, default="results/centralized", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Training device")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (increased for GPU utilization)")
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
        if platform.system() == 'Windows':
            args.num_workers = 0
            print("[INFO] Windows detected: Setting num_workers=0 (multiprocessing can cause issues on Windows)")
        else:
            args.num_workers = 4  # Default for Linux/Mac

    include_spatial = args.include_spatial and not args.no_spatial

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = args.data_dir
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    clients = _list_clients(data_dir)
    if not clients:
        raise FileNotFoundError(
            f"No clients found under {Path(data_dir) / 'clients'}. "
            "Make sure Milestone 2 outputs exist, or pass --data_dir to the correct location."
        )

    genes = load_gene_list(data_dir)
    num_genes = len(genes)

    train_df = load_all_clients("train", data_dir=data_dir, validate=True)
    val_df = load_all_clients("val", data_dir=data_dir, validate=True)
    test_df = load_all_clients("test", data_dir=data_dir, validate=True)

    if args.max_train is not None and len(train_df) > args.max_train:
        train_df = train_df.sample(n=args.max_train, random_state=args.seed).reset_index(drop=True)
    if args.max_val is not None and len(val_df) > args.max_val:
        val_df = val_df.sample(n=args.max_val, random_state=args.seed).reset_index(drop=True)
    if args.max_test is not None and len(test_df) > args.max_test:
        test_df = test_df.sample(n=args.max_test, random_state=args.seed).reset_index(drop=True)

    num_labels = len(load_label_map(data_dir))

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
        test_df, genes, batch_size=args.batch_size, shuffle=False,
        include_spatial=include_spatial, num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    model = create_model(
        num_genes=num_genes, num_labels=num_labels,
        pretrained_path=args.pretrained_path,
        fine_tune_mode=args.fine_tune_mode,
        include_spatial=include_spatial,
    )
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(device)

    # Setup AMP (Automatic Mixed Precision) for GPU training
    use_amp = args.use_amp and not args.no_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) for faster GPU training")

    optimizer = create_optimizer(model, learning_rate=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, scheduler_type="cosine", num_epochs=args.epochs)

    history = TrainingHistory()

    print(f"Centralized training on pooled clients: {clients}")
    print(f"Train/Val/Test sizes: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
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

    print("\nFinal evaluation on pooled test")
    pooled_test = evaluate(model, test_loader, device, verbose=True, use_amp=use_amp)

    per_client_test: Dict[str, Dict[str, float]] = {}
    for cid in clients:
        cdf = load_client_data(cid, "test", data_dir=data_dir, validate=True)
        cloader = create_dataloader(
            cdf, genes, batch_size=args.batch_size, shuffle=False,
            include_spatial=include_spatial, num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )
        print(f"Client {cid} test:")
        m = evaluate(model, cloader, device, verbose=True, use_amp=use_amp)
        per_client_test[cid] = {"loss": m["loss"], "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]}

    _save_training_curves(history, plots_dir / "loss_curve.png")
    _save_confusion_matrix(pooled_test["labels"], pooled_test["predictions"], num_labels, plots_dir / "confusion_matrix.png")
    _save_per_client_accuracy(per_client_test, plots_dir / "per_client_accuracy.png")

    eval_summary = {
        "global_test": {"loss": pooled_test["loss"], "accuracy": pooled_test["accuracy"], "f1_macro": pooled_test["f1_macro"]},
        "per_client_test": per_client_test,
        "clients": clients,
    }
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    rows = [{"split": "pooled_test", **eval_summary["global_test"]}]
    for cid, m in per_client_test.items():
        rows.append({"split": f"test_{cid}", **m})
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
        experiment_name="centralized",
    )

    save_training_artifacts(
        output_dir=str(out_dir),
        model=model,
        history=history,
        config=asdict(config),
        metrics=None,
    )

    print(f"\nSaved artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
