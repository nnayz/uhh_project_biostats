"""
Global Test Evaluation Script

Evaluates a trained model (centralized or federated)
on the GLOBAL test set and produces evaluation-only plots.
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# -------------------------------------------------
# Add project root to PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    create_dataloader,
    load_gene_list,
    load_label_map,
)
from src.model import create_model
from src.training import evaluate, load_model


# -------------------------------------------------
# Plot helpers
# -------------------------------------------------
def plot_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix (Global Test Set)")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar(values, labels, title, ylabel, out_path):
    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate model on global test set")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true")

    args = parser.parse_args()

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    out_dir = Path(args.output_dir) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Load global test data
    # -------------------------------------------------
    test_path = Path(args.data_dir) / "global" / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Global test set not found: {test_path}")

    test_df = pd.read_parquet(test_path)

    genes = load_gene_list(args.data_dir)
    label_map = load_label_map(args.data_dir)
    inv_label_map = {v: k for k, v in label_map.items()}

    test_loader = create_dataloader(
        test_df,
        genes,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------
    model = create_model(
        num_genes=len(genes),
        num_labels=len(label_map),
        pretrained_path=None,
        fine_tune_mode="full",
    )
    load_model(model, args.model_path)
    model.to(device)
    model.eval()

    # -------------------------------------------------
    # Run evaluation
    # -------------------------------------------------
    print("\nEvaluating on GLOBAL test set...")
    metrics, y_true, y_pred = evaluate(
        model,
        test_loader,
        device,
        return_predictions=True,
        use_amp=args.use_amp,
    )

    # -------------------------------------------------
    # Save metrics
    # -------------------------------------------------
    report = classification_report(
        y_true,
        y_pred,
        target_names=[inv_label_map[i] for i in range(len(label_map))],
        output_dict=True,
        zero_division=0,
    )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "overall": metrics,
                "classification_report": report,
            },
            f,
            indent=2,
        )

    # -------------------------------------------------
    # Plots
    # -------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm,
        labels=[inv_label_map[i] for i in range(len(label_map))],
        out_path=out_dir / "confusion_matrix.png",
    )

    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    plot_bar(
        f1,
        labels=[inv_label_map[i] for i in range(len(label_map))],
        title="Per-class F1 score (Global Test Set)",
        ylabel="F1 score",
        out_path=out_dir / "per_class_f1.png",
    )

    support = np.bincount(y_true, minlength=len(label_map))
    plot_bar(
        support,
        labels=[inv_label_map[i] for i in range(len(label_map))],
        title="Per-class sample count (Global Test Set)",
        ylabel="Number of samples",
        out_path=out_dir / "per_class_support.png",
    )

    print("\nEvaluation complete.")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
