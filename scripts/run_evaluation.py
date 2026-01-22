#!/usr/bin/env python

"""
Evaluation script for CENTRALIZED vs FEDERATED learning
under a SINGLE GLOBAL TEST SET protocol.

Assumptions:
- No per-client test sets exist
- Both centralized and federated models are evaluated
  on the SAME global test set
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_experiment(history_path, metrics_path, eval_summary_path):
    with history_path.open("r") as f:
        history = json.load(f)

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
    else:
        metrics_df = None  # centralized has no metrics.csv

    with eval_summary_path.open("r") as f:
        eval_summary = json.load(f)

    return history, metrics_df, eval_summary


def load_predictions(pred_path: Path):
    """
    Optional: used only if you saved predictions.npz
    """
    if not pred_path.exists():
        return None, None
    data = np.load(pred_path)
    return data["y_true"], data["y_pred"]


# ---------------------------------------------------------------------
# Plotting: CENTRALIZED
# ---------------------------------------------------------------------

def plot_centralized_training_curves(history, out_dir: Path):
    epochs = range(1, len(history["train"]["loss"]) + 1)

    def _plot(metric, title, fname):
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["train"][metric], label="Train")
        plt.plot(epochs, history["val"][metric], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()

    _plot("loss", "Centralized Loss", "centralized_loss.png")
    _plot("accuracy", "Centralized Accuracy", "centralized_accuracy.png")
    _plot("f1_macro", "Centralized Macro-F1", "centralized_f1.png")


# ---------------------------------------------------------------------
# Plotting: FEDERATED
# ---------------------------------------------------------------------

def plot_federated_validation(metrics_df: pd.DataFrame, out_dir: Path):
    rounds = metrics_df["round"].values

    def _plot(metric, title, fname):
        if metric not in metrics_df.columns:
            return
        plt.figure(figsize=(6, 4))
        plt.plot(rounds, metrics_df[metric], marker="o")
        plt.xlabel("Round")
        plt.ylabel(metric)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()

    _plot("val_loss", "Federated Validation Loss", "federated_val_loss.png")
    _plot("val_accuracy", "Federated Validation Accuracy", "federated_val_accuracy.png")
    _plot("val_f1_macro", "Federated Validation Macro-F1", "federated_val_f1.png")


# ---------------------------------------------------------------------
# Plotting: GLOBAL TEST COMPARISON
# ---------------------------------------------------------------------

def plot_global_metrics(eval_centralized, eval_federated, out_dir: Path):
    metrics = ["loss", "accuracy", "f1_macro"]
    labels = ["Loss", "Accuracy", "Macro-F1"]

    centralized_vals = [eval_centralized["global_test"][m] for m in metrics]
    federated_vals = [eval_federated["global_test"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, centralized_vals, width, label="Centralized")
    plt.bar(x + width / 2, federated_vals, width, label="Federated")

    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title("Global Test Performance")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "global_test_comparison.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# Optional: Confusion matrices & per-class metrics
# ---------------------------------------------------------------------

def plot_confusion(y_true, y_pred, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(
        cmap="Blues",
        values_format=".2f",
        ax=ax,
        xticks_rotation="vertical",
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_class_metrics(
    y_true_c, y_pred_c, y_true_f, y_pred_f, out_dir: Path
):
    metrics = {
        "Precision": precision_recall_fscore_support(
            y_true_c, y_pred_c, average=None, zero_division=0
        )[0],
        "Recall": precision_recall_fscore_support(
            y_true_c, y_pred_c, average=None, zero_division=0
        )[1],
        "F1": precision_recall_fscore_support(
            y_true_c, y_pred_c, average=None, zero_division=0
        )[2],
    }

    metrics_f = {
        "Precision": precision_recall_fscore_support(
            y_true_f, y_pred_f, average=None, zero_division=0
        )[0],
        "Recall": precision_recall_fscore_support(
            y_true_f, y_pred_f, average=None, zero_division=0
        )[1],
        "F1": precision_recall_fscore_support(
            y_true_f, y_pred_f, average=None, zero_division=0
        )[2],
    }

    num_classes = len(metrics["F1"])
    x = np.arange(num_classes)
    width = 0.4

    for name in metrics:
        plt.figure(figsize=(10, 4))
        plt.bar(x - width / 2, metrics[name], width, label="Centralized")
        plt.bar(x + width / 2, metrics_f[name], width, label="Federated")
        plt.xlabel("Class")
        plt.ylabel(name)
        plt.title(f"Per-class {name}")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"per_class_{name.lower()}.png", dpi=200)
        plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parents[1]

    centralized_dir = project_root / "results" / "centralized"
    federated_dir = project_root / "results" / "federated"
    eval_dir = project_root / "results" / "evaluation"
    ensure_dir(eval_dir)

    cent_history, cent_metrics, cent_eval = load_experiment(
        centralized_dir / "history.json",
        centralized_dir / "metrics.csv",
        centralized_dir / "eval_summary.json",
    )

    fed_history, fed_metrics, fed_eval = load_experiment(
        federated_dir / "history.json",
        federated_dir / "metrics.csv",
        federated_dir / "eval_summary.json",
    )

    # Plots
    plot_centralized_training_curves(cent_history, eval_dir)
    plot_federated_validation(fed_metrics, eval_dir)
    plot_global_metrics(cent_eval, fed_eval, eval_dir)

    # Optional advanced evaluation
    y_true_c, y_pred_c = load_predictions(centralized_dir / "predictions.npz")
    y_true_f, y_pred_f = load_predictions(federated_dir / "predictions.npz")

    if y_true_c is not None and y_true_f is not None:
        plot_confusion(
            y_true_c,
            y_pred_c,
            eval_dir / "confusion_centralized.png",
            "Confusion Matrix (Centralized)",
        )
        plot_confusion(
            y_true_f,
            y_pred_f,
            eval_dir / "confusion_federated.png",
            "Confusion Matrix (Federated)",
        )
        plot_per_class_metrics(y_true_c, y_pred_c, y_true_f, y_pred_f, eval_dir)

    print(f"\n✔ Evaluation complete")
    print(f"✔ Results saved to: {eval_dir}")


if __name__ == "__main__":
    main()
