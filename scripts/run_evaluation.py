#!/usr/bin/env python

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


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_centralized(history_path: Path, metrics_path: Path, eval_summary_path: Path):
    with history_path.open("r") as f:
        history = json.load(f)

    metrics_df = pd.read_csv(metrics_path)

    with eval_summary_path.open("r") as f:
        eval_summary = json.load(f)

    return history, metrics_df, eval_summary


def load_federated(history_path: Path, metrics_path: Path, eval_summary_path: Path):
    with history_path.open("r") as f:
        history = json.load(f)

    metrics_df = pd.read_csv(metrics_path)

    with eval_summary_path.open("r") as f:
        eval_summary = json.load(f)

    return history, metrics_df, eval_summary


def plot_centralized_training_curves(history, out_dir: Path):
    epochs = list(range(1, len(history["train"]["loss"]) + 1))

    # Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train"]["loss"], label="Train loss")
    plt.plot(epochs, history["val"]["loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Centralized training/validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "centralized_loss_curves.png", dpi=200)
    plt.close()

    # Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train"]["accuracy"], label="Train accuracy")
    plt.plot(epochs, history["val"]["accuracy"], label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Centralized training/validation accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "centralized_accuracy_curves.png", dpi=200)
    plt.close()

    # Macro F1
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train"]["f1_macro"], label="Train macro-F1")
    plt.plot(epochs, history["val"]["f1_macro"], label="Val macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Centralized training/validation macro-F1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "centralized_f1_curves.png", dpi=200)
    plt.close()


def plot_federated_val_loss(history, metrics_df, out_dir: Path):
    # From history.json
    rounds_hist = [r["round"] for r in history["rounds"]]
    val_loss_hist = [r["val_loss"] for r in history["rounds"]]

    # From metrics.csv (round,val_loss)
    rounds_metrics = metrics_df["round"].tolist()
    val_loss_metrics = metrics_df["val_loss"].tolist()

    plt.figure(figsize=(6, 4))
    plt.plot(rounds_hist, val_loss_hist, "o-", label="Val loss")
    plt.plot(rounds_metrics, val_loss_metrics, "s--", label="Val loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Federated validation loss per round")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "federated_val_loss_rounds.png", dpi=200)
    plt.close()


def plot_per_client_bar(eval_centralized, eval_federated, out_dir: Path):
    clients = eval_centralized["clients"]

    metrics = ["loss", "accuracy", "f1_macro"]
    metric_labels = {
        "loss": "Loss",
        "accuracy": "Accuracy",
        "f1_macro": "Macro-F1",
    }

    for metric in metrics:
        centralized_vals = [
            eval_centralized["per_client_test"][c][metric] for c in clients
        ]
        federated_vals = [
            eval_federated["per_client_test"][c][metric] for c in clients
        ]

        x = range(len(clients))
        width = 0.35

        plt.figure(figsize=(6, 4))
        plt.bar(
            [i - width / 2 for i in x],
            centralized_vals,
            width=width,
            label="Centralized",
        )
        plt.bar(
            [i + width / 2 for i in x],
            federated_vals,
            width=width,
            label="Federated",
        )

        plt.xticks(list(x), clients, rotation=0)
        plt.ylabel(metric_labels[metric])
        plt.title(f"{metric_labels[metric]} per client: centralized vs federated")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            out_dir / f"per_client_{metric}_centralized_vs_federated.png", dpi=200
        )
        plt.close()


def plot_global_bar(eval_centralized, eval_federated, out_dir: Path):
    metrics = ["loss", "accuracy", "f1_macro"]
    metric_labels = {
        "loss": "Loss",
        "accuracy": "Accuracy",
        "f1_macro": "Macro-F1",
    }

    x = range(len(metrics))
    width = 0.35

    centralized_vals = [eval_centralized["global_test"][m] for m in metrics]
    federated_vals = [eval_federated["global_test"][m] for m in metrics]

    plt.figure(figsize=(6, 4))
    plt.bar(
        [i - width / 2 for i in x],
        centralized_vals,
        width=width,
        label="Centralized",
    )
    plt.bar(
        [i + width / 2 for i in x],
        federated_vals,
        width=width,
        label="Federated",
    )

    plt.xticks(list(x), [metric_labels[m] for m in metrics])
    plt.ylabel("Value")
    plt.title("Global test metrics: centralized vs federated")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "global_metrics_centralized_vs_federated.png", dpi=200)
    plt.close()


# ---------- NEW: per-class metrics & confusion matrices ----------


def load_predictions(pred_path: Path):
    if not pred_path.exists():
        return None, None
    data = np.load(pred_path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    return y_true, y_pred


def plot_confusion(y_true, y_pred, out_path: Path, title: str, class_names=None):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(
        cmap="Blues",
        values_format=".2f",
        ax=ax,
        colorbar=True,
        xticks_rotation="vertical",
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_class_metrics(
    y_true_c, y_pred_c, y_true_f, y_pred_f, out_dir: Path, class_names=None
):
    # precision, recall, f1 per class
    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
        y_true_c, y_pred_c, average=None, zero_division=0
    )
    prec_f, rec_f, f1_f, _ = precision_recall_fscore_support(
        y_true_f, y_pred_f, average=None, zero_division=0
    )

    num_classes = len(f1_c)
    x = np.arange(num_classes)
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    def _bar(metric_c, metric_f, name):
        width = 0.4
        plt.figure(figsize=(10, 4))
        plt.bar(x - width / 2, metric_c, width=width, label="Centralized")
        plt.bar(x + width / 2, metric_f, width=width, label="Federated")
        plt.xticks(x, class_names, rotation=90)
        plt.ylabel(name)
        plt.title(f"Per-class {name}: centralized vs federated")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"per_class_{name.lower()}_centralized_vs_federated.png", dpi=200)
        plt.close()

    _bar(prec_c, prec_f, "Precision")
    _bar(rec_c, rec_f, "Recall")
    _bar(f1_c, f1_f, "F1")


def main():
    project_root = Path(__file__).resolve().parents[1]

    centralized_dir = project_root / "results" / "centralized"
    federated_dir = project_root / "results" / "federated"
    eval_dir = project_root / "results" / "evaluation"
    ensure_dir(eval_dir)

    # Paths (adapt if your filenames differ)
    centralized_history = centralized_dir / "history.json"
    centralized_metrics = centralized_dir / "metrics.csv"
    centralized_eval_summary = centralized_dir / "eval_summary.json"

    federated_history = federated_dir / "history.json"
    federated_metrics = federated_dir / "metrics.csv"
    federated_eval_summary = federated_dir / "eval_summary.json"

    # Optional prediction files
    centralized_pred_path = centralized_dir / "predictions.npz"
    federated_pred_path = federated_dir / "predictions.npz"

    # Load data
    cent_history, cent_metrics_df, cent_eval = load_centralized(
        centralized_history, centralized_metrics, centralized_eval_summary
    )
    fed_history, fed_metrics_df, fed_eval = load_federated(
        federated_history, federated_metrics, federated_eval_summary
    )

    # Basic plots
    plot_centralized_training_curves(cent_history, eval_dir)
    plot_federated_val_loss(fed_history, fed_metrics_df, eval_dir)
    plot_per_client_bar(cent_eval, fed_eval, eval_dir)
    plot_global_bar(cent_eval, fed_eval, eval_dir)

    # Extended evaluation: per-class + confusion matrices if predictions exist
    y_true_c, y_pred_c = load_predictions(centralized_pred_path)
    y_true_f, y_pred_f = load_predictions(federated_pred_path)

    if y_true_c is not None and y_true_f is not None:
        # Confusion matrices
        plot_confusion(
            y_true_c,
            y_pred_c,
            eval_dir / "confusion_centralized.png",
            title="Confusion matrix (centralized)",
        )
        plot_confusion(
            y_true_f,
            y_pred_f,
            eval_dir / "confusion_federated.png",
            title="Confusion matrix (federated)",
        )

        # Per-class bar plots
        plot_per_class_metrics(y_true_c, y_pred_c, y_true_f, y_pred_f, eval_dir)

    print(f"Saved evaluation plots to: {eval_dir}")


if __name__ == "__main__":
    main()
