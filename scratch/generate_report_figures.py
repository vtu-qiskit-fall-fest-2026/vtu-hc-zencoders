"""Publication-Quality Figures Generation for Q-Nav Step 3 Quantum Accuracy Optimization.

Generates:
1. outputs/figures/quantum_training_curve.png
2. outputs/figures/quantum_validation_accuracy.png
3. outputs/figures/quantum_confusion_matrix.png
4. outputs/figures/quantum_class_distribution.png
5. outputs/figures/architecture_comparison.png
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = Path("outputs/metrics")

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

ACTIONS = [
    "UP", "UP_RIGHT", "RIGHT", "DOWN_RIGHT",
    "DOWN", "DOWN_LEFT", "LEFT", "UP_LEFT", "STAY"
]


def plot_training_curve(history: dict, save_path: Path):
    """Plot Loss and Accuracy curves across training epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "b-o", linewidth=2.2, label="Train Loss", markersize=5)
    ax1.plot(epochs, history["val_loss"], "r--s", linewidth=2.2, label="Validation Loss", markersize=5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Accuracy subplot
    train_acc = [a * 100.0 if a <= 1.0 else a for a in history["train_accuracy"]]
    val_acc = [a * 100.0 if a <= 1.0 else a for a in history["val_accuracy"]]
    ax2.plot(epochs, train_acc, "b-o", linewidth=2.2, label="Train Accuracy", markersize=5)
    ax2.plot(epochs, val_acc, "r--s", linewidth=2.2, label="Validation Accuracy", markersize=5)
    ax2.axhline(11.11, color="gray", linestyle=":", label="Random Chance (11.11%)")
    ax2.axhline(90.0, color="green", linestyle="--", alpha=0.8, label="Target Goal (90%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curve to {save_path}")


def plot_validation_accuracy_comparison(models_data: dict, save_path: Path):
    """Plot benchmark comparison of validation accuracy across baseline, models, and target."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(models_data.keys())
    accs = [models_data[k]["accuracy"] for k in names]
    colors = [models_data[k].get("color", "#1f77b4") for k in names]

    bars = ax.bar(names, accs, color=colors, width=0.55, edgecolor="black", linewidth=1.1, alpha=0.9)
    ax.axhline(90.0, color="crimson", linestyle="--", linewidth=2, label="Theoretical Target (90.0%)")
    ax.axhline(11.11, color="black", linestyle=":", linewidth=1.5, label="Random Guess (11.11%)")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("VQC Classifier Generalization on Unseen Flight Trajectories")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved validation accuracy comparison to {save_path}")


def plot_confusion_matrix(cm_data: np.ndarray, action_names: list, save_path: Path):
    """Plot normalized confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(9, 8))
    # Class-normalized confusion matrix
    cm_norm = cm_data.astype(float) / (cm_data.sum(axis=1, keepdims=True) + 1e-8)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(cm_data.shape[1]),
           yticks=np.arange(cm_data.shape[0]),
           xticklabels=action_names, yticklabels=action_names,
           title="Quantum Classifier Confusion Matrix (Normalized)",
           ylabel="True Teacher Action",
           xlabel="Predicted Model Action")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.0
    for i in range(cm_data.shape[0]):
        for j in range(cm_data.shape[1]):
            val_pct = f"{cm_norm[i, j]*100:.1f}%\n({cm_data[i, j]})" if cm_data[i, j] > 0 else "0%"
            ax.text(j, i, val_pct,
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=8)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_class_distribution(dist_data: dict, save_path: Path):
    """Plot dataset action distribution showing balanced distribution across directions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = [a for a in ACTIONS if a in dist_data]
    total = sum(dist_data.values())
    pcts = [(dist_data[a] / total) * 100.0 for a in names]

    bars = ax.bar(names, pcts, color="#2ca02c", width=0.55, edgecolor="black", linewidth=1.1, alpha=0.85)

    # Highlight balanced range (8% to 14%)
    ax.axhspan(8.0, 14.0, color="orange", alpha=0.18, label="Target Balanced Range (8% - 14%)")
    ax.axhline(11.11, color="blue", linestyle=":", linewidth=1.5, label="Perfect Uniformity (11.11%)")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Representation in Dataset (%)")
    ax.set_title(f"Synthetic Dataset Class Distribution (N={total:,} samples across 10 Topologies)")
    ax.set_ylim(0, max(pcts) + 5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved class distribution to {save_path}")


def plot_architecture_comparison(arch_data: list, save_path: Path):
    """Plot 4-panel comparison of Models A through E."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10))

    names = [m["model_name"] for m in arch_data]
    val_accs = [m["val_accuracy"] for m in arch_data]
    f1s = [m["macro_f1"] for m in arch_data]
    params = [m["total_params"] for m in arch_data]
    inf_times = [m["inference_time_ms"] for m in arch_data]

    # Panel 1: Val Acc
    bars1 = ax1.bar(names, val_accs, color="#1f77b4", width=0.5, edgecolor="black")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("A. Generalization Accuracy")
    ax1.set_ylim(0, 55)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars1:
        ax1.annotate(f"{b.get_height():.1f}%", (b.get_x()+b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")

    # Panel 2: Macro F1
    bars2 = ax2.bar(names, f1s, color="#ff7f0e", width=0.5, edgecolor="black")
    ax2.set_ylabel("Macro F1 Score")
    ax2.set_title("B. Macro F1 Score (Multi-Class Balance)")
    ax2.set_ylim(0, 0.45)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars2:
        ax2.annotate(f"{b.get_height():.3f}", (b.get_x()+b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")

    # Panel 3: Total Parameters
    bars3 = ax3.bar(names, params, color="#2ca02c", width=0.5, edgecolor="black")
    ax3.set_ylabel("Parameter Count")
    ax3.set_title("C. Model Capacity (Weights + Biases)")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars3:
        ax3.annotate(f"{int(b.get_height())}", (b.get_x()+b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")

    # Panel 4: Latency
    bars4 = ax4.bar(names, inf_times, color="#d62728", width=0.5, edgecolor="black")
    ax4.set_ylabel("Inference Time (ms / step)")
    ax4.set_title("D. Quantum Inference Execution Latency")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars4:
        ax4.annotate(f"{b.get_height():.2f} ms", (b.get_x()+b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved architecture comparison to {save_path}")
