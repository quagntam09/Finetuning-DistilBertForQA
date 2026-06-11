#!/usr/bin/env python3
"""Generate clean training-history visualizations from saved QA run JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


COLORS = {
    "train": "#2563eb",
    "val": "#dc2626",
    "accuracy": "#16a34a",
    "precision": "#0891b2",
    "recall": "#7c3aed",
    "f1": "#f97316",
    "has_answer_f1": "#be123c",
    "no_answer_exact": "#475569",
    "threshold": "#9333ea",
    "best": "#111827",
    "grid": "#cbd5e1",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0f172a",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#e2e8f0",
            "lines.linewidth": 2.25,
            "lines.markersize": 6,
            "savefig.dpi": 220,
        }
    )


def epochs_for(values: list[float]) -> np.ndarray:
    return np.arange(1, len(values) + 1)


def prettify_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=COLORS["grid"], alpha=0.45, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def annotate_point(ax: plt.Axes, x: int, y: float, label: str, color: str = COLORS["best"]) -> None:
    ax.scatter([x], [y], s=72, color=color, zorder=5, edgecolor="white", linewidth=1.5)
    ax.annotate(
        f"{label}\n{x}: {y:.2f}",
        xy=(x, y),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=8.5,
        color=color,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#e2e8f0", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
    )


def best_epoch(values: list[float], mode: str) -> tuple[int, float]:
    arr = np.asarray(values, dtype=float)
    idx = int(np.argmin(arr) if mode == "min" else np.argmax(arr))
    return idx + 1, float(arr[idx])


def plot_dashboard(history: dict[str, list[float]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True)
    fig.suptitle("DistilBERT QA - Vietnamese fine-tuning from English checkpoint", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    if "train_loss" in history and "val_loss" in history:
        epochs = epochs_for(history["train_loss"])
        ax.plot(epochs, history["train_loss"], marker="o", label="Train loss", color=COLORS["train"])
        ax.plot(epochs, history["val_loss"], marker="s", label="Validation loss", color=COLORS["val"])
        x_best, y_best = best_epoch(history["val_loss"], "min")
        annotate_point(ax, x_best, y_best, "Best val loss")
        ax.set_title("Loss by epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="best")
        prettify_axis(ax)

    ax = axes[0, 1]
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    for key in metric_keys:
        if key in history:
            ax.plot(epochs_for(history[key]), history[key], marker="o", label=key.replace("_", " ").title(), color=COLORS[key])
    if "f1" in history:
        x_best, y_best = best_epoch(history["f1"], "max")
        annotate_point(ax, x_best, y_best, "Best F1")
    ax.set_title("Main QA scores")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(max(0, ax.get_ylim()[0] - 2), min(100, ax.get_ylim()[1] + 2))
    ax.legend(loc="lower right")
    prettify_axis(ax)

    ax = axes[1, 0]
    answer_keys = ["has_answer_f1", "no_answer_exact"]
    for key in answer_keys:
        if key in history:
            label = "Has-answer F1" if key == "has_answer_f1" else "No-answer exact"
            ax.plot(epochs_for(history[key]), history[key], marker="o", label=label, color=COLORS[key])
    ax.set_title("Answerability metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.legend(loc="best")
    prettify_axis(ax)

    ax = axes[1, 1]
    summary_metrics = {
        "Best val loss": ("val_loss", "min"),
        "Best accuracy": ("accuracy", "max"),
        "Best F1": ("f1", "max"),
        "Best has-answer F1": ("has_answer_f1", "max"),
        "Best no-answer exact": ("no_answer_exact", "max"),
    }
    labels: list[str] = []
    values: list[float] = []
    epoch_labels: list[str] = []
    for label, (key, mode) in summary_metrics.items():
        if key not in history:
            continue
        epoch, value = best_epoch(history[key], mode)
        labels.append(label)
        values.append(value)
        epoch_labels.append(f"epoch {epoch}")

    bar_colors = ["#ef4444", "#22c55e", "#f97316", "#e11d48", "#64748b"][: len(labels)]
    bars = ax.barh(labels, values, color=bar_colors, alpha=0.9)
    ax.invert_yaxis()
    ax.set_title("Best checkpoints by metric")
    ax.set_xlabel("Loss or score")
    for bar, value, epoch_label in zip(bars, values, epoch_labels):
        ax.text(
            bar.get_width() + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} ({epoch_label})",
            va="center",
            fontsize=9,
            color="#334155",
        )
    prettify_axis(ax)

    output_path = output_dir / "vi_from_en_training_dashboard.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metric_heatmap(history: dict[str, list[float]], output_dir: Path) -> Path:
    keys = ["accuracy", "precision", "recall", "f1", "has_answer_f1", "no_answer_exact"]
    keys = [key for key in keys if key in history]
    data = np.asarray([history[key] for key in keys], dtype=float)

    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    image = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_title("Metric heatmap by epoch")
    ax.set_xlabel("Epoch")
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([key.replace("_", " ").title() for key in keys])
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(np.arange(1, data.shape[1] + 1))

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", color="#0f172a", fontsize=8.5)

    cbar = fig.colorbar(image, ax=ax, shrink=0.88)
    cbar.set_label("Score (%)")

    output_path = output_dir / "vi_from_en_metric_heatmap.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pipeline_comparison(pipeline: dict[str, Any], output_dir: Path) -> Path | None:
    steps = pipeline.get("steps", [])
    if not steps:
        return None

    labels = [step.get("profile", f"step_{idx + 1}") for idx, step in enumerate(steps)]
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    for idx, metric in enumerate(metrics):
        values = []
        for step in steps:
            series = step.get("history", {}).get(metric, [])
            values.append(float(series[-1]) if series else np.nan)
        offset = (idx - (len(metrics) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, label=metric.title(), color=COLORS[metric], alpha=0.9)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}", ha="center", fontsize=8)

    ax.set_title("Pipeline final-epoch score comparison")
    ax.set_xlabel("Training profile")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 90)
    ax.legend(ncols=4, loc="upper center")
    prettify_axis(ax)

    output_path = output_dir / "pipeline_final_epoch_comparison.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/checkpoints_vi_from_en"),
        help="Directory containing training_history.json and pipeline_history.json.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Training history JSON. Defaults to run-dir/best_model/training_history.json if present.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated plots.")
    args = parser.parse_args()

    run_dir = args.run_dir
    history_path = args.history or run_dir / "best_model" / "training_history.json"
    if not history_path.exists():
        history_path = run_dir / "training_history.json"

    output_dir = args.output_dir or run_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    history = load_json(history_path)

    written: list[Path] = [
        plot_dashboard(history, output_dir),
        plot_metric_heatmap(history, output_dir),
    ]

    pipeline_path = run_dir / "pipeline_history.json"
    if pipeline_path.exists():
        pipeline_plot = plot_pipeline_comparison(load_json(pipeline_path), output_dir)
        if pipeline_plot is not None:
            written.append(pipeline_plot)

    print("Generated visualizations:")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
