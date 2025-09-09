from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_metrics_bars(metrics_csv_path: str | Path, output_dir: str | Path) -> None:
    metrics_csv_path = Path(metrics_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv_path, index_col=0)
    
    # Check if this is a baseline CSV (with split column) or ensemble CSV (without split)
    if "split" in df.columns:
        # Baseline format: separate by split
        val_df = df[df["split"] == "val"].drop(columns=["split"])  # rows=models
        test_df = df[df["split"] == "test"].drop(columns=["split"])  # rows=models
        
        # For each metric, bar plot across models for both splits
        metrics = [c for c in val_df.columns if c not in ("split",)]
        for metric in metrics:
            plt.figure(figsize=(8, 5))
            # Combine val and test side-by-side
            plot_df = pd.DataFrame({
                "val": val_df[metric],
                "test": test_df[metric].reindex(val_df.index),
            })
            plot_df.plot(kind="bar")
            plt.title(f"{metric} by model")
            plt.ylabel(metric)
            plt.xlabel("model")
            plt.tight_layout()
            plt.legend(title="split")
            plt.savefig(output_dir / f"metrics_bar_{metric}.png", dpi=150)
            plt.close()
    else:
        # Ensemble format: single split, all models including ensemble
        # For each metric, bar plot across all models
        metrics = [c for c in df.columns if c not in ("split",)]
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            # Highlight ensemble row
            colors = ['tab:blue' if name != 'ENSEMBLE' else 'tab:red' for name in df.index]
            bars = plt.bar(range(len(df.index)), df[metric], color=colors, alpha=0.7)
            
            # Make ensemble bar stand out
            if 'ENSEMBLE' in df.index:
                ensemble_idx = list(df.index).index('ENSEMBLE')
                bars[ensemble_idx].set_alpha(1.0)
                bars[ensemble_idx].set_edgecolor('black')
                bars[ensemble_idx].set_linewidth(2)
            
            plt.title(f"{metric} by model (Ensemble highlighted)")
            plt.ylabel(metric)
            plt.xlabel("model")
            plt.xticks(range(len(df.index)), df.index, rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / f"metrics_bar_{metric}.png", dpi=150)
            plt.close()


def plot_uncertainty_and_calibration(
    y_true: np.ndarray,
    model_to_proba: Dict[str, np.ndarray],
    output_dir: str | Path,
    n_bins: int = 15,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from evaluation.metrics import (
        compute_uncertainty, compute_entropy, positive_class_probability, expected_calibration_error
    )

    for name, proba in model_to_proba.items():
        # Compute quantities
        pos_prob = positive_class_probability(proba)
        uncertainty = compute_uncertainty(proba)
        entropy = compute_entropy(proba)

        # Probability vs Uncertainty scatter
        plt.figure(figsize=(6, 5))
        plt.scatter(pos_prob, uncertainty, alpha=0.5, s=12)
        plt.xlabel("P(y=1)")
        plt.ylabel("Uncertainty (1 - confidence)")
        plt.title(f"{name}: probability vs uncertainty")
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_prob_vs_uncertainty.png", dpi=150)
        plt.close()

        # Entropy histogram
        plt.figure(figsize=(6, 5))
        plt.hist(entropy, bins=30, alpha=0.7, color="tab:blue")
        plt.xlabel("Entropy (bits)")
        plt.ylabel("Count")
        plt.title(f"{name}: entropy histogram")
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_entropy_hist.png", dpi=150)
        plt.close()

        # Reliability diagram and ECE (bar-style with color-coded miscalibration)
        prob_true, prob_pred = calibration_curve(y_true, pos_prob, n_bins=n_bins, strategy="uniform")
        ece = expected_calibration_error(y_true, proba, n_bins=n_bins)

        plt.figure(figsize=(7, 5))
        # Diagonal
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")

        # Bars: centered at prob_pred, width = 1/n_bins
        bar_width = 1.0 / n_bins
        for pt, pp in zip(prob_true, prob_pred):
            left = pp - bar_width / 2
            color = "tab:red" if pt > pp else "tab:blue"
            plt.bar(left, pt, width=bar_width*0.95, align="edge", color=color, alpha=0.6, edgecolor="black", linewidth=0.5)

        # Overlay points for clarity
        plt.scatter(prob_pred, prob_true, s=18, color="black", zorder=3)

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Mean predicted probability (per bin)")
        plt.ylabel("Fraction of positives (per bin)")
        plt.title(f"Reliability diagram ({name})  ECE={ece:.4f}")
        # Custom legend handles
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor="tab:blue", edgecolor="black", label="under-confident (below y=x)"),
            Patch(facecolor="tab:red", edgecolor="black", label="over-confident (above y=x)"),
        ]
        plt.legend(handles=handles, loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_reliability.png", dpi=150)
        plt.close()


