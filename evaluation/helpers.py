"""
evaluation/helpers.py
Small helpers to compute standard metrics + ECE in one go.
"""
from __future__ import annotations
import numpy as np
from extra_util.np_utils import ece
from evaluation.metrics import compute_basic_metrics


def metrics_block(y, proba):
    """Compute metrics and add ECE.
    Returns a dict with accuracy, precision, recall, f1, roc_auc, log_loss, ece.
    """
    pred = np.argmax(proba, axis=1)
    m = compute_basic_metrics(y, proba, pred)
    m["ece"] = ece(y, proba)
    return m


def conditional_prediction_correlation(preds_a: np.ndarray, preds_b: np.ndarray, y_true: np.ndarray) -> float:
    """Average Pearson correlation of predictions conditioned on class labels.
    preds_a/preds_b: 1D arrays of predicted labels (or probabilities thresholded)
    y_true: 1D array of true labels (binary assumed here)
    """
    corrs = []
    for label in np.unique(y_true):
        idx = (y_true == label)
        if idx.sum() > 1:
            corrs.append(np.corrcoef(preds_a[idx], preds_b[idx])[0, 1])
    return float(np.nanmean(corrs)) if corrs else 0.0
