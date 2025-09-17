"""
extra_util/np_utils.py
Lightweight numpy/pandas helpers used across the project.
"""
from __future__ import annotations
import numpy as np
from evaluation.metrics import expected_calibration_error


def to_np1d(y):
    """Return a 1D numpy array for labels/targets."""
    return y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)


def to_np2d(X):
    """Return a 2D numpy array for feature matrices."""
    return X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)


def ece(y_true, proba) -> float:
    """Compute ECE using the shared metric function and return float."""
    y_np = to_np1d(y_true)
    return float(expected_calibration_error(y_np, proba))
