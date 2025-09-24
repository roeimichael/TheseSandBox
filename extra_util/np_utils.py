"""
extra_util/np_utils.py
Lightweight numpy/pandas helpers used across the project.
"""
from __future__ import annotations
import numpy as np
try:
    from evaluation.metrics import expected_calibration_error
except ModuleNotFoundError:
    import sys, pathlib, importlib.util
    _NP_ROOT = pathlib.Path(__file__).resolve().parents[1]
    metrics_path = _NP_ROOT / 'evaluation' / 'metrics.py'
    if metrics_path.exists():
        spec = importlib.util.spec_from_file_location('evaluation.metrics', metrics_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[arg-type]
            expected_calibration_error = getattr(mod, 'expected_calibration_error')  # type: ignore


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
