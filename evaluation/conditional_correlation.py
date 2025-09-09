from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


def _pairwise_corr(matrix: np.ndarray) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise Pearson correlation for columns of a 2D array.
    matrix: shape (n_samples, n_models)
    Returns dict of ((i_name, j_name) -> corr) placeholder names must be provided by wrapper.
    This function is internal; use the public helpers below.
    """
    # np.corrcoef expects rows as variables if rowvar=True, so set rowvar=False
    if matrix.shape[0] < 2:
        return {}
    C = np.corrcoef(matrix, rowvar=False)
    # Return upper triangle pairwise excluding diagonal
    n = C.shape[0]
    corrs = {}
    for i in range(n):
        for j in range(i + 1, n):
            corrs[(i, j)] = float(0.0 if np.isnan(C[i, j]) else C[i, j])
    return corrs


def error_correlation(
    y_true: np.ndarray,
    member_preds: Dict[str, np.ndarray],
    condition: str | None = None
) -> Dict[str, float | Dict[str, float]]:
    """
    Compute pairwise Pearson correlation of error indicators for ensemble members.
    Errors are binary: e_m = 1 if pred_m != y_true else 0.

    Args:
      y_true: shape (n_samples,)
      member_preds: dict[name] -> predicted labels array shape (n_samples,)
      condition: None | 'y=0' | 'y=1'  → filter samples before computing correlations

    Returns:
      {
        'avg_correlation': float,
        'pairwise': { 'name_i_vs_name_j': corr, ... },
        'num_samples': int
      }
    """
    names = list(member_preds.keys())
    if condition == 'y=0':
        mask = (y_true == 0)
    elif condition == 'y=1':
        mask = (y_true == 1)
    else:
        mask = np.ones_like(y_true, dtype=bool)

    if mask.sum() < 2:
        return { 'avg_correlation': float('nan'), 'pairwise': {}, 'num_samples': int(mask.sum()) }

    # Build error matrix: columns = models, rows = samples
    errors = []
    for name in names:
        pred = member_preds[name]
        err = (pred[mask] != y_true[mask]).astype(float)
        errors.append(err)
    E = np.vstack(errors).T  # shape (n_masked, n_models)

    # Compute pairwise correlations
    raw = _pairwise_corr(E)

    # Map indices to name pairs
    pairwise: Dict[str, float] = {}
    n = len(names)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{names[i]}_vs_{names[j]}"
            pairwise[key] = raw.get((i, j), 0.0)
            k += 1

    avg_corr = float(np.mean(list(pairwise.values()))) if pairwise else float('nan')
    return { 'avg_correlation': avg_corr, 'pairwise': pairwise, 'num_samples': int(mask.sum()) }


def error_correlation_summary(y_true: np.ndarray, member_preds: Dict[str, np.ndarray]) -> Dict:
    """
    Convenience wrapper producing overall and class-conditional error correlations.
    """
    overall = error_correlation(y_true, member_preds, condition=None)
    y0 = error_correlation(y_true, member_preds, condition='y=0')
    y1 = error_correlation(y_true, member_preds, condition='y=1')
    return { 'overall': overall, 'y=0': y0, 'y=1': y1 }
