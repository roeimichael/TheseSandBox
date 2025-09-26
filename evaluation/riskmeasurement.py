# riskmeasurement.py
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

# NOTE: we import via your evaluation.metrics module to stay consistent
from importlib import import_module as _import_module
positive_class_probability = _import_module('evaluation.metrics').positive_class_probability
expected_calibration_error = _import_module('evaluation.metrics').expected_calibration_error

# -------------------------
# helpers
# -------------------------

def _clip01(x, lo: float = 0.0, hi: float = 1.0):
    return float(np.clip(x, lo, hi))

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1: a = a.ravel()
    if b.ndim != 1: b = b.ravel()
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(c):
        return 0.0
    return float(c)

def _log_loss_from_pos(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

# -------------------------
# components (all ~[0,1])
# -------------------------

def component_quality(y_val, cand_proba: np.ndarray,
                      selection_metric: str = "f1",
                      alpha_auc: float = 0.0) -> float:
    """
    Intrinsic quality on validation using a simple, stable blend:
    q = ( (1 - alpha_auc) * Q_base + alpha_auc * AUC_norm ) * (1 - ECE)

    where:
      - Q_base is F1 or accuracy at 0.5 threshold
      - AUC_norm maps AUC in [0.5,1.0] -> [0,1]
      - ECE is in [0,1] and downweights overconfident miscalibrated models
    """
    y = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.asarray(y_val)
    pos = positive_class_probability(cand_proba)
    pred = (pos >= 0.5).astype(int)

    if selection_metric.lower() == "accuracy":
        q_base = float((pred == y).mean())
    else:
        from sklearn.metrics import f1_score
        q_base = float(f1_score(y, pred, zero_division=0))

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, pos))
    except Exception:
        auc = 0.5
    auc_norm = float(np.clip((auc - 0.5) / 0.5, 0.0, 1.0))

    ece = float(expected_calibration_error(y, cand_proba))  # 0..1
    q = ((1 - alpha_auc) * q_base + alpha_auc * auc_norm) * (1.0 - ece)
    return _clip01(q)

def component_diversity_vs_ensemble(cand_proba: np.ndarray,
                                    selected_val_probas: List[np.ndarray]) -> float:
    """
    Prediction-space orthogonality to current ensemble.
    d = sqrt( 1 - rho_pos^2 ), where rho_pos = max(0, corr(p_cand, p_ens))

    - If no selected models yet: return 1.0 (max diversity).
    - Negative correlation counts as fully diverse (rho_pos=0).
    """
    if not selected_val_probas:
        return 1.0

    p_c = positive_class_probability(cand_proba)
    p_ens = np.mean([positive_class_probability(sp) for sp in selected_val_probas], axis=0)

    rho = _safe_corr(p_c, p_ens)
    rho_pos = max(0.0, rho)  # ignore negative corr (helpful diversity)
    d = np.sqrt(max(0.0, 1.0 - rho_pos * rho_pos))
    return _clip01(d)

def component_uncertainty_alignment(cand_proba: np.ndarray,
                                    selected_val_probas: List[np.ndarray]) -> float:
    """
    How much the candidate *disagrees* with the current ensemble *exactly where*
    the ensemble is uncertain.

    u = mean( |p_c - p_ens| * w ),  where w = 4 * p_ens * (1 - p_ens)  (peaks at 0.5)

    - If no selected models yet: return 0.0 (undefined / not applicable).
    """
    if not selected_val_probas:
        return 0.0

    p_c = positive_class_probability(cand_proba)
    p_ens = np.mean([positive_class_probability(sp) for sp in selected_val_probas], axis=0)

    w = 4.0 * p_ens * (1.0 - p_ens)  # in [0,1], max at 0.5
    u = float(np.mean(np.abs(p_c - p_ens) * w))  # already in [0,1]
    return _clip01(u)

def component_marginal_gain(y_val,
                            cand_proba: np.ndarray,
                            selected_val_probas: List[np.ndarray],
                            cap: float = 0.10) -> float:
    """
    Approximate Shapley-like *marginal utility* on validation via log-loss delta:
      mg = clip( (LL_before - LL_after) / cap , 0, 1)

    where:
      - LL_before is ensemble log loss before adding the candidate
      - LL_after  is after including candidate by simple averaging
      - 'cap' stabilizes scaling; 0.05-0.15 is reasonable for most tabular sets
    If no selected models yet, return 0.0 (no 'ensemble' to improve).
    """
    if not selected_val_probas:
        return 0.0

    y = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.asarray(y_val)

    p_ens_before = np.mean([positive_class_probability(sp) for sp in selected_val_probas], axis=0)
    ll_before = _log_loss_from_pos(y, p_ens_before)

    p_c = positive_class_probability(cand_proba)
    p_ens_after = (p_ens_before * len(selected_val_probas) + p_c) / (len(selected_val_probas) + 1)
    ll_after = _log_loss_from_pos(y, p_ens_after)

    gain = max(0.0, ll_before - ll_after)
    mg = gain / max(cap, 1e-12)
    return _clip01(mg)

# -------------------------
# unified risk in [0,1]
# -------------------------

def compute_candidate_risk(y_val,
                           cand_proba: np.ndarray,
                           selected_val_probas: List[np.ndarray],
                           selection_metric: str = "f1",
                           alpha_auc: float = 0.0,
                           weights: Optional[Dict[str, float]] = None) -> float:
    """
    Cohesive, prediction-space-first risk/utility score in [0,1]:

      R = w_div * diversity
        + w_mg  * marginal_gain
        + w_unc * uncertainty_alignment
        + w_q   * quality

    Defaults emphasize prediction-space terms.
    """
    if weights is None:
        weights = {
            "diversity": 0.40,
            "marginal_gain": 0.30,
            "uncertainty_alignment": 0.15,
            "quality": 0.15,
        }

    q   = component_quality(y_val, cand_proba, selection_metric=selection_metric, alpha_auc=alpha_auc)
    d   = component_diversity_vs_ensemble(cand_proba, selected_val_probas)
    u   = component_uncertainty_alignment(cand_proba, selected_val_probas)
    mg  = component_marginal_gain(y_val, cand_proba, selected_val_probas, cap=0.10)

    R = (
        weights.get("diversity", 0.40) * d
      + weights.get("marginal_gain", 0.30) * mg
      + weights.get("uncertainty_alignment", 0.15) * u
      + weights.get("quality", 0.15) * q
    )
    return _clip01(R)
