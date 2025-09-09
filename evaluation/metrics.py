from __future__ import annotations
import numpy as np
from typing import Tuple
from typing import Dict


def positive_class_probability(proba: np.ndarray) -> np.ndarray:
    """
    Returns probability of the positive class for binary classification.
    If more than 2 classes, falls back to the max probability across classes.
    """
    if proba.ndim != 2:
        raise ValueError("proba must be 2D: (n_samples, n_classes)")
    if proba.shape[1] == 2:
        return proba[:, 1]
    return proba.max(axis=1)


def compute_confidence(proba: np.ndarray) -> np.ndarray:
    return proba.max(axis=1)


def compute_uncertainty(proba: np.ndarray) -> np.ndarray:
    return 1.0 - compute_confidence(proba)


def compute_entropy(proba: np.ndarray, base: float = 2.0) -> np.ndarray:
    eps = 1e-12
    p = np.clip(proba, eps, 1.0)
    log_p = np.log(p)
    if base != np.e:
        log_p = log_p / np.log(base)
    ent = -np.sum(p * log_p, axis=1)
    return ent


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE computed over bins of predicted positive-class probability.
    ECE = sum_b (n_b/N) * |acc_b - conf_b|
    """
    y_true = np.asarray(y_true)
    y_prob = positive_class_probability(proba)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1  # 0..n_bins-1
    ece = 0.0
    N = len(y_prob)

    for b in range(n_bins):
        in_bin = bin_indices == b
        n_b = np.sum(in_bin)
        if n_b == 0:
            continue
        conf_b = np.mean(y_prob[in_bin])
        acc_b = np.mean(y_true[in_bin] == 1)
        ece += (n_b / N) * abs(acc_b - conf_b)
    return float(ece)


def compute_basic_metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> Dict:
    """Compute basic classification metrics with numerical safeguards."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
    # Ensure numerical stability
    proba = np.clip(proba, 1e-7, 1 - 1e-7)
    pos_proba = positive_class_probability(proba)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, pos_proba) if len(np.unique(y_true)) == 2 else float("nan"),
        "log_loss": log_loss(y_true, proba),
    }


# Diversity metrics for ensemble analysis
def pairwise_disagreement(predictions: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise disagreement between models.
    
    Args:
        predictions: Dict mapping model names to prediction arrays
        
    Returns:
        Dict mapping (model1, model2) tuples to disagreement rates
    """
    model_names = list(predictions.keys())
    disagreements = {}
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names[i+1:], i+1):
            pred1 = predictions[name1]
            pred2 = predictions[name2]
            disagreement = np.mean(pred1 != pred2)
            disagreements[(name1, name2)] = disagreement
    
    return disagreements


def q_statistic(predictions: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
    """
    Compute Q-statistic between pairs of models.
    Q = (N11*N00 - N01*N10) / (N11*N00 + N01*N10)
    where Nxy = number of samples where model1 predicts x and model2 predicts y
    """
    model_names = list(predictions.keys())
    q_stats = {}
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names[i+1:], i+1):
            pred1 = predictions[name1]
            pred2 = predictions[name2]
            
            # Count agreements/disagreements
            n11 = np.sum((pred1 == 1) & (pred2 == 1))
            n00 = np.sum((pred1 == 0) & (pred2 == 0))
            n01 = np.sum((pred1 == 0) & (pred2 == 1))
            n10 = np.sum((pred1 == 1) & (pred2 == 0))
            
            # Q-statistic
            numerator = n11 * n00 - n01 * n10
            denominator = n11 * n00 + n01 * n10
            
            if denominator == 0:
                q_stats[(name1, name2)] = 0.0
            else:
                q_stats[(name1, name2)] = numerator / denominator
    
    return q_stats


def correlation_diversity(predictions: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
    """
    Compute correlation-based diversity between pairs of models.
    Higher correlation = lower diversity.
    """
    model_names = list(predictions.keys())
    correlations = {}
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names[i+1:], i+1):
            pred1 = predictions[name1].astype(float)
            pred2 = predictions[name2].astype(float)
            
            # Pearson correlation
            corr = np.corrcoef(pred1, pred2)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            correlations[(name1, name2)] = corr
    
    return correlations


def ensemble_diversity_summary(predictions: Dict[str, np.ndarray]) -> Dict:
    """
    Compute comprehensive diversity summary for an ensemble.
    
    Returns:
        Dict with average disagreement, Q-statistic, and correlation diversity
    """
    disagreements = pairwise_disagreement(predictions)
    q_stats = q_statistic(predictions)
    correlations = correlation_diversity(predictions)
    
    # Convert tuple keys to strings for JSON serialization
    def convert_tuple_keys(d):
        return {f"{k[0]}_vs_{k[1]}": v for k, v in d.items()}
    
    return {
        "avg_disagreement": np.mean(list(disagreements.values())),
        "avg_q_statistic": np.mean(list(q_stats.values())),
        "avg_correlation": np.mean(list(correlations.values())),
        "pairwise_disagreements": convert_tuple_keys(disagreements),
        "pairwise_q_stats": convert_tuple_keys(q_stats),
        "pairwise_correlations": convert_tuple_keys(correlations)
    }


