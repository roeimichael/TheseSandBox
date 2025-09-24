from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, List, Iterable, Any
from math import log2


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
    """Legacy compact summary (hard-pred based)."""
    disagreements = pairwise_disagreement(predictions)
    q_stats = q_statistic(predictions)
    correlations = correlation_diversity(predictions)
    def convert_tuple_keys(d):
        return {f"{k[0]}_vs_{k[1]}": v for k, v in d.items()}
    return {
        "avg_disagreement": float(np.mean(list(disagreements.values()))) if disagreements else float('nan'),
        "avg_q_statistic": float(np.mean(list(q_stats.values()))) if q_stats else float('nan'),
        "avg_correlation": float(np.mean(list(correlations.values()))) if correlations else float('nan'),
        "pairwise_disagreements": convert_tuple_keys(disagreements),
        "pairwise_q_stats": convert_tuple_keys(q_stats),
        "pairwise_correlations": convert_tuple_keys(correlations)
    }

# ---- Extended Diversity / Correlation Framework ---------------------------------

def _positive_class_probs(member_probas: Iterable[np.ndarray]) -> np.ndarray:
    probs = [positive_class_probability(p) for p in member_probas]
    return np.vstack(probs)  # shape (n_members, n_samples)


def _hard_predictions_from_proba(member_probas: Iterable[np.ndarray]) -> np.ndarray:
    preds = [np.argmax(p, axis=1) for p in member_probas]
    return np.vstack(preds)  # (n_members, n_samples)


def pairwise_pearson_probability_corr(member_probas: List[np.ndarray]) -> Dict[Tuple[int, int], float]:
    P = _positive_class_probs(member_probas)  # (m, n)
    if P.shape[1] < 2:
        return {}
    C = np.corrcoef(P)
    out: Dict[Tuple[int, int], float] = {}
    m = C.shape[0]
    for i in range(m):
        for j in range(i+1, m):
            val = C[i, j]
            if np.isnan(val):
                val = 0.0
            out[(i, j)] = float(val)
    return out


def pairwise_spearman_probability_corr(member_probas: List[np.ndarray]) -> Dict[Tuple[int, int], float]:
    from scipy.stats import spearmanr  # optional dependency; if missing, skip
    P = _positive_class_probs(member_probas)  # (m, n)
    out: Dict[Tuple[int, int], float] = {}
    m = P.shape[0]
    for i in range(m):
        for j in range(i+1, m):
            try:
                r, _ = spearmanr(P[i], P[j])
            except Exception:
                r = np.nan
            if np.isnan(r):
                r = 0.0
            out[(i, j)] = float(r)
    return out


def _safe_kl(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def pairwise_symmetric_kl(member_probas: List[np.ndarray]) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    m = len(member_probas)
    for i in range(m):
        Pi = member_probas[i]
        for j in range(i+1, m):
            Pj = member_probas[j]
            # average over samples
            kl_ij = np.mean([_safe_kl(Pi[k], Pj[k]) for k in range(Pi.shape[0])])
            kl_ji = np.mean([_safe_kl(Pj[k], Pi[k]) for k in range(Pj.shape[0])])
            out[(i, j)] = float(0.5 * (kl_ij + kl_ji))
    return out


def pairwise_js_divergence(member_probas: List[np.ndarray]) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    eps = 1e-12
    m = len(member_probas)
    for i in range(m):
        Pi = member_probas[i]
        for j in range(i+1, m):
            Pj = member_probas[j]
            M = 0.5 * (Pi + Pj)
            M = np.clip(M, eps, 1 - eps)
            Pi_c = np.clip(Pi, eps, 1 - eps)
            Pj_c = np.clip(Pj, eps, 1 - eps)
            js_samples = 0.5 * (np.sum(Pi_c * (np.log(Pi_c) - np.log(M)), axis=1) +
                                 np.sum(Pj_c * (np.log(Pj_c) - np.log(M)), axis=1))
            out[(i, j)] = float(np.mean(js_samples))
    return out


def error_correlation(member_preds: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[Tuple[str, str], float]:
    names = list(member_preds.keys())
    out: Dict[Tuple[str, str], float] = {}
    for i, ni in enumerate(names):
        ei = (member_preds[ni] != y_true).astype(float)
        for j in range(i+1, len(names)):
            nj = names[j]
            ej = (member_preds[nj] != y_true).astype(float)
            if ei.std() < 1e-9 or ej.std() < 1e-9:
                corr = 0.0
            else:
                corr = float(np.corrcoef(ei, ej)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            out[(ni, nj)] = corr
    return out


def _avg_dict_values(d: Dict[Tuple[Any, Any], float]) -> float:
    return float(np.mean(list(d.values()))) if d else float('nan')


def vote_entropy(member_preds: Dict[str, np.ndarray]) -> float:
    preds = np.vstack(list(member_preds.values()))  # (m, n)
    m, n = preds.shape
    ent = 0.0
    for j in range(n):
        counts, _ = np.histogram(preds[:, j], bins=[-0.5, 0.5, 1.5])  # binary assumption
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent += -np.sum(probs * np.log2(probs + 1e-12))
    return float(ent / n)


def oracle_accuracy(member_preds: Dict[str, np.ndarray], y_true: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    preds = np.vstack(list(member_preds.values()))  # (m, n)
    correct_any = np.any(preds == y_true[None, :], axis=0)
    return float(np.mean(correct_any))


def coverage(member_preds: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    preds = np.vstack(list(member_preds.values()))
    n = preds.shape[1]
    correct_matrix = (preds == y_true)
    at_least_one = np.any(correct_matrix, axis=0).mean()
    all_correct = np.all(correct_matrix, axis=0).mean()
    none_correct = np.all(~correct_matrix, axis=0).mean()
    return {
        'coverage_at_least_one_correct': float(at_least_one),
        'coverage_all_correct': float(all_correct),
        'coverage_none_correct': float(none_correct),
    }


def build_extended_diversity(member_probas: List[np.ndarray], y_true: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    # Hard predictions map name -> array
    member_preds = {f"member_{i+1:03d}": np.argmax(p, axis=1) for i, p in enumerate(member_probas)}
    names = list(member_preds.keys())
    # Pairwise metrics
    pearson_proba = pairwise_pearson_probability_corr(member_probas)
    try:
        spearman_proba = pairwise_spearman_probability_corr(member_probas)
    except Exception:
        spearman_proba = {}
    sym_kl = pairwise_symmetric_kl(member_probas)
    js_div = pairwise_js_divergence(member_probas)
    hard_disagreement = pairwise_disagreement(member_preds)
    hard_q = q_statistic(member_preds)
    hard_corr = correlation_diversity(member_preds)
    err_corr = error_correlation(member_preds, y_true)

    # Aggregate
    summary = {
        'avg_pearson_proba_corr': _avg_dict_values(pearson_proba),
        'avg_spearman_proba_corr': _avg_dict_values(spearman_proba),
        'avg_sym_kl': _avg_dict_values(sym_kl),
        'avg_js_div': _avg_dict_values(js_div),
        'avg_disagreement': _avg_dict_values(hard_disagreement),
        'avg_q_statistic': _avg_dict_values(hard_q),
        'avg_hard_prediction_corr': _avg_dict_values(hard_corr),
        'avg_error_corr': _avg_dict_values(err_corr),
        'vote_entropy': vote_entropy(member_preds),
        'oracle_accuracy': oracle_accuracy(member_preds, y_true),
    }
    summary.update(coverage(member_preds, y_true))

    # Convert pairwise to serializable string keys
    def conv(d: Dict[Tuple[Any, Any], float], key_fmt: List[str] | None = None, is_name: bool = False):
        out = {}
        for (i, j), v in d.items():
            if isinstance(i, str):
                a, b = i, j
            else:
                a, b = (f"member_{i+1:03d}", f"member_{j+1:03d}")
            out[f"{a}_vs_{b}"] = v
        return out

    pairwise_section = {
        'pearson_proba_corr': conv(pearson_proba),
        'spearman_proba_corr': conv(spearman_proba),
        'symmetric_kl': conv(sym_kl),
        'js_divergence': conv(js_div),
        'disagreement': conv(hard_disagreement),
        'q_statistic': conv(hard_q),
        'hard_prediction_corr': conv(hard_corr),
        'error_corr': conv(err_corr),
    }
    return { 'summary': summary, 'pairwise': pairwise_section }


def write_full_diversity_artifacts(val_member_probas: List[np.ndarray], test_member_probas: List[np.ndarray],
                                   y_val: np.ndarray, y_test: np.ndarray,
                                   out_dir, summary_dir,
                                   write_pairwise: bool = False,
                                   write_extended_json: bool = True,
                                   write_summary_csv: bool = True) -> Dict[str, Any]:
    import json, pandas as pd
    val_report = build_extended_diversity(val_member_probas, y_val if hasattr(y_val, 'values') else y_val)
    test_report = build_extended_diversity(test_member_probas, y_test if hasattr(y_test, 'values') else y_test)
    full = {'val': val_report, 'test': test_report}
    if write_extended_json:
        with open(out_dir / 'diversity_extended.json', 'w') as f:
            json.dump(full, f, indent=2)

    # Optional pairwise CSVs (disabled by default to reduce clutter)
    if write_pairwise:
        def _write_pairwise(name: str, mapping: Dict[str, float]):
            rows = []
            for k, v in mapping.items():
                a, b = k.split('_vs_') if '_vs_' in k else (k, '')
                rows.append({'model_a': a, 'model_b': b, name: v})
            pd.DataFrame(rows).to_csv(summary_dir / f'pairwise_{name}.csv', index=False)
        for metric_name, mapping in val_report['pairwise'].items():
            _write_pairwise(metric_name, mapping)

    # Summary CSV (both splits)
    if write_summary_csv:
        rows = []
        for split, rep in full.items():
            row = {'split': split}
            row.update(rep['summary'])
            rows.append(row)
        pd.DataFrame(rows).to_csv(summary_dir / 'correlation_summary.csv', index=False)

    # Edge list generation skipped unless pairwise requested
    if write_pairwise and write_summary_csv:
        pearson_map = val_report['pairwise']['pearson_proba_corr']
        rows = []
        for edge, val in pearson_map.items():
            a, b = edge.split('_vs_')
            rows.append({'source': a, 'target': b, 'pearson_proba_corr': val, 'distance': 1 - val})
        pd.DataFrame(rows).to_csv(summary_dir / 'graph_edges_pearson.csv', index=False)
    return full



