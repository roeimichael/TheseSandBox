from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable
import itertools
import json
import numpy as np
import pandas as pd
import yaml
import sys, pathlib
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ensure local packages are importable before any intra-repo imports
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# === Risk (new) ===
import evaluation.riskmeasurement as riskmeasurement

from extra_util.logger import get_logger
from extra_util.data_loader import DatasetConfig, load_dataset, preprocess_and_split
from models.baseline_models import build_mlp
from importlib import import_module as _import_module

positive_class_probability = _import_module('evaluation.metrics').positive_class_probability
compute_basic_metrics      = _import_module('evaluation.metrics').compute_basic_metrics
expected_calibration_error = _import_module('evaluation.metrics').expected_calibration_error
from evaluation.helpers import metrics_block
from evaluation.metrics import write_full_diversity_artifacts

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ensemble_mlp_classic.yaml"


# -------------------------
# IO / setup
# -------------------------

def load_cfg_and_data(config_path: Path | None):
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    logger = get_logger("mlp_classic_ensemble_simplified")

    dcfg = DatasetConfig(**cfg['dataset'])
    X, y = load_dataset(dcfg)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_and_split(
        X, y,
        test_size=dcfg.test_size,
        val_size=dcfg.val_size,
        random_state=dcfg.random_state,
        scale_numeric=dcfg.scale_numeric,
    )
    return cfg, logger, (X_train, X_val, X_test, y_train, y_val, y_test)


def ensure_dirs(base_dir: Path, results_family: str, exp_name: str) -> Tuple[Path, Path]:
    out_base = base_dir / results_family
    out_dir = out_base / exp_name
    if out_dir.exists():
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        out_dir = out_base / f"{exp_name}-{ts}"
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    return out_dir, out_dir / "summary"


# -------------------------
# Candidate generation & training
# -------------------------

def candidate_space_from_cfg(cfg: dict) -> Iterable[dict]:
    """
    Deterministic product of variations. You can shuffle externally if desired.
    """
    base = cfg['model'].get('base', {})
    var = cfg['model'].get('variations', {})
    seeds = var.get('seeds', [1,2,3,4,5,6,7,8,9,10])
    hls_list = var.get('hidden_layer_sizes', [base.get('hidden_layer_sizes', [100, 50])])
    alphas = var.get('alpha', [base.get('alpha', 1e-4)])
    activations = var.get('activation', [base.get('activation', 'relu')])
    solvers = var.get('solver', [base.get('solver', 'adam')])
    lri_list = var.get('learning_rate_init', [base.get('learning_rate_init', None)])  # None means use estimator default

    # Build product; if learning_rate_init is None, we'll simply not set it
    for i, (seed, hls, alpha, act, solv, lri) in enumerate(
        itertools.product(seeds, hls_list, alphas, activations, solvers, lri_list), 1
    ):
        params = dict(base)
        params.update({
            'hidden_layer_sizes': tuple(hls),
            'alpha': float(alpha),
            'random_state': int(seed),
            'activation': str(act),
            'solver': str(solv),
        })
        if lri is not None:
            params['learning_rate_init'] = float(lri)
        yield {'mlp': params, 'name': f"mlp_{i:03d}"}


def fit_candidates_with_cap(
    variants: List[dict],
    X_train, y_train, X_val, X_test,
    max_iter_cap: int | None
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Train models; if max_iter_cap is provided, override each MLP's max_iter.
    Returns names, val_probas, test_probas, and n_iter_ actually run (best guess from estimator).
    """
    names, val_probas, test_probas, iters = [], [], [], []
    for v in variants:
        v2 = {**v}
        mlp_cfg = {**v2['mlp']}
        if max_iter_cap is not None:
            mlp_cfg['max_iter'] = int(max_iter_cap)
        v2['mlp'] = mlp_cfg
        clf = build_mlp(v2)
        # Suppress non-critical sklearn convergence warnings during fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(X_train, y_train)
        val_probas.append(clf.predict_proba(X_val))
        test_probas.append(clf.predict_proba(X_test))
        names.append(v2['name'])
        n_run = getattr(clf, 'n_iter_', None)
        iters.append(int(n_run) if n_run is not None else int(mlp_cfg.get('max_iter', 0)))
    return names, val_probas, test_probas, iters


# -------------------------
# Small helpers (new)
# -------------------------

def _map_risk_weights(old: Dict[str, float]) -> Dict[str, float]:
    """
    Back-compat shim:
    - If new keys exist, return as-is.
    - Otherwise map old {quality, error_corr, proba_corr, disagreement, ece, log_loss}
      to the new prediction-space-first scheme with sensible defaults.
    """
    if any(k in old for k in ("diversity", "marginal_gain", "uncertainty_alignment")):
        return old
    return {
        "diversity": 0.40,
        "marginal_gain": 0.30,
        "uncertainty_alignment": 0.15,
        "quality": old.get("quality", 0.15),
    }


def compute_risk_score(
    y_val,
    cand_proba: np.ndarray,
    selected_val_probas: List[np.ndarray],
    selection_metric: str,
    alpha_auc: float,
    risk_weights: Dict[str, float],
):
    """Single place to compute candidate risk via riskmeasurement.
    Keeps the callsite consistent and avoids accidental double-computation.
    """
    return riskmeasurement.compute_candidate_risk(
        y_val=y_val,
        cand_proba=cand_proba,
        selected_val_probas=selected_val_probas,
        selection_metric=selection_metric,
        alpha_auc=alpha_auc,
        weights=risk_weights,
    )


def train_seed_and_cap(
    first_variant: dict,
    X_train, y_train, X_val, X_test,
    base_max_iter_fallback: int,
    converge_clip_ratio: float,
    logger
) -> Tuple[str, np.ndarray, np.ndarray, int, int]:
    """Train the seed fully and derive the capped max_iter for later candidates."""
    names, val_ps, test_ps, iters = fit_candidates_with_cap(
        [first_variant], X_train, y_train, X_val, X_test, max_iter_cap=None
    )
    first_name = names[0]
    first_val  = val_ps[0]
    first_test = test_ps[0]

    if len(iters) > 0 and int(iters[0]) > 0:
        base_max_iter = int(iters[0])
    else:
        base_max_iter = int(first_variant.get('mlp', {}).get('max_iter', base_max_iter_fallback))

    seed_cap = max(50, int(round(base_max_iter * converge_clip_ratio)))
    logger.info(f"Seed '{first_name}' accepted. Derived max_iter cap for subsequent models = {seed_cap} (base {base_max_iter})")
    return first_name, first_val, first_test, base_max_iter, seed_cap


def propose_and_maybe_accept(
    variant: dict,
    X_train, y_train, X_val, X_test,
    cap: int,
    y_val,
    selected_val_probas: List[np.ndarray],
    selection_metric: str,
    alpha_auc: float,
    risk_weights: Dict[str, float],
    risk_accept_threshold: float,
    logger
) -> Tuple[str, np.ndarray, np.ndarray, bool, float]:
    """
    Train a capped candidate, compute risk vs current ensemble, decide accept/reject.
    Returns (name, val_proba, test_proba, accepted, score).
    """
    names, val_ps, test_ps, _ = fit_candidates_with_cap(
        [variant], X_train, y_train, X_val, X_test, max_iter_cap=cap
    )
    cname, cval, ctest = names[0], val_ps[0], test_ps[0]

    score = compute_risk_score(
        y_val=y_val,
        cand_proba=cval,
        selected_val_probas=selected_val_probas,
        selection_metric=selection_metric,
        alpha_auc=alpha_auc,
        risk_weights=risk_weights,
    )

    accepted = bool(score >= risk_accept_threshold)
    if accepted:
        logger.info(f"Accepted '{cname}'  risk={score:.4f} (>= {risk_accept_threshold})")
    else:
        logger.info(f"Rejected '{cname}'  risk={score:.4f} (< {risk_accept_threshold})")
    return cname, cval, ctest, accepted, float(score)


def tune_threshold(y_val, ens_val: np.ndarray, objective: str = "f1") -> float:
    """Grid-search threshold on validation for the ensemble."""
    yv_np = y_val.to_numpy() if hasattr(y_val, 'to_numpy') else np.asarray(y_val)
    pos_val = positive_class_probability(ens_val)
    grid = np.linspace(0.05, 0.95, 181)
    best_t, best_s = 0.5, -1.0
    for t in grid:
        pred = (pos_val >= t).astype(int)
        if objective == 'accuracy':
            s = float((pred == yv_np).mean())
        elif objective in ('balanced_accuracy', 'bal_acc'):
            tp = int(((pred == 1) & (yv_np == 1)).sum()); fn = int(((pred == 0) & (yv_np == 1)).sum())
            tn = int(((pred == 0) & (yv_np == 0)).sum()); fp = int(((pred == 1) & (yv_np == 0)).sum())
            tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
            s = 0.5 * (tpr + tnr)
        else:
            from sklearn.metrics import f1_score
            s = float(f1_score(yv_np, pred, zero_division=0))
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t


def save_member_tables(out_dir: Path, split: str, plist: List[np.ndarray], names: List[str] | None = None) -> None:
    """Write per-member probability and prediction tables.
    Only the provided plist (e.g., SELECTED members) will be written.
    Optionally use provided names as row indices.
    """
    if not plist:
        return
    pm = np.array([positive_class_probability(p) for p in plist])
    if names is not None and len(names) == len(plist):
        idx = list(names)
    else:
        idx = [f"member_{i+1:03d}" for i in range(len(plist))]
    dfp = pd.DataFrame(
        pm,
        index=idx,
        columns=[f"sample_{j+1:04d}" for j in range(pm.shape[1])]
    )
    dfp.to_csv(out_dir / f"member_{split}_proba_table.csv")
    (dfp >= 0.5).astype(int).to_csv(out_dir / f"member_{split}_predictions.csv")


def export_member_and_ensemble_metrics(
    summary_dir: Path,
    selected_names: List[str],
    selected_val: List[np.ndarray],
    selected_test: List[np.ndarray],
    y_val, y_test,
    ens_val: np.ndarray,
    ens_test: np.ndarray,
    thr: float
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Export metrics for SELECTED members only, plus ensemble metrics."""
    rows = []
    for i, (name, v, t) in enumerate(zip(selected_names, selected_val, selected_test)):
        mid = f"{i+1:03d}"
        vm = metrics_block(y_val, v); tm = metrics_block(y_test, t)
        rows.append({
            'model': 'mlp', 'type': 'member', 'member_id': mid, 'name': str(name),
            'accuracy_val': float(vm['accuracy']), 'precision_val': float(vm['precision']),
            'recall_val': float(vm['recall']),   'f1_val': float(vm['f1']), 'roc_auc_val': float(vm['roc_auc']),
            'log_loss_val': float(vm['log_loss']), 'ece_val': float(vm['ece']),
            'accuracy_test': float(tm['accuracy']), 'precision_test': float(tm['precision']),
            'recall_test': float(tm['recall']),   'f1_test': float(tm['f1']), 'roc_auc_test': float(tm['roc_auc']),
            'log_loss_test': float(tm['log_loss']), 'ece_test': float(tm['ece']),
        })

    # Ensemble metrics with tuned threshold
    yv_np = y_val.to_numpy() if hasattr(y_val, 'to_numpy') else np.asarray(y_val)
    yt_np = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else np.asarray(y_test)

    pos_val = positive_class_probability(ens_val)
    pos_test = positive_class_probability(ens_test)

    vm = compute_basic_metrics(yv_np, ens_val, (pos_val >= thr).astype(int))
    vm['ece'] = float(expected_calibration_error(yv_np, ens_val))

    tm = compute_basic_metrics(yt_np, ens_test, (pos_test >= thr).astype(int))
    tm['ece'] = float(expected_calibration_error(yt_np, ens_test))

    rows.append({
        'model': 'mlp', 'type': 'ensemble', 'member_id': 'ensemble', 'n_members': len(selected_names),
        'accuracy_val': float(vm['accuracy']), 'precision_val': float(vm['precision']), 'recall_val': float(vm['recall']),
        'f1_val': float(vm['f1']), 'roc_auc_val': float(vm['roc_auc']), 'log_loss_val': float(vm['log_loss']), 'ece_val': float(vm['ece']),
        'accuracy_test': float(tm['accuracy']), 'precision_test': float(tm['precision']), 'recall_test': float(tm['recall']),
        'f1_test': float(tm['f1']), 'roc_auc_test': float(tm['roc_auc']), 'log_loss_test': float(tm['log_loss']), 'ece_test': float(tm['ece']),
    })

    df = pd.DataFrame(rows)
    df.to_csv(summary_dir / 'members_and_ensemble_metrics.csv', index=False)
    return vm, tm


def append_experiments_summary(
    base_dir: Path, results_family: str, exp: str,
    selected_names: List[str], tm: Dict[str, float], thr: float,
    trained_candidates: int
) -> None:
    exp_dir = base_dir / results_family / 'experiments'
    exp_dir.mkdir(parents=True, exist_ok=True)
    row = {
        'experiment_name': exp,
        'n_members': int(len(selected_names)),
        'f1_test': float(tm['f1']),
        'accuracy_test': float(tm['accuracy']),
        'roc_auc_test': float(tm['roc_auc']),
        'log_loss_test': float(tm['log_loss']),
        'ece_test': float(tm['ece']),
        'threshold_objective': 'tuned_on_val',
        'threshold': thr,
        'trained_candidates': int(trained_candidates),
    }
    exp_csv = exp_dir / 'experiments_summary.csv'
    if exp_csv.exists():
        df_exp = pd.read_csv(exp_csv)
        df_exp = pd.concat([df_exp, pd.DataFrame([row])], ignore_index=True)
    else:
        df_exp = pd.DataFrame([row])
    df_exp.to_csv(exp_csv, index=False)


# -------------------------
# Main
# -------------------------

def run_ensemble_mlp_classic_simplified(config_path: Path | None = None,
                                        exp_name: str | None = None) -> Dict[str, Any]:
    cfg, logger, (X_train, X_val, X_test, y_train, y_val, y_test) = load_cfg_and_data(config_path)

    base_dir = Path(cfg['output']['results_dir'])
    results_family = cfg['output'].get('results_family', 'ensemble_mlp_classic')
    exp = exp_name or cfg['output'].get('experiment_name', 'mlp_classic_simplified')
    out_dir, summary_dir = ensure_dirs(base_dir, results_family, exp)

    sel_cfg = cfg.get('selection', {}) or {}
    selection_metric = str(sel_cfg.get('selection_metric', 'f1')).lower()
    alpha_auc = float(sel_cfg.get('selection_alpha_auc', 0.0))
    raw_weights = (sel_cfg.get('risk_weights', {}) or {})
    risk_weights = _map_risk_weights(raw_weights)

    target_members = int(sel_cfg.get('target_members', 20))
    risk_accept_threshold = float(sel_cfg.get('risk_accept_threshold', 0.3))
    base_max_iter_fallback = int(sel_cfg.get('base_max_iter_fallback', 300))
    converge_clip_ratio = float(sel_cfg.get('convergence_clip_ratio', 0.85))  # default 85%
    # Optional per-candidate cap randomization
    cap_min = float(sel_cfg.get('cap_clip_ratio_min', None) or np.nan)
    cap_max = float(sel_cfg.get('cap_clip_ratio_max', None) or np.nan)
    use_cap_range = np.isfinite(cap_min) and np.isfinite(cap_max) and (0.1 <= cap_min < cap_max <= 1.5)
    # Shuffle control
    shuffle_variants = bool(sel_cfg.get('shuffle_variants', True))
    shuffle_seed = int(sel_cfg.get('shuffle_seed', cfg.get('dataset', {}).get('random_state', 42)))

    # Build deterministic variant grid
    all_variants = list(candidate_space_from_cfg(cfg))
    if shuffle_variants:
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(all_variants)
        get_logger("mlp_classic_ensemble_simplified").info(
            f"Shuffled variant order with seed={shuffle_seed} (n={len(all_variants)})"
        )
    if not all_variants:
        raise RuntimeError("Variant grid is empty. Check your YAML model.variations/base.")

    # --- State holders
    selected_names: List[str] = []
    selected_val_probas: List[np.ndarray] = []
    selected_test_probas: List[np.ndarray] = []

    # Registry of ALL trained candidates (for export tables)
    registry_names: List[str] = []
    registry_val: List[np.ndarray] = []
    registry_test: List[np.ndarray] = []

    # Track per-candidate risk (including seed) for analysis/export
    risk_records: List[Dict[str, Any]] = []

    # === 1) Seed: full train + accept, derive cap
    first_variant = all_variants[0]
    first_name, first_val, first_test, base_max_iter_seed, seed_cap = train_seed_and_cap(
        first_variant, X_train, y_train, X_val, X_test,
        base_max_iter_fallback, converge_clip_ratio, logger
    )
    registry_names.append(first_name); registry_val.append(first_val); registry_test.append(first_test)
    selected_names.append(first_name); selected_val_probas.append(first_val); selected_test_probas.append(first_test)

    # Compute and record seed risk (vs empty ensemble) for reference
    try:
        seed_risk = compute_risk_score(
            y_val=y_val,
            cand_proba=first_val,
            selected_val_probas=[],
            selection_metric=selection_metric,
            alpha_auc=alpha_auc,
            risk_weights=risk_weights,
        )
    except Exception:
        seed_risk = None
    risk_records.append({
        'order': 0,
        'name': first_name,
        'accepted': True,
        'risk': float(seed_risk) if seed_risk is not None else np.nan,
        'seed': True,
    })

    # === 2) Sequential proposals until target_members or variants exhausted
    idx = 1
    while len(selected_names) < target_members and idx < len(all_variants):
        v = all_variants[idx]
        # Determine per-candidate cap
        if use_cap_range:
            ratio = float(np.random.default_rng(shuffle_seed + idx).uniform(cap_min, cap_max))
        else:
            ratio = converge_clip_ratio
        cap_this = max(50, int(round(base_max_iter_seed * ratio)))
        cname, cval, ctest, accepted, score = propose_and_maybe_accept(
            variant=v,
            X_train=X_train, y_train=y_train, X_val=X_val, X_test=X_test,
            cap=cap_this,
            y_val=y_val,
            selected_val_probas=selected_val_probas,
            selection_metric=selection_metric,
            alpha_auc=alpha_auc,
            risk_weights=risk_weights,
            risk_accept_threshold=risk_accept_threshold,
            logger=logger
        )
        registry_names.append(cname); registry_val.append(cval); registry_test.append(ctest)
        if accepted:
            selected_names.append(cname); selected_val_probas.append(cval); selected_test_probas.append(ctest)
        # Record risk for this candidate
        risk_records.append({
            'order': idx,
            'name': cname,
            'accepted': bool(accepted),
            'risk': float(score),
            'seed': False,
        })
        idx += 1

    if len(selected_names) < 1:
        raise RuntimeError("No models selected — check the configuration.")

    if len(selected_names) < target_members:
        logger.warning(
            f"Stopped with {len(selected_names)} selected (target={target_members}). "
            f"Consider lowering selection.risk_accept_threshold or enlarging the grid."
        )

    # === Save per-member probability/prediction tables for SELECTED members only
    save_member_tables(out_dir, 'val', selected_val_probas, names=selected_names)
    save_member_tables(out_dir, 'test', selected_test_probas, names=selected_names)

    # === Export candidate risk log for inspection
    try:
        risk_df = pd.DataFrame(risk_records)
        risk_df.to_csv(summary_dir / 'candidate_risks.csv', index=False)
        if len(risk_df) > 0 and risk_df['risk'].notna().any():
            rvals = risk_df['risk'].dropna()
            logger.info(
                f"Risk stats — count={len(rvals)}, min={rvals.min():.3f}, med={rvals.median():.3f}, "
                f"mean={rvals.mean():.3f}, max={rvals.max():.3f}"
            )
    except Exception as e:
        logger.warning(f"Failed to write candidate_risks.csv: {e}")

    # === Build ensemble from SELECTED models
    sel_val_stack = np.stack(selected_val_probas, axis=0)
    sel_test_stack = np.stack(selected_test_probas, axis=0)
    ens_val = np.mean(sel_val_stack, axis=0)
    ens_test = np.mean(sel_test_stack, axis=0)

    # === Threshold tuning on validation for ensemble
    thr_cfg = cfg.get('threshold', {}) or {}
    objective = str(thr_cfg.get('objective', 'f1')).lower()
    thr = tune_threshold(y_val, ens_val, objective)

    # === Export metrics
    vm, tm = export_member_and_ensemble_metrics(
        summary_dir, selected_names, selected_val_probas, selected_test_probas, y_val, y_test, ens_val, ens_test, thr
    )
    get_logger("mlp_classic_ensemble_simplified").info(f"Wrote members_and_ensemble_metrics.csv to {summary_dir}")

    summary = {
        'experiment_name': exp,
        'n_members': int(len(selected_names)),
        'val_metrics': vm,
        'test_metrics': tm,
        'selected_member_names': list(selected_names),
        'threshold_objective': objective,
        'adaptive_threshold': thr,
        'trained_candidates': int(len(registry_names)),
        'risk_accept_threshold': float(risk_accept_threshold),
    }
    # Add risk distribution to summary (if available)
    try:
        rvals = pd.Series([r['risk'] for r in risk_records], dtype=float).dropna()
        if len(rvals) > 0:
            summary['risk_stats'] = {
                'count': int(len(rvals)),
                'min': float(rvals.min()),
                'median': float(rvals.median()),
                'mean': float(rvals.mean()),
                'max': float(rvals.max()),
            }
    except Exception:
        pass
    with open(summary_dir / 'ensemble_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    get_logger("mlp_classic_ensemble_simplified").info(f"Saved ensemble_summary.json to {summary_dir}")

    # Diversity artifacts (on SELECTED only)
    try:
        write_full_diversity_artifacts(
            selected_val_probas,
            selected_test_probas,
            y_val if hasattr(y_val, 'values') else y_val,
            y_test if hasattr(y_test, 'values') else y_test,
            out_dir=out_dir,
            summary_dir=summary_dir,
            write_pairwise=False,
            write_extended_json=True,
            write_summary_csv=True,
        )
    except Exception as e:
        get_logger('mlp_classic_ensemble_simplified').warning(f"Failed to write diversity artifacts: {e}")
    else:
        get_logger("mlp_classic_ensemble_simplified").info("Wrote diversity artifacts (summary CSV and extended JSON)")

    # Experiments summary
    try:
        append_experiments_summary(
            base_dir, results_family, exp, selected_names, tm, thr,
            trained_candidates=len(registry_names)
        )
    except Exception as e:
        get_logger('mlp_classic_ensemble_simplified').warning(f"Failed to update experiments summary: {e}")

    return summary


if __name__ == '__main__':
    run_ensemble_mlp_classic_simplified()
