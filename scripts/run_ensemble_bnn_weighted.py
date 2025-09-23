"""
Weighted variant of the BNN (MLP, SVI) ensemble pipeline.
Creates outputs under results/ensemble_bnn_weighted/<experiment_name> mirroring the
unweighted pipeline but combining member probabilities with validation-based weights.

Weighting scheme:
  raw_weight_i = alpha_f1 * f1_val_i + (1 - alpha_f1) * accuracy_val_i
  weights = raw_weight / sum(raw_weight)
Defaults: alpha_f1 = 0.6 (puts more emphasis on F1).
If all raw weights are zero (edge case), falls back to uniform weights.

Artifacts:
  - members_and_ensemble_metrics.csv (adds a weighted_ensemble row)
  - weighted_ensemble_summary.json (summary with weights + metrics)
  - member probability & prediction tables (same as base pipeline)
  - diversity / correlation summaries (computed on the selected members as usual)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import copy
import numpy as np
import pandas as pd
import yaml
from extra_util.logger import get_logger
from extra_util.data_loader import DatasetConfig, load_dataset, preprocess_and_split
from models.bayes_mlp_pyro import (
    fit_bayes_mlp,
    select_member_indices,
    proba_for_indices,
    average_proba,
)
from evaluation.metrics import positive_class_probability
from sklearn.metrics import f1_score
from evaluation.helpers import metrics_block
from scripts.build_correlation_summary import (
    export_diversity_and_correlation,
    export_correlation_summary,
    write_correlation_formulas,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ensemble_mlp_config.yaml"
ALPHA_F1 = 0.6  # weighting emphasis toward F1 vs accuracy


def _ensure_dirs(base_dir: Path, exp_name: str):
    out_dir = base_dir / "ensemble_bnn_weighted" / exp_name
    (out_dir / "members").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    return out_dir, out_dir / "summary"


def _load_config(config_path: Path | None) -> dict:
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


def save_member_proba_tables(val_member_probas, test_member_probas, out_dir: Path) -> None:
    import numpy as _np
    def write(name: str, plist):
        pm = _np.array([positive_class_probability(p) for p in plist])
        dfp = pd.DataFrame(
            pm,
            index=[f"member_{i+1:03d}" for i in range(len(plist))],
            columns=[f"sample_{j+1:04d}" for j in range(pm.shape[1])]
        )
        dfp.to_csv(out_dir / f"member_{name}_proba_table.csv")
        (dfp >= 0.5).astype(int).to_csv(out_dir / f"member_{name}_predictions.csv")
    write("val", val_member_probas)
    write("test", test_member_probas)


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _compute_and_persist_weighted(
    cfg: dict,
    exp_name: str,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    strategy: dict | None = None,
) -> Dict:
    base_dir = Path(cfg["output"]["results_dir"])
    out_dir, summary_dir = _ensure_dirs(base_dir, exp_name)
    logger = get_logger(f"bnn_ensemble_weighted:{exp_name}")

    posterior = fit_bayes_mlp(X_train, y_train, cfg, strategy=strategy)
    member_indices = select_member_indices(posterior, X_val, y_val, cfg, strategy=strategy)
    val_member_probas = proba_for_indices(posterior, X_val, member_indices)
    test_member_probas = proba_for_indices(posterior, X_test, member_indices)

    save_member_proba_tables(val_member_probas, test_member_probas, out_dir)

    rows = []
    val_f1_list = []
    val_acc_list = []
    for i, (v, t) in enumerate(zip(val_member_probas, test_member_probas)):
        mid = f"{i+1:03d}"
        vm = metrics_block(y_val, v); tm = metrics_block(y_test, t)
        val_f1_list.append(float(vm["f1"]))
        val_acc_list.append(float(vm["accuracy"]))
        rows.append({
            "model": "bayes_mlp", "type": "member", "member_id": mid,
            "accuracy_val": float(vm["accuracy"]), "precision_val": float(vm["precision"]),
            "recall_val": float(vm["recall"]), "f1_val": float(vm["f1"]),
            "roc_auc_val": float(vm["roc_auc"]), "log_loss_val": float(vm["log_loss"]),
            "ece_val": float(vm["ece"]), "accuracy_test": float(tm["accuracy"]),
            "precision_test": float(tm["precision"]), "recall_test": float(tm["recall"]),
            "f1_test": float(tm["f1"]), "roc_auc_test": float(tm["roc_auc"]),
            "log_loss_test": float(tm["log_loss"]), "ece_test": float(tm["ece"]),
        })

    # Unweighted baseline ensemble (for reference inside weighted run)
    ens_val_unw = average_proba(val_member_probas); ens_test_unw = average_proba(test_member_probas)
    vm_unw = metrics_block(y_val, ens_val_unw); tm_unw = metrics_block(y_test, ens_test_unw)

    # Weighted ensemble probabilities
    val_f1 = np.array(val_f1_list); val_acc = np.array(val_acc_list)
    raw_w = ALPHA_F1 * val_f1 + (1 - ALPHA_F1) * val_acc
    if np.all(raw_w <= 0):
        weights = np.ones_like(raw_w) / len(raw_w)
    else:
        # shift to ensure non-negative
        min_raw = raw_w.min()
        if min_raw < 0:
            raw_w = raw_w - min_raw
        weights = raw_w / raw_w.sum()
    # Combine
    weighted_val = np.sum([w * v for w, v in zip(weights, val_member_probas)], axis=0)
    weighted_test = np.sum([w * v for w, v in zip(weights, test_member_probas)], axis=0)
    vm_w = metrics_block(y_val, weighted_val); tm_w = metrics_block(y_test, weighted_test)

    # Adaptive threshold logic (reuse from base) for weighted ensemble
    pos_val = positive_class_probability(weighted_val); pred_val = (pos_val >= 0.5).astype(int)
    if pred_val.std() == 0:
        thresholds = np.linspace(0.1, 0.9, 17); best_t, best_f1 = 0.5, -1
        yv_np = y_val.to_numpy() if hasattr(y_val, 'to_numpy') else np.asarray(y_val)
        for t in thresholds:
            f1_tmp = f1_score(yv_np, (pos_val >= t).astype(int), zero_division=0)
            if f1_tmp > best_f1:
                best_f1, best_t = f1_tmp, t
        adaptive_threshold = best_t
    else:
        adaptive_threshold = 0.5
    vm_w['adaptive_threshold'] = adaptive_threshold; tm_w['adaptive_threshold'] = adaptive_threshold

    # Append rows for unweighted and weighted ensemble
    rows.append({
        "model": "bayes_mlp", "type": "ensemble_unweighted", "member_id": "ensemble_unw", "n_members": len(member_indices),
        "accuracy_val": float(vm_unw["accuracy"]), "precision_val": float(vm_unw["precision"]),
        "recall_val": float(vm_unw["recall"]), "f1_val": float(vm_unw["f1"]), "roc_auc_val": float(vm_unw["roc_auc"]),
        "log_loss_val": float(vm_unw["log_loss"]), "ece_val": float(vm_unw["ece"]),
        "accuracy_test": float(tm_unw["accuracy"]), "precision_test": float(tm_unw["precision"]),
        "recall_test": float(tm_unw["recall"]), "f1_test": float(tm_unw["f1"]), "roc_auc_test": float(tm_unw["roc_auc"]),
        "log_loss_test": float(tm_unw["log_loss"]), "ece_test": float(tm_unw["ece"]),
    })
    rows.append({
        "model": "bayes_mlp", "type": "ensemble_weighted", "member_id": "ensemble_w", "n_members": len(member_indices),
        "accuracy_val": float(vm_w["accuracy"]), "precision_val": float(vm_w["precision"]),
        "recall_val": float(vm_w["recall"]), "f1_val": float(vm_w["f1"]), "roc_auc_val": float(vm_w["roc_auc"]),
        "log_loss_val": float(vm_w["log_loss"]), "ece_val": float(vm_w["ece"]),
        "accuracy_test": float(tm_w["accuracy"]), "precision_test": float(tm_w["precision"]),
        "recall_test": float(tm_w["recall"]), "f1_test": float(tm_w["f1"]), "roc_auc_test": float(tm_w["roc_auc"]),
        "log_loss_test": float(tm_w["log_loss"]), "ece_test": float(tm_w["ece"]),
    })

    df = pd.DataFrame(rows); df.to_csv(summary_dir / "members_and_ensemble_metrics.csv", index=False)

    # Diversity metrics (still based on member predictions themselves)
    diversity, cond_err_corr = export_diversity_and_correlation(
        val_member_probas, test_member_probas, y_val, y_test, out_dir, summary_dir
    )
    export_correlation_summary(diversity, cond_err_corr, summary_dir); write_correlation_formulas(summary_dir)

    summary = {
        "experiment_name": exp_name,
        "n_members": len(member_indices),
        "val_metrics_weighted": vm_w,
        "test_metrics_weighted": tm_w,
        "val_metrics_unweighted": vm_unw,
        "test_metrics_unweighted": tm_unw,
        "weights": weights.tolist(),
        "weight_formula": f"raw_w = {ALPHA_F1}*f1_val + {1-ALPHA_F1}*accuracy_val (normalized)",
        "member_indices": list(map(int, member_indices)),
        "bayes_config": cfg.get("bayes", {}).get("bayes_mlp", {}),
    }
    with open(summary_dir / "weighted_ensemble_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # Also emit weights CSV
    pd.DataFrame({
        'member_id': [f"member_{i+1:03d}" for i in range(len(weights))],
        'weight': weights,
        'f1_val': val_f1,
        'accuracy_val': val_acc,
    }).to_csv(summary_dir / 'member_weights.csv', index=False)

    logger.info(f"Weighted BNN ensemble complete. Results saved to {out_dir}")
    return summary


def run_ensemble_bnn_weighted(config_path: Path | None = None) -> Dict:
    cfg = _load_config(config_path)
    logger = get_logger("bnn_ensemble_weighted:loader")
    dcfg = DatasetConfig(**cfg["dataset"])
    X, y = load_dataset(dcfg)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_and_split(
        X, y, test_size=dcfg.test_size, val_size=dcfg.val_size,
        random_state=dcfg.random_state, scale_numeric=dcfg.scale_numeric,
    )
    logger.info(f"Shapes | train:{X_train.shape} val:{X_val.shape} test:{X_test.shape}")
    strategies = (cfg.get('experiments', {}) or {}).get('strategies', {}) or {}
    summaries: list[Dict[str, Any]] = []
    if strategies:
        logger.info(f"Running {len(strategies)} weighted strategies...")
        for name, overrides in strategies.items():
            run_cfg = copy.deepcopy(cfg)
            _deep_update(run_cfg, overrides)
            exp_name_local = overrides.get('output', {}).get('experiment_name', f"{cfg['output'].get('experiment_name','bnn')}_{name}")
            summary = _compute_and_persist_weighted(run_cfg, exp_name_local, X_train, X_val, X_test, y_train, y_val, y_test, strategy=None)
            summaries.append({'experiment': exp_name_local, 'f1_val_weighted': summary['val_metrics_weighted']['f1'], 'f1_test_weighted': summary['test_metrics_weighted']['f1']})
    else:
        base_exp = cfg['output'].get('experiment_name', 'bnn')
        summary_single = _compute_and_persist_weighted(cfg, base_exp, X_train, X_val, X_test, y_train, y_val, y_test, strategy=None)
        summaries.append({'experiment': base_exp, 'f1_val_weighted': summary_single['val_metrics_weighted']['f1'], 'f1_test_weighted': summary_single['test_metrics_weighted']['f1']})

    if summaries:
        agg_dir = Path(cfg['output']['results_dir']) / 'ensemble_bnn_weighted' / 'experiments'
        agg_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(agg_dir / 'experiments_summary.csv', index=False)
    return {'experiments': summaries}


if __name__ == "__main__":
    run_ensemble_bnn_weighted()
