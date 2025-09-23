"""
scripts/run_ensemble_bnn.py
SVI-based Bayesian MLP ensemble pipeline (mirrors the NUTS LR runner outputs).
"""
from __future__ import annotations
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any
import copy
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


def _ensure_dirs(base_dir: Path, exp_name: str):
    out_dir = base_dir / "ensemble_bnn" / exp_name
    (out_dir / "members").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    return out_dir, out_dir / "summary"


def _load_config(config_path: Path | None) -> dict:
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


def save_member_proba_tables(val_member_probas, test_member_probas, out_dir: Path) -> None:
    def write(name: str, plist):
        pm = np.array([positive_class_probability(p) for p in plist])
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


def _compute_and_persist(
    cfg: dict,
    exp_name: str,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    strategy: dict | None = None,
) -> Dict:
    base_dir = Path(cfg["output"]["results_dir"])
    out_dir, summary_dir = _ensure_dirs(base_dir, exp_name)
    logger = get_logger(f"bnn_ensemble:{exp_name}")
    posterior = fit_bayes_mlp(X_train, y_train, cfg, strategy=strategy)
    member_indices = select_member_indices(posterior, X_val, y_val, cfg, strategy=strategy)
    val_member_probas = proba_for_indices(posterior, X_val, member_indices)
    test_member_probas = proba_for_indices(posterior, X_test, member_indices)
    save_member_proba_tables(val_member_probas, test_member_probas, out_dir)
    rows = []
    for i, (v, t) in enumerate(zip(val_member_probas, test_member_probas)):
        mid = f"{i+1:03d}"
        vm = metrics_block(y_val, v); tm = metrics_block(y_test, t)
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
    ens_val = average_proba(val_member_probas); ens_test = average_proba(test_member_probas)
    pos_val = positive_class_probability(ens_val); pred_val = (pos_val >= 0.5).astype(int)
    if pred_val.std() == 0:
        thresholds = np.linspace(0.1, 0.9, 17); best_t, best_f1 = 0.5, -1
        yv_np = y_val.to_numpy() if hasattr(y_val, 'to_numpy') else np.asarray(y_val)
        for t in thresholds:
            f1 = f1_score(yv_np, (pos_val >= t).astype(int), zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, t
        adaptive_threshold = best_t
    else:
        adaptive_threshold = 0.5
    vm = metrics_block(y_val, ens_val); tm = metrics_block(y_test, ens_test)
    vm['adaptive_threshold'] = adaptive_threshold; tm['adaptive_threshold'] = adaptive_threshold
    rows.append({
        "model": "bayes_mlp", "type": "ensemble", "member_id": "ensemble", "n_members": len(member_indices),
        "accuracy_val": float(vm["accuracy"]), "precision_val": float(vm["precision"]),
        "recall_val": float(vm["recall"]), "f1_val": float(vm["f1"]), "roc_auc_val": float(vm["roc_auc"]),
        "log_loss_val": float(vm["log_loss"]), "ece_val": float(vm["ece"]),
        "accuracy_test": float(tm["accuracy"]), "precision_test": float(tm["precision"]),
        "recall_test": float(tm["recall"]), "f1_test": float(tm["f1"]), "roc_auc_test": float(tm["roc_auc"]),
        "log_loss_test": float(tm["log_loss"]), "ece_test": float(tm["ece"]),
    })
    df = pd.DataFrame(rows); df.to_csv(summary_dir / "members_and_ensemble_metrics.csv", index=False)
    diversity, cond_err_corr = export_diversity_and_correlation(
        val_member_probas, test_member_probas, y_val, y_test, out_dir, summary_dir
    )
    export_correlation_summary(diversity, cond_err_corr, summary_dir); write_correlation_formulas(summary_dir)
    summary = {
        "experiment_name": exp_name, "n_members": len(member_indices),
        "val_metrics": vm, "test_metrics": tm,
        "bayes_config": cfg.get("bayes", {}).get("bayes_mlp", {}),
        "strategy_overrides": strategy or {},
        "selected_member_indices": list(map(int, member_indices)),
        "adaptive_threshold": adaptive_threshold,
    }
    with open(summary_dir / "ensemble_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"BNN ensemble analysis complete. Results saved to {out_dir}")
    return summary




def run_ensemble_bnn(config_path: Path | None = None, exp_name: str | None = None, strategy_name: str | None = None) -> Dict:
    cfg = _load_config(config_path)
    logger = get_logger("bnn_ensemble:loader")
    dcfg = DatasetConfig(**cfg["dataset"])
    X, y = load_dataset(dcfg)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_and_split(
        X, y, test_size=dcfg.test_size, val_size=dcfg.val_size,
        random_state=dcfg.random_state, scale_numeric=dcfg.scale_numeric,
    )
    logger.info(f"Shapes | train:{X_train.shape} val:{X_val.shape} test:{X_test.shape}")
    def _dist(yset, name):
        import numpy as _np
        arr = yset.to_numpy() if hasattr(yset, 'to_numpy') else _np.asarray(yset)
        pos = (arr == 1).sum(); neg = (arr == 0).sum(); logger.info(f"Class dist {name}: pos={pos} neg={neg} pos_rate={pos/len(arr):.3f}")
    _dist(y_train,'train'); _dist(y_val,'val'); _dist(y_test,'test')

    strategies = (cfg.get('experiments', {}) or {}).get('strategies', {}) or {}
    summaries: list[Dict[str, Any]] = []

    if strategies:
        logger.info(f"Running {len(strategies)} configured strategies...")
        for name, overrides in strategies.items():
            run_cfg = copy.deepcopy(cfg)
            _deep_update(run_cfg, overrides)
            exp_name_local = overrides.get('output', {}).get('experiment_name', f"{cfg['output'].get('experiment_name','bnn')}_{name}")
            summary = _compute_and_persist(run_cfg, exp_name_local, X_train, X_val, X_test, y_train, y_val, y_test, strategy=None)
            summaries.append({'experiment': exp_name_local, 'f1_val': summary['val_metrics']['f1'], 'f1_test': summary['test_metrics']['f1']})

    # If no strategies defined, fall back to single original run
    if not strategies:
        base_exp = exp_name or cfg['output'].get('experiment_name', 'bnn')
        summary_single = _compute_and_persist(cfg, base_exp, X_train, X_val, X_test, y_train, y_val, y_test, strategy=None)
        summaries.append({'experiment': base_exp, 'f1_val': summary_single['val_metrics']['f1'], 'f1_test': summary_single['test_metrics']['f1']})

    # Aggregate summary CSV
    if summaries:
        agg_dir = Path(cfg['output']['results_dir']) / 'ensemble_bnn' / 'experiments'
        agg_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(agg_dir / 'experiments_summary.csv', index=False)
        logger.info(f"Aggregated experiments summary written to {agg_dir / 'experiments_summary.csv'}")
    return {'experiments': summaries}


if __name__ == "__main__":
    run_ensemble_bnn()
