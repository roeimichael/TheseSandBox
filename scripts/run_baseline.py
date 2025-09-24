"""
scripts/run_baseline.py
Baseline training & evaluation pipeline (no CLI args).
"""
from __future__ import annotations
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import sys, pathlib
# Guarantee project root and evaluation package on path early
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from extra_util.logger import get_logger
from extra_util.data_loader import DatasetConfig, load_dataset, preprocess_and_split
from models.baseline_models import get_models, fit_and_predict
# Local import to avoid external package name collision
from importlib import import_module as _import_module
compute_basic_metrics = _import_module('evaluation.metrics').compute_basic_metrics

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment_config.yaml"


def _ensure_dirs(base_dir: Path):
    (base_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (base_dir / "predictions").mkdir(parents=True, exist_ok=True)



def run_baseline(config_path: Path | None = None) -> dict:
    """
    Runs the baseline workflow end-to-end using the given config path.
    If config_path is None, uses DEFAULT_CONFIG_PATH.
    """
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("baseline")
    out_dir = Path(cfg["output"]["results_dir"])
    _ensure_dirs(out_dir)

    # Load & split data
    dcfg = DatasetConfig(**cfg["dataset"])
    X, y = load_dataset(dcfg)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_and_split(
        X, y,
        test_size=dcfg.test_size, val_size=dcfg.val_size,
        random_state=dcfg.random_state, scale_numeric=dcfg.scale_numeric
    )
    logger.info(f"Shapes | train:{X_train.shape} val:{X_val.shape} test:{X_test.shape}")

    # Build, fit, predict
    models = get_models(cfg["models"])
    logger.info(f"Training models: {list(models.keys())}")
    fitted, val_proba, test_proba, val_pred = fit_and_predict(models, X_train, y_train, X_val, y_val, X_test)

    # Validation metrics per model
    val_scores = {}
    for name in models.keys():
        m = compute_basic_metrics(y_val, val_proba[name], val_pred[name])
        val_scores[name] = m
        logger.info(f"[VAL] {name}: " + ", ".join([f"{k}={v:.4f}" for k, v in m.items()]))

    best_name = max(val_scores, key=lambda n: val_scores[n]["f1"])
    logger.info(f"Best model on VAL (by F1): {best_name}")

    # Test metrics per model
    test_scores = {}
    for name in models.keys():
        y_prob = test_proba[name]
        y_pred = np.argmax(y_prob, axis=1)
        tm = compute_basic_metrics(y_test, y_prob, y_pred)
        test_scores[name] = tm
        logger.info(f"[TEST] {name}: " + ", ".join([f"{k}={v:.4f}" for k, v in tm.items()]))

    # Save outputs
    exp_name = cfg["output"].get("experiment_name", "baseline")
    metrics_csv_path = None
    if cfg["output"].get("save_metrics_csv", True):
        md = pd.DataFrame(val_scores).T
        md["split"] = "val"
        td = pd.DataFrame(test_scores).T
        td["split"] = "test"
        allm = pd.concat([md, td])
        (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
        metrics_csv_path = out_dir / "metrics" / f"{exp_name}_metrics.csv"
        allm.to_csv(metrics_csv_path, index=True)

    if cfg["output"].get("save_predictions", True):
        (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
        y_test_proba_best = test_proba[best_name]
        y_test_pred_best = np.argmax(y_test_proba_best, axis=1)
        np.save(out_dir / "predictions" / f"{exp_name}_{best_name}_test_proba.npy", y_test_proba_best)
        np.save(out_dir / "predictions" / f"{exp_name}_{best_name}_test_pred.npy", y_test_pred_best)
        np.save(out_dir / "predictions" / f"{exp_name}_{best_name}_y_test.npy", y_test.values)

    summary = {
        "experiment": exp_name,
        "best_model": best_name,
        "val_scores": val_scores[best_name],
        "test_scores": test_scores[best_name],
    }
    with open(out_dir / f"{exp_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Baseline run complete.")
    return summary

