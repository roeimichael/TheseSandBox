"""
scripts/run_ensemble_nuts.py
NUTS ensemble pipeline for Bayesian Logistic Regression.
"""
from __future__ import annotations
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List

from utils.logger import get_logger
from utils.data_loader import DatasetConfig, load_dataset, preprocess_and_split
from models.bayes_logreg_pymc import fit_bayes_logreg, predict_proba_members, average_proba
from evaluation.metrics import compute_basic_metrics
from evaluation.metrics import ensemble_diversity_summary
from evaluation.metrics import expected_calibration_error, positive_class_probability
from evaluation.conditional_correlation import error_correlation_summary

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ensemble_config.yaml"


def _ensure_nuts_dirs(base_dir: Path, exp_name: str):
    """Create directory structure for NUTS ensemble results."""
    nuts_dir = base_dir / "ensemble_nuts" / exp_name
    (nuts_dir / "members").mkdir(parents=True, exist_ok=True)
    return nuts_dir


def run_ensemble_nuts(
    config_path: Path | None = None,
    exp_name: str | None = None
) -> Dict:
    """
    Run NUTS ensemble analysis for Bayesian Logistic Regression.
    """
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    
    logger = get_logger("nuts_ensemble")
    out_dir = Path(cfg["output"]["results_dir"])
    exp_name = exp_name or cfg["output"]["experiment_name"]
    
    # Load & split data
    dcfg = DatasetConfig(**cfg["dataset"])
    X, y = load_dataset(dcfg)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_and_split(
        X, y,
        test_size=dcfg.test_size, val_size=dcfg.val_size,
        random_state=dcfg.random_state, scale_numeric=dcfg.scale_numeric
    )
    logger.info(f"Shapes | train:{X_train.shape} val:{X_val.shape} test:{X_test.shape}")
    
    # Fit Bayesian Logistic Regression with NUTS
    logger.info("Fitting Bayesian Logistic Regression with NUTS...")
    idata = fit_bayes_logreg(X_train, y_train, cfg)
    logger.info("NUTS sampling completed successfully")
    
    # Generate member predictions
    logger.info("Generating ensemble member predictions...")
    val_member_probas = predict_proba_members(idata, X_val, cfg)
    test_member_probas = predict_proba_members(idata, X_test, cfg)
    
    n_members = len(val_member_probas)
    logger.info(f"Generated {n_members} ensemble members")
    
    # Create output directory
    nuts_dir = _ensure_nuts_dirs(out_dir, exp_name)
    
    # Compute metrics for each member and save member CSVs (with ECE)
    for i, (val_proba, test_proba) in enumerate(zip(val_member_probas, test_member_probas)):
        member_id = f"{i+1:03d}"  # 001, 002, etc.
        
        # Validation metrics
        val_pred = np.argmax(val_proba, axis=1)
        val_metrics = compute_basic_metrics(y_val, val_proba, val_pred)
        val_ece = expected_calibration_error(y_val.values if hasattr(y_val, 'values') else y_val, val_proba)
        val_metrics["ece"] = float(val_ece)
        
        # Test metrics
        test_pred = np.argmax(test_proba, axis=1)
        test_metrics = compute_basic_metrics(y_test, test_proba, test_pred)
        test_ece = expected_calibration_error(y_test.values if hasattr(y_test, 'values') else y_test, test_proba)
        test_metrics["ece"] = float(test_ece)
        
        # Save individual member CSV
        member_df = pd.DataFrame([
            {"model_name": "bayes_logreg", "member_id": member_id, "split": "val", **val_metrics},
            {"model_name": "bayes_logreg", "member_id": member_id, "split": "test", **test_metrics},
        ])
        member_csv_path = nuts_dir / "members" / f"bayes_logreg_{member_id}.csv"
        member_df.to_csv(member_csv_path, index=False)
    
    # Compute ensemble metrics (average of members) and ECE
    logger.info("Computing ensemble metrics...")
    
    # Average validation probabilities
    ensemble_val_proba = average_proba(val_member_probas)
    ensemble_val_pred = np.argmax(ensemble_val_proba, axis=1)
    ensemble_val_metrics = compute_basic_metrics(y_val, ensemble_val_proba, ensemble_val_pred)
    ensemble_val_metrics["ece"] = float(expected_calibration_error(y_val.values if hasattr(y_val, 'values') else y_val, ensemble_val_proba))
    
    # Average test probabilities
    ensemble_test_proba = average_proba(test_member_probas)
    ensemble_test_pred = np.argmax(ensemble_test_proba, axis=1)
    ensemble_test_metrics = compute_basic_metrics(y_test, ensemble_test_proba, ensemble_test_pred)
    ensemble_test_metrics["ece"] = float(expected_calibration_error(y_test.values if hasattr(y_test, 'values') else y_test, ensemble_test_proba))
    
    # Save ensemble metrics CSV
    ensemble_df = pd.DataFrame([
        {"model": "bayes_logreg", "type": "ensemble", "split": "val", "n_members": n_members, **ensemble_val_metrics},
        {"model": "bayes_logreg", "type": "ensemble", "split": "test", "n_members": n_members, **ensemble_test_metrics},
    ])
    ensemble_csv_path = nuts_dir / "ensemble_metrics.csv"
    ensemble_df.to_csv(ensemble_csv_path, index=False)

    # Diversity analysis across members (predicted labels)
    val_member_preds = {f"member_{i+1:03d}": np.argmax(p, axis=1) for i, p in enumerate(val_member_probas)}
    test_member_preds = {f"member_{i+1:03d}": np.argmax(p, axis=1) for i, p in enumerate(test_member_probas)}

    diversity = {
        "val": ensemble_diversity_summary(val_member_preds),
        "test": ensemble_diversity_summary(test_member_preds),
    }

    # Conditional error correlations (overall and by class)
    cond_err_corr = {
        "val": error_correlation_summary(y_val.values if hasattr(y_val, 'values') else y_val, val_member_preds),
        "test": error_correlation_summary(y_test.values if hasattr(y_test, 'values') else y_test, test_member_preds),
    }

    with open(nuts_dir / "diversity.json", "w") as f:
        json.dump(diversity, f, indent=2)

    with open(nuts_dir / "conditional_error_correlation.json", "w") as f:
        json.dump(cond_err_corr, f, indent=2)
    
    # Save summary JSON
    summary = {
        "experiment_name": exp_name,
        "n_draws_used": int(idata.posterior["intercept"].sizes["chain"] * idata.posterior["intercept"].sizes["draw"]),
        "thin": cfg.get("bayes", {}).get("bayes_logreg", {}).get("thin", 20),
        "n_members": n_members,
        "val_metrics": ensemble_val_metrics,
        "test_metrics": ensemble_test_metrics,
        "bayes_config": cfg.get("bayes", {}).get("bayes_logreg", {})
    }
    
    summary_path = nuts_dir / "ensemble_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Log results
    logger.info(f"Ensemble validation F1: {ensemble_val_metrics['f1']:.4f}")
    logger.info(f"Ensemble test F1: {ensemble_test_metrics['f1']:.4f}")
    logger.info(f"NUTS ensemble analysis complete. Results saved to {nuts_dir}")
    
    return summary


if __name__ == "__main__":
    run_ensemble_nuts()
