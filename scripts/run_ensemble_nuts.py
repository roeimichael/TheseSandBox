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
from extra_util.logger import get_logger
from extra_util.data_loader import DatasetConfig, load_dataset, preprocess_and_split
from models.bayes_logreg_pyro import (
    fit_bayes_logreg,
    select_member_indices,
    proba_for_indices,
    average_proba,
)
from evaluation.metrics import positive_class_probability
from evaluation.helpers import metrics_block
from scripts.build_correlation_summary import (
    export_diversity_and_correlation,
    export_correlation_summary,
    write_correlation_formulas,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ensemble_config.yaml"


def _ensure_nuts_dirs(base_dir: Path, exp_name: str):
    """Create directory structure for NUTS ensemble results."""
    nuts_dir = base_dir / "ensemble_nuts" / exp_name
    return nuts_dir

def save_member_proba_tables(val_member_probas: list[np.ndarray], test_member_probas: list[np.ndarray], out_dir: Path) -> None:
    """Write proba and prediction tables for both val and test sets."""
    def write(name: str, plist: list[np.ndarray]):
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


def _load_config(config_path: Path | None) -> tuple[dict, Path]:
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg, cfg_file


def _load_experiments_cfg() -> dict:
    exp_file = Path(__file__).resolve().parents[1] / "configs" / "ensemble_experiments.yaml"
    if not exp_file.exists():
        return {}
    with open(exp_file, "r") as f:
        return yaml.safe_load(f) or {}


def _load_and_split_data(cfg: dict):
    dcfg = DatasetConfig(**cfg["dataset"])
    X, y = load_dataset(dcfg)
    return preprocess_and_split(
        X, y,
        test_size=dcfg.test_size, val_size=dcfg.val_size,
        random_state=dcfg.random_state, scale_numeric=dcfg.scale_numeric,
    )


def _fit_and_predict(cfg: dict, X_train, y_train, X_val, y_val, X_test, y_test, strategy: dict | None = None):
    idata = fit_bayes_logreg(X_train, y_train, cfg, strategy=strategy)
    # Select indices once on validation
    member_indices = select_member_indices(idata, X_val, y_val, cfg, strategy=strategy)
    # Compute probas for both val & test using the same indices
    val_member_probas = proba_for_indices(idata, X_val, member_indices)
    test_member_probas = proba_for_indices(idata, X_test, member_indices)
    return idata, member_indices, val_member_probas, test_member_probas


def _member_metrics_rows(val_member_probas, test_member_probas, y_val, y_test) -> list[dict]:
    rows: list[dict] = []
    for i, (val_proba, test_proba) in enumerate(zip(val_member_probas, test_member_probas)):
        member_id = f"{i+1:03d}"
        val_metrics = metrics_block(y_val, val_proba)
        test_metrics = metrics_block(y_test, test_proba)
        rows.append({
            "model": "bayes_logreg",
            "type": "member",
            "member_id": member_id,
            # Validation
            "accuracy_val": float(val_metrics["accuracy"]),
            "precision_val": float(val_metrics["precision"]),
            "recall_val": float(val_metrics["recall"]),
            "f1_val": float(val_metrics["f1"]),
            "roc_auc_val": float(val_metrics["roc_auc"]),
            "log_loss_val": float(val_metrics["log_loss"]),
            "ece_val": float(val_metrics["ece"]),
            # Test
            "accuracy_test": float(test_metrics["accuracy"]),
            "precision_test": float(test_metrics["precision"]),
            "recall_test": float(test_metrics["recall"]),
            "f1_test": float(test_metrics["f1"]),
            "roc_auc_test": float(test_metrics["roc_auc"]),
            "log_loss_test": float(test_metrics["log_loss"]),
            "ece_test": float(test_metrics["ece"]),
        })
    return rows


def _ensemble_metrics_row(val_member_probas, test_member_probas, y_val, y_test, n_members: int) -> tuple[dict, dict, dict]:
    ensemble_val_proba = average_proba(val_member_probas)
    ensemble_val_metrics = metrics_block(y_val, ensemble_val_proba)

    ensemble_test_proba = average_proba(test_member_probas)
    ensemble_test_metrics = metrics_block(y_test, ensemble_test_proba)

    row = {
        "model": "bayes_logreg",
        "type": "ensemble",
        "member_id": "ensemble",
        "n_members": n_members,
        # Validation
        "accuracy_val": float(ensemble_val_metrics["accuracy"]),
        "precision_val": float(ensemble_val_metrics["precision"]),
        "recall_val": float(ensemble_val_metrics["recall"]),
        "f1_val": float(ensemble_val_metrics["f1"]),
        "roc_auc_val": float(ensemble_val_metrics["roc_auc"]),
        "log_loss_val": float(ensemble_val_metrics["log_loss"]),
        "ece_val": float(ensemble_val_metrics["ece"]),
        # Test
        "accuracy_test": float(ensemble_test_metrics["accuracy"]),
        "precision_test": float(ensemble_test_metrics["precision"]),
        "recall_test": float(ensemble_test_metrics["recall"]),
        "f1_test": float(ensemble_test_metrics["f1"]),
        "roc_auc_test": float(ensemble_test_metrics["roc_auc"]),
        "log_loss_test": float(ensemble_test_metrics["log_loss"]),
        "ece_test": float(ensemble_test_metrics["ece"]),
    }
    return row, ensemble_val_metrics, ensemble_test_metrics




def _save_summary_json(summary_dir: Path, exp_name: str, idata, n_members: int, ensemble_val_metrics: dict, ensemble_test_metrics: dict, cfg: dict, member_indices: list[int] | None = None, strategy: dict | None = None) -> dict:
    summary = {
        "experiment_name": exp_name,
        "n_draws_used": int(idata.posterior["intercept"].sizes["chain"] * idata.posterior["intercept"].sizes["draw"]),
        "n_members": n_members,
        "val_metrics": ensemble_val_metrics,
        "test_metrics": ensemble_test_metrics,
        "bayes_config": cfg.get("bayes", {}).get("bayes_logreg", {}),
    }
    if strategy:
        summary["strategy_overrides"] = strategy
    if member_indices is not None:
        summary["selected_member_indices"] = list(map(int, member_indices))
    with open(summary_dir / "ensemble_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    return summary

def run_ensemble_nuts(
    config_path: Path | None = None,
    exp_name: str | None = None,
    strategy_name: str | None = None,
) -> Dict:
    """
    Run NUTS ensemble analysis for Bayesian Logistic Regression.
    """
    cfg, _ = _load_config(config_path)
    strategies = _load_experiments_cfg()
    strategy = None
    if strategy_name and strategy_name in strategies:
        s = strategies[strategy_name]
        strategy = s.get("bayes", {}).get("bayes_logreg", {})
        exp_name = s.get("output", {}).get("experiment_name", exp_name)
    # Determine output dirs after resolving exp_name
    out_dir = Path(cfg["output"]["results_dir"]) 
    exp_name = exp_name or cfg["output"].get("experiment_name", "ensemble")
    nuts_dir = _ensure_nuts_dirs(out_dir, exp_name)
    summary_dir = nuts_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("nuts_ensemble")
    X_train, X_val, X_test, y_train, y_val, y_test = _load_and_split_data(cfg)
    logger.info(f"Shapes | train:{X_train.shape} val:{X_val.shape} test:{X_test.shape}")
    
    # Fit Bayesian Logistic Regression with NUTS
    logger.info("Fitting Bayesian Logistic Regression with NUTS...")
    idata, member_indices, val_member_probas, test_member_probas = _fit_and_predict(cfg, X_train, y_train, X_val, y_val, X_test, y_test, strategy=strategy)
    logger.info("NUTS sampling completed successfully")

    save_member_proba_tables(val_member_probas, test_member_probas, nuts_dir)
    
    n_members = len(val_member_probas)
    logger.info(f"Generated {n_members} ensemble members")
    
    # Output directory already created above
    
    # Compute metrics for each member and aggregate into one CSV row per member (val & test)
    combined_rows: list[dict] = _member_metrics_rows(val_member_probas, test_member_probas, y_val, y_test)
    
    # Compute ensemble metrics (average of members) and ECE
    logger.info("Computing ensemble metrics...")
    
    # Ensemble row
    ensemble_row, ensemble_val_metrics, ensemble_test_metrics = _ensemble_metrics_row(
        val_member_probas, test_member_probas, y_val, y_test, n_members
    )
    combined_rows.append(ensemble_row)

    # Write combined metrics CSV
    combined_df = pd.DataFrame(combined_rows)
    combined_csv_path = summary_dir / "members_and_ensemble_metrics.csv"
    combined_df.to_csv(combined_csv_path, index=False)

    # Diversity and correlation exports
    diversity, cond_err_corr = export_diversity_and_correlation(
        val_member_probas, test_member_probas, y_val, y_test, nuts_dir, summary_dir
    )

    # Summarized correlation export (compact file for quick reading)
    export_correlation_summary(diversity, cond_err_corr, summary_dir)
    write_correlation_formulas(summary_dir)
    
    # Save summary JSON
    result_summary = _save_summary_json(summary_dir, exp_name or cfg["output"]["experiment_name"], idata, n_members, ensemble_val_metrics, ensemble_test_metrics, cfg, member_indices=member_indices, strategy=strategy)
    
    # Log results
    logger.info(f"Ensemble validation F1: {ensemble_val_metrics['f1']:.4f}")
    logger.info(f"Ensemble test F1: {ensemble_test_metrics['f1']:.4f}")
    logger.info(f"NUTS ensemble analysis complete. Results saved to {nuts_dir}")
    
    return result_summary


def run_all_strategies(config_path: Path | None = None) -> list[Dict]:
    """Iterate all strategies from ensemble_experiments.yaml and run them sequentially."""
    strategies = _load_experiments_cfg()
    results = []
    for name in strategies.keys():
        print(f"\n=== Running strategy: {name} ===")
        res = run_ensemble_nuts(config_path=config_path, exp_name=None, strategy_name=name)
        results.append({"strategy": name, **res})
    # Aggregate summary across experiments for quick comparison
    if results:
        cfg, _ = _load_config(config_path)
        base_dir = Path(cfg["output"]["results_dir"]) / "ensemble_nuts"
        agg_dir = base_dir / "experiments"
        agg_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in results:
            tm = r.get("test_metrics", {}) or {}
            vm = r.get("val_metrics", {}) or {}
            rows.append({
                "strategy": r.get("strategy"),
                "experiment_name": r.get("experiment_name"),
                "n_members": r.get("n_members"),
                # quick picks
                "f1_test": tm.get("f1"),
                "accuracy_test": tm.get("accuracy"),
                "roc_auc_test": tm.get("roc_auc"),
                "log_loss_test": tm.get("log_loss"),
                "ece_test": tm.get("ece"),
                "f1_val": vm.get("f1"),
                "accuracy_val": vm.get("accuracy"),
                "roc_auc_val": vm.get("roc_auc"),
                "log_loss_val": vm.get("log_loss"),
                "ece_val": vm.get("ece"),
            })
        pd.DataFrame(rows).to_csv(agg_dir / "experiments_summary.csv", index=False)
    return results


if __name__ == "__main__":
    # Change to run_all_strategies() to execute all presets
    run_ensemble_nuts()
