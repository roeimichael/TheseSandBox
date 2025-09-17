"""
Build a compact correlation/diversity summary from existing artifacts.
Reads diversity.json and conditional_error_correlation.json and writes:
- results/ensemble_nuts/<exp>/summary/correlation_summary.csv
- results/ensemble_nuts/<exp>/summary/correlation_formulas.md
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from typing import Tuple
from evaluation.metrics import (
    ensemble_diversity_summary,
)
from evaluation.conditional_correlation import error_correlation_summary

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "ensemble_config.yaml"


def export_diversity_and_correlation(
    val_member_probas,
    test_member_probas,
    y_val,
    y_test,
    nuts_dir: Path,
    summary_dir: Path,
) -> Tuple[dict, dict]:
    """Compute and write diversity.json, conditional_error_correlation.json, and pairwise CSVs."""
    # Build predictions dicts for diversity/error correlation
    val_member_preds = {f"member_{i+1:03d}": np.argmax(p, axis=1) for i, p in enumerate(val_member_probas)}
    test_member_preds = {f"member_{i+1:03d}": np.argmax(p, axis=1) for i, p in enumerate(test_member_probas)}

    diversity = {
        "val": ensemble_diversity_summary(val_member_preds),
        "test": ensemble_diversity_summary(test_member_preds),
    }
    with open(nuts_dir / "diversity.json", "w") as f:
        json.dump(diversity, f, indent=2)

    # Helper to dump pairwise CSVs (validation split for quick glance)
    def _write_pairwise_csv(name: str, pairwise_map: dict):
        rows = []
        for k, v in pairwise_map.items():
            if isinstance(k, str) and "_vs_" in k:
                a, b = k.split("_vs_", 1)
            else:
                a, b = str(k), ""
            rows.append({"model_a": a, "model_b": b, name: v})
        pd.DataFrame(rows).to_csv(summary_dir / f"pairwise_{name}.csv", index=False)

    _write_pairwise_csv("disagreement", diversity["val"]["pairwise_disagreements"])  # val
    _write_pairwise_csv("q_stat", diversity["val"]["pairwise_q_stats"])            # val
    _write_pairwise_csv("correlation", diversity["val"]["pairwise_correlations"])  # val

    cond_err_corr = {
        "val": error_correlation_summary(y_val.values if hasattr(y_val, 'values') else y_val, val_member_preds),
        "test": error_correlation_summary(y_test.values if hasattr(y_test, 'values') else y_test, test_member_preds),
    }
    with open(nuts_dir / "conditional_error_correlation.json", "w") as f:
        json.dump(cond_err_corr, f, indent=2)

    # Write conditional error correlation pairwise CSVs for validation
    def _cond_to_df(section: dict) -> pd.DataFrame:
        rows = []
        for k, v in section.get("pairwise", {}).items():
            if isinstance(k, str) and "_vs_" in k:
                a, b = k.split("_vs_", 1)
            else:
                a, b = str(k), ""
            rows.append({"model_a": a, "model_b": b, "corr": v})
        return pd.DataFrame(rows)

    _cond_to_df(cond_err_corr["val"]["overall"]).to_csv(summary_dir / "pairwise_conditional_error_corr_overall.csv", index=False)
    _cond_to_df(cond_err_corr["val"]["y=0"]).to_csv(summary_dir / "pairwise_conditional_error_corr_y0.csv", index=False)
    _cond_to_df(cond_err_corr["val"]["y=1"]).to_csv(summary_dir / "pairwise_conditional_error_corr_y1.csv", index=False)

    return diversity, cond_err_corr


def export_correlation_summary(diversity: dict, cond_err_corr: dict, summary_dir: Path) -> Path:
    rows = []
    for split in ("val", "test"):
        div = diversity.get(split, {})
        avg_proba_corr = div.get("avg_correlation")
        avg_disagreement = div.get("avg_disagreement")
        avg_q = div.get("avg_q_statistic")

        cond = cond_err_corr.get(split, {})
        overall = cond.get("overall", {})
        y0 = cond.get("y=0", {})
        y1 = cond.get("y=1", {})

        rows.append({
            "split": split,
            "avg_pairwise_proba_corr": overall.get("avg_correlation") if avg_proba_corr is None else avg_proba_corr,
            "avg_disagreement": avg_disagreement,
            "avg_q_statistic": avg_q,
            "avg_error_corr_overall": overall.get("avg_correlation"),
            "avg_error_corr_y0": y0.get("avg_correlation"),
            "avg_error_corr_y1": y1.get("avg_correlation"),
            "n_samples_overall": overall.get("num_samples"),
            "n_samples_y0": y0.get("num_samples"),
            "n_samples_y1": y1.get("num_samples"),
        })

    df = pd.DataFrame(rows)
    out_path = summary_dir / "correlation_summary.csv"
    df.to_csv(out_path, index=False)
    return out_path


def write_correlation_formulas(summary_dir: Path) -> Path:
    md = []
    md.append("# Correlation and Diversity Metrics\n")
    md.append("\n## Pairwise probability correlation\n")
    md.append("Pearson correlation between two members’ positive-class probabilities across samples.\\\n")
    md.append("Let p_i^A and p_i^B be the probabilities for sample i (i=1..N):\\\n")
    md.append("r = cov(p^A, p^B) / (std(p^A) * std(p^B)) = ")
    md.append("\\frac{\\sum_i (p_i^A - \\bar p^A)(p_i^B - \\bar p^B)}{\\sqrt{\\sum_i (p_i^A - \\bar p^A)^2} \\sqrt{\\sum_i (p_i^B - \\bar p^B)^2}}\n")

    md.append("\n## Disagreement\n")
    md.append("Fraction of samples where hard predictions differ:\\\n")
    md.append("D = (1/N) \\sum_i 1[\\hat y_i^A \\neq \\hat y_i^B]\n")

    md.append("\n## Q-statistic\n")
    md.append("Based on joint correctness counts: N11 (both correct), N00 (both incorrect), N10 (A correct, B incorrect), N01 (A incorrect, B correct).\\\n")
    md.append("Q = (N11 \\cdot N00 - N10 \\cdot N01) / (N11 \\cdot N00 + N10 \\cdot N01)\n")

    md.append("\n## Error correlation (overall and conditional)\n")
    md.append("Let E_i^A = 1[\\hat y_i^A \\neq y_i], E_i^B = 1[\\hat y_i^B \\neq y_i].\\\n")
    md.append("Error correlation is the Pearson correlation (phi coefficient) between E^A and E^B (overall or restricted to y=c):\\\n")
    md.append("corr(E^A, E^B) = ")
    md.append("\\frac{\\operatorname{cov}(E^A, E^B)}{\\sqrt{\\operatorname{var}(E^A)} \\sqrt{\\operatorname{var}(E^B)}}\n")

    content = "".join(md)
    out_path = summary_dir / "correlation_formulas.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


def main(exp_name: str | None = None, config_path: str | None = None):
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    exp = exp_name or cfg["output"]["experiment_name"]

    base = Path(cfg["output"]["results_dir"]) / "ensemble_nuts" / exp
    summary_dir = base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(base / "diversity.json", "r") as f:
        diversity = json.load(f)
    with open(base / "conditional_error_correlation.json", "r") as f:
        cond_err_corr = json.load(f)

    csv_path = export_correlation_summary(diversity, cond_err_corr, summary_dir)
    md_path = write_correlation_formulas(summary_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
