"""
main.py
Main orchestrator for baseline and NUTS ensemble analysis.
"""
from __future__ import annotations
from scripts.run_baseline import run_baseline
from scripts.run_ensemble_nuts import run_all_strategies


def main():
    print("=" * 60)
    print("Running Baseline Analysis...")
    print("=" * 60)
    baseline_results = run_baseline()
    
    print("\n" + "=" * 60)
    print("Running NUTS Ensemble Experiments...")
    print("=" * 60)
    nuts_results_list = run_all_strategies()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Baseline best model: {baseline_results['best_model']}")
    
    if nuts_results_list:
        # Show top experiment by test F1 if available
        best = max(
            nuts_results_list,
            key=lambda r: (r.get('test_metrics') or {}).get('f1', float('-inf'))
        )
        print(f"Best NUTS experiment: {best.get('experiment_name')} with F1={best.get('test_metrics',{}).get('f1'):.4f}")
        print(f"Baseline best test F1: {baseline_results['test_scores']['f1']:.4f}")
        if (best.get('test_metrics') or {}).get('f1', 0.0) > baseline_results['test_scores']['f1']:
            print("✅ Best NUTS experiment outperforms baseline!")
        else:
            print("📊 Best NUTS experiment on par with baseline")
    else:
        print("⚠️  NUTS ensemble analysis failed or was disabled")


if __name__ == "__main__":
    main()
