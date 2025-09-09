"""
main.py
Main orchestrator for baseline and NUTS ensemble analysis.
"""
from __future__ import annotations
from scripts.run_baseline import run_baseline
from scripts.run_ensemble_nuts import run_ensemble_nuts


def main():
    print("=" * 60)
    print("Running Baseline Analysis...")
    print("=" * 60)
    baseline_results = run_baseline()
    
    print("\n" + "=" * 60)
    print("Running NUTS Ensemble Analysis...")
    print("=" * 60)
    nuts_results = run_ensemble_nuts()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Baseline best model: {baseline_results['best_model']}")
    
    if nuts_results:
        print(f"NUTS ensemble members: {nuts_results['n_members']}")
        print(f"NUTS ensemble test F1: {nuts_results['test_metrics']['f1']:.4f}")
        print(f"Baseline best test F1: {baseline_results['test_scores']['f1']:.4f}")
        
        # Quick comparison
        if nuts_results['test_metrics']['f1'] > baseline_results['test_scores']['f1']:
            print("✅ NUTS ensemble outperforms baseline!")
        else:
            print("📊 NUTS ensemble performance on par with baseline")
    else:
        print("⚠️  NUTS ensemble analysis failed or was disabled")


if __name__ == "__main__":
    main()
