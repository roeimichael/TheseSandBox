"""
main.py
Main orchestrator for baseline and ensemble analysis.
Currently configured to run the BNN (MLP, SVI) ensemble. The NUTS (logreg) ensemble is commented out.
"""
from __future__ import annotations
from scripts.run_baseline import run_baseline
from scripts.run_ensemble_bnn import run_ensemble_bnn
# from scripts.run_ensemble_nuts import run_all_strategies  # NUTS (logreg) runner


def main():
    print("=" * 60)
    print("Running Baseline Analysis...")
    print("=" * 60)
    baseline_results = run_baseline()
    
    print("\n" + "=" * 60)
    print("Running BNN (MLP, SVI) Ensemble...")
    print("=" * 60)
    bnn_result = run_ensemble_bnn()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Baseline best model: {baseline_results['best_model']}")
    
    if bnn_result and (bnn_result.get('test_metrics') is not None):
        bnn_f1 = (bnn_result.get('test_metrics') or {}).get('f1')
        print(f"BNN experiment: {bnn_result.get('experiment_name')} with F1={bnn_f1:.4f}")
        print(f"Baseline best test F1: {baseline_results['test_scores']['f1']:.4f}")
        if bnn_f1 is not None and bnn_f1 > baseline_results['test_scores']['f1']:
            print("✅ BNN ensemble outperforms baseline!")
        else:
            print("📊 BNN ensemble on par with baseline")
    else:
        print("⚠️  BNN ensemble analysis failed or was disabled")


if __name__ == "__main__":
    main()
