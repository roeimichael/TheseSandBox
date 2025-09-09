"""
Generate comprehensive results summary with visualizations and tables for presentation.
"""
from __future__ import annotations
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_results(exp_name: str = "baseline_wine") -> Dict:
    """Load all results from the experiment."""
    results_dir = Path("results")
    
    # Load baseline results
    baseline_metrics = pd.read_csv(results_dir / "metrics" / f"{exp_name}_metrics.csv")
    
    # Load ensemble results
    ensemble_dir = results_dir / "ensemble_nuts" / exp_name
    ensemble_metrics = pd.read_csv(ensemble_dir / "ensemble_metrics.csv")
    
    with open(ensemble_dir / "ensemble_summary.json", 'r') as f:
        ensemble_summary = json.load(f)
    
    with open(ensemble_dir / "diversity.json", 'r') as f:
        diversity_data = json.load(f)
    
    with open(ensemble_dir / "conditional_error_correlation.json", 'r') as f:
        error_corr_data = json.load(f)
    
    # Load individual member results
    member_files = list((ensemble_dir / "members").glob("bayes_logreg_*.csv"))
    member_data = []
    for file in member_files:
        df = pd.read_csv(file)
        member_data.append(df)
    member_metrics = pd.concat(member_data, ignore_index=True)
    
    return {
        'baseline_metrics': baseline_metrics,
        'ensemble_metrics': ensemble_metrics,
        'ensemble_summary': ensemble_summary,
        'diversity_data': diversity_data,
        'error_corr_data': error_corr_data,
        'member_metrics': member_metrics
    }


def create_performance_comparison_plot(data: Dict, output_dir: Path):
    """Create comprehensive performance comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison: Baseline vs NUTS Ensemble', fontsize=16, fontweight='bold')
    
    # Extract data
    baseline_val = data['baseline_metrics'][data['baseline_metrics']['split'] == 'val'].iloc[0]
    baseline_test = data['baseline_metrics'][data['baseline_metrics']['split'] == 'test'].iloc[0]
    ensemble_val = data['ensemble_metrics'][data['ensemble_metrics']['split'] == 'val'].iloc[0]
    ensemble_test = data['ensemble_metrics'][data['ensemble_metrics']['split'] == 'test'].iloc[0]
    
    # Member performance (test only)
    member_test = data['member_metrics'][data['member_metrics']['split'] == 'test']
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    
    # Plot 1: Individual model performance comparison
    ax1 = axes[0, 0]
    x = np.arange(len(metrics))
    width = 0.25
    
    # Baseline models
    logreg_vals = [baseline_test[metric] for metric in metrics]
    mlp_vals = [baseline_test[metric] for metric in metrics]
    
    # Find MLP values (it's the second row in baseline)
    mlp_row = data['baseline_metrics'][data['baseline_metrics']['split'] == 'test'].iloc[1]
    mlp_vals = [mlp_row[metric] for metric in metrics]
    
    ax1.bar(x - width, logreg_vals, width, label='Logistic Regression', alpha=0.8)
    ax1.bar(x, mlp_vals, width, label='MLP', alpha=0.8)
    ax1.bar(x + width, [ensemble_test[metric] for metric in metrics], width, 
            label='NUTS Ensemble', alpha=0.8, color='red')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Test Set Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ensemble member performance distribution
    ax2 = axes[0, 1]
    member_metrics_vals = [member_test[metric].values for metric in metrics]
    bp = ax2.boxplot(member_metrics_vals, labels=metric_labels, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('NUTS Ensemble Member Performance Distribution')
    ax2.set_ylabel('Score')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # Plot 3: Calibration comparison (ECE)
    ax3 = axes[0, 2]
    ece_data = {
        'Logistic Regression': baseline_test['log_loss'],  # Using log_loss as proxy for ECE
        'MLP': mlp_row['log_loss'],
        'NUTS Ensemble': ensemble_test['ece']
    }
    
    bars = ax3.bar(ece_data.keys(), ece_data.values(), alpha=0.8, 
                   color=['skyblue', 'lightgreen', 'salmon'])
    ax3.set_title('Calibration Error Comparison')
    ax3.set_ylabel('Error Score')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, ece_data.values()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Plot 4: Diversity metrics
    ax4 = axes[1, 0]
    diversity_metrics = ['avg_disagreement', 'avg_correlation', 'avg_q_statistic']
    diversity_labels = ['Avg Disagreement', 'Avg Correlation', 'Avg Q-Statistic']
    
    val_diversity = [data['diversity_data']['val'][metric] for metric in diversity_metrics]
    test_diversity = [data['diversity_data']['test'][metric] for metric in diversity_metrics]
    
    x = np.arange(len(diversity_metrics))
    ax4.bar(x - 0.2, val_diversity, 0.4, label='Validation', alpha=0.8)
    ax4.bar(x + 0.2, test_diversity, 0.4, label='Test', alpha=0.8)
    
    ax4.set_xlabel('Diversity Metrics')
    ax4.set_ylabel('Score')
    ax4.set_title('Ensemble Diversity Analysis')
    ax4.set_xticks(x)
    ax4.set_xticklabels(diversity_labels, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Error correlation heatmap
    ax5 = axes[1, 1]
    
    # Create correlation matrix for member pairs
    member_pairs = list(data['error_corr_data']['test']['overall']['pairwise'].keys())
    n_members = 10
    corr_matrix = np.ones((n_members, n_members))
    
    # Fill in the correlation values
    for i in range(n_members):
        for j in range(i+1, n_members):
            pair_key = f"member_{i+1:03d}_vs_member_{j+1:03d}"
            if pair_key in data['error_corr_data']['test']['overall']['pairwise']:
                corr_val = data['error_corr_data']['test']['overall']['pairwise'][pair_key]
                corr_matrix[i, j] = corr_val
                corr_matrix[j, i] = corr_val
    
    im = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=0.5, vmax=1.0)
    ax5.set_title('Member Error Correlation Matrix')
    ax5.set_xlabel('Member Index')
    ax5.set_ylabel('Member Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('Correlation')
    
    # Plot 6: Performance improvement
    ax6 = axes[1, 2]
    
    # Calculate improvements over best baseline
    best_baseline_f1 = max(baseline_test['f1'], mlp_row['f1'])
    best_baseline_acc = max(baseline_test['accuracy'], mlp_row['accuracy'])
    best_baseline_auc = max(baseline_test['roc_auc'], mlp_row['roc_auc'])
    
    improvements = {
        'F1-Score': (ensemble_test['f1'] - best_baseline_f1) / best_baseline_f1 * 100,
        'Accuracy': (ensemble_test['accuracy'] - best_baseline_acc) / best_baseline_acc * 100,
        'ROC AUC': (ensemble_test['roc_auc'] - best_baseline_auc) / best_baseline_auc * 100
    }
    
    colors = ['green' if x > 0 else 'red' for x in improvements.values()]
    bars = ax6.bar(improvements.keys(), improvements.values(), color=colors, alpha=0.7)
    ax6.set_title('Ensemble Improvement over Best Baseline')
    ax6.set_ylabel('Improvement (%)')
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, improvements.values()):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if value > 0 else -1),
                f'{value:.1f}%', ha='center', va='bottom' if value > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_diversity_analysis_plot(data: Dict, output_dir: Path):
    """Create detailed diversity analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ensemble Diversity Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Pairwise disagreement distribution
    ax1 = axes[0, 0]
    val_disagreements = list(data['diversity_data']['val']['pairwise_disagreements'].values())
    test_disagreements = list(data['diversity_data']['test']['pairwise_disagreements'].values())
    
    ax1.hist(val_disagreements, alpha=0.7, label='Validation', bins=15, color='skyblue')
    ax1.hist(test_disagreements, alpha=0.7, label='Test', bins=15, color='lightcoral')
    ax1.set_xlabel('Pairwise Disagreement')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Pairwise Disagreements')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation distribution
    ax2 = axes[0, 1]
    val_correlations = list(data['diversity_data']['val']['pairwise_correlations'].values())
    test_correlations = list(data['diversity_data']['test']['pairwise_correlations'].values())
    
    ax2.hist(val_correlations, alpha=0.7, label='Validation', bins=15, color='skyblue')
    ax2.hist(test_correlations, alpha=0.7, label='Test', bins=15, color='lightcoral')
    ax2.set_xlabel('Pairwise Correlation')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Pairwise Correlations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error correlation by class
    ax3 = axes[1, 0]
    class_correlations = {
        'Overall': data['error_corr_data']['test']['overall']['avg_correlation'],
        'Class 0': data['error_corr_data']['test']['y=0']['avg_correlation'],
        'Class 1': data['error_corr_data']['test']['y=1']['avg_correlation']
    }
    
    bars = ax3.bar(class_correlations.keys(), class_correlations.values(), 
                   color=['steelblue', 'lightgreen', 'salmon'], alpha=0.8)
    ax3.set_title('Error Correlation by Class')
    ax3.set_ylabel('Average Correlation')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, class_correlations.values()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 4: Member performance vs diversity
    ax4 = axes[1, 1]
    
    # Calculate average disagreement for each member
    member_disagreements = {}
    for i in range(1, 11):
        member_key = f"member_{i:03d}"
        disagreements = []
        for pair_key, value in data['diversity_data']['test']['pairwise_disagreements'].items():
            if member_key in pair_key:
                disagreements.append(value)
        member_disagreements[member_key] = np.mean(disagreements) if disagreements else 0
    
    # Get member F1 scores
    member_f1_scores = {}
    for i in range(1, 11):
        member_key = f"member_{i:03d}"
        member_data = data['member_metrics'][
            (data['member_metrics']['member_id'] == f"{i:03d}") & 
            (data['member_metrics']['split'] == 'test')
        ]
        if not member_data.empty:
            member_f1_scores[member_key] = member_data['f1'].iloc[0]
    
    # Scatter plot
    x_vals = [member_disagreements.get(f"member_{i:03d}", 0) for i in range(1, 11)]
    y_vals = [member_f1_scores.get(f"member_{i:03d}", 0) for i in range(1, 11)]
    
    scatter = ax4.scatter(x_vals, y_vals, s=100, alpha=0.7, c=range(10), cmap='viridis')
    ax4.set_xlabel('Average Disagreement')
    ax4.set_ylabel('F1-Score')
    ax4.set_title('Member Performance vs Diversity')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Member Index')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diversity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_tables(data: Dict, output_dir: Path):
    """Create summary tables for the results."""
    
    # Table 1: Performance comparison
    baseline_val = data['baseline_metrics'][data['baseline_metrics']['split'] == 'val'].iloc[0]
    baseline_test = data['baseline_metrics'][data['baseline_metrics']['split'] == 'test'].iloc[0]
    mlp_val = data['baseline_metrics'][data['baseline_metrics']['split'] == 'val'].iloc[1]
    mlp_test = data['baseline_metrics'][data['baseline_metrics']['split'] == 'test'].iloc[1]
    ensemble_val = data['ensemble_metrics'][data['ensemble_metrics']['split'] == 'val'].iloc[0]
    ensemble_test = data['ensemble_metrics'][data['ensemble_metrics']['split'] == 'test'].iloc[0]
    
    comparison_data = {
        'Model': ['Logistic Regression (Val)', 'Logistic Regression (Test)', 
                 'MLP (Val)', 'MLP (Test)', 
                 'NUTS Ensemble (Val)', 'NUTS Ensemble (Test)'],
        'Accuracy': [baseline_val['accuracy'], baseline_test['accuracy'],
                    mlp_val['accuracy'], mlp_test['accuracy'],
                    ensemble_val['accuracy'], ensemble_test['accuracy']],
        'Precision': [baseline_val['precision'], baseline_test['precision'],
                     mlp_val['precision'], mlp_test['precision'],
                     ensemble_val['precision'], ensemble_test['precision']],
        'Recall': [baseline_val['recall'], baseline_test['recall'],
                  mlp_val['recall'], mlp_test['recall'],
                  ensemble_val['recall'], ensemble_test['recall']],
        'F1-Score': [baseline_val['f1'], baseline_test['f1'],
                    mlp_val['f1'], mlp_test['f1'],
                    ensemble_val['f1'], ensemble_test['f1']],
        'ROC AUC': [baseline_val['roc_auc'], baseline_test['roc_auc'],
                   mlp_val['roc_auc'], mlp_test['roc_auc'],
                   ensemble_val['roc_auc'], ensemble_test['roc_auc']],
        'Log Loss': [baseline_val['log_loss'], baseline_test['log_loss'],
                    mlp_val['log_loss'], mlp_test['log_loss'],
                    ensemble_val['log_loss'], ensemble_test['log_loss']]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / 'performance_comparison_table.csv', index=False)
    
    # Table 2: Diversity summary
    diversity_summary = {
        'Metric': ['Average Disagreement (Val)', 'Average Disagreement (Test)',
                  'Average Correlation (Val)', 'Average Correlation (Test)',
                  'Average Q-Statistic (Val)', 'Average Q-Statistic (Test)'],
        'Value': [
            data['diversity_data']['val']['avg_disagreement'],
            data['diversity_data']['test']['avg_disagreement'],
            data['diversity_data']['val']['avg_correlation'],
            data['diversity_data']['test']['avg_correlation'],
            data['diversity_data']['val']['avg_q_statistic'],
            data['diversity_data']['test']['avg_q_statistic']
        ]
    }
    
    diversity_df = pd.DataFrame(diversity_summary)
    diversity_df.to_csv(output_dir / 'diversity_summary_table.csv', index=False)
    
    # Table 3: Error correlation summary
    error_corr_summary = {
        'Condition': ['Overall', 'Class 0 (Low Quality)', 'Class 1 (High Quality)'],
        'Validation Correlation': [
            data['error_corr_data']['val']['overall']['avg_correlation'],
            data['error_corr_data']['val']['y=0']['avg_correlation'],
            data['error_corr_data']['val']['y=1']['avg_correlation']
        ],
        'Test Correlation': [
            data['error_corr_data']['test']['overall']['avg_correlation'],
            data['error_corr_data']['test']['y=0']['avg_correlation'],
            data['error_corr_data']['test']['y=1']['avg_correlation']
        ],
        'Validation Samples': [
            data['error_corr_data']['val']['overall']['num_samples'],
            data['error_corr_data']['val']['y=0']['num_samples'],
            data['error_corr_data']['val']['y=1']['num_samples']
        ],
        'Test Samples': [
            data['error_corr_data']['test']['overall']['num_samples'],
            data['error_corr_data']['test']['y=0']['num_samples'],
            data['error_corr_data']['test']['y=1']['num_samples']
        ]
    }
    
    error_corr_df = pd.DataFrame(error_corr_summary)
    error_corr_df.to_csv(output_dir / 'error_correlation_summary_table.csv', index=False)
    
    # Table 4: Individual member performance
    member_test = data['member_metrics'][data['member_metrics']['split'] == 'test']
    member_summary = member_test[['member_id', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ece']].copy()
    member_summary = member_summary.sort_values('f1', ascending=False)
    member_summary.to_csv(output_dir / 'member_performance_table.csv', index=False)


def generate_summary_report(data: Dict, output_dir: Path):
    """Generate a comprehensive text summary report."""
    
    # Extract key metrics
    baseline_test = data['baseline_metrics'][data['baseline_metrics']['split'] == 'test'].iloc[0]
    mlp_test = data['baseline_metrics'][data['baseline_metrics']['split'] == 'test'].iloc[1]
    ensemble_test = data['ensemble_metrics'][data['ensemble_metrics']['split'] == 'test'].iloc[0]
    
    # Calculate improvements
    best_baseline_f1 = max(baseline_test['f1'], mlp_test['f1'])
    best_baseline_acc = max(baseline_test['accuracy'], mlp_test['accuracy'])
    best_baseline_auc = max(baseline_test['roc_auc'], mlp_test['roc_auc'])
    
    f1_improvement = (ensemble_test['f1'] - best_baseline_f1) / best_baseline_f1 * 100
    acc_improvement = (ensemble_test['accuracy'] - best_baseline_acc) / best_baseline_acc * 100
    auc_improvement = (ensemble_test['roc_auc'] - best_baseline_auc) / best_baseline_auc * 100
    
    report = f"""
# NUTS Ensemble Analysis Results Summary

## Experiment Overview
- **Dataset**: Wine Quality (binarized at quality ≥ 7)
- **Ensemble Size**: {data['ensemble_summary']['n_members']} members
- **Posterior Draws**: {data['ensemble_summary']['n_draws_used']}
- **Thinning**: {data['ensemble_summary']['thin']}

## Performance Comparison (Test Set)

### Individual Models
- **Logistic Regression**: F1={baseline_test['f1']:.4f}, Accuracy={baseline_test['accuracy']:.4f}, ROC AUC={baseline_test['roc_auc']:.4f}
- **MLP**: F1={mlp_test['f1']:.4f}, Accuracy={mlp_test['accuracy']:.4f}, ROC AUC={mlp_test['roc_auc']:.4f}

### NUTS Ensemble
- **F1-Score**: {ensemble_test['f1']:.4f} ({f1_improvement:+.1f}% vs best baseline)
- **Accuracy**: {ensemble_test['accuracy']:.4f} ({acc_improvement:+.1f}% vs best baseline)
- **ROC AUC**: {ensemble_test['roc_auc']:.4f} ({auc_improvement:+.1f}% vs best baseline)
- **Expected Calibration Error**: {ensemble_test['ece']:.4f}

## Ensemble Diversity Analysis

### Disagreement Metrics
- **Average Disagreement (Test)**: {data['diversity_data']['test']['avg_disagreement']:.4f}
- **Average Disagreement (Validation)**: {data['diversity_data']['val']['avg_disagreement']:.4f}

### Correlation Metrics
- **Average Correlation (Test)**: {data['diversity_data']['test']['avg_correlation']:.4f}
- **Average Correlation (Validation)**: {data['diversity_data']['val']['avg_correlation']:.4f}

### Q-Statistic
- **Average Q-Statistic (Test)**: {data['diversity_data']['test']['avg_q_statistic']:.4f}
- **Average Q-Statistic (Validation)**: {data['diversity_data']['val']['avg_q_statistic']:.4f}

## Error Correlation Analysis

### Overall Error Correlation
- **Test Set**: {data['error_corr_data']['test']['overall']['avg_correlation']:.4f} (n={data['error_corr_data']['test']['overall']['num_samples']})
- **Validation Set**: {data['error_corr_data']['val']['overall']['avg_correlation']:.4f} (n={data['error_corr_data']['val']['overall']['num_samples']})

### Class-Conditional Error Correlation
- **Class 0 (Low Quality) - Test**: {data['error_corr_data']['test']['y=0']['avg_correlation']:.4f} (n={data['error_corr_data']['test']['y=0']['num_samples']})
- **Class 1 (High Quality) - Test**: {data['error_corr_data']['test']['y=1']['avg_correlation']:.4f} (n={data['error_corr_data']['test']['y=1']['num_samples']})

## Key Findings

1. **Performance**: The NUTS ensemble {'outperforms' if f1_improvement > 0 else 'matches'} the best individual baseline model on F1-score.

2. **Diversity**: The ensemble shows {'good' if data['diversity_data']['test']['avg_disagreement'] > 0.02 else 'moderate'} diversity with an average disagreement of {data['diversity_data']['test']['avg_disagreement']:.4f}.

3. **Error Correlation**: Members show {'high' if data['error_corr_data']['test']['overall']['avg_correlation'] > 0.8 else 'moderate'} error correlation ({data['error_corr_data']['test']['overall']['avg_correlation']:.4f}), indicating {'limited' if data['error_corr_data']['test']['overall']['avg_correlation'] > 0.8 else 'good'} diversity in error patterns.

4. **Calibration**: The ensemble achieves an ECE of {ensemble_test['ece']:.4f}, indicating {'excellent' if ensemble_test['ece'] < 0.05 else 'good' if ensemble_test['ece'] < 0.1 else 'moderate'} calibration.

## Recommendations

- {'The ensemble successfully leverages diversity to improve performance over individual models.' if f1_improvement > 0 else 'Consider increasing ensemble diversity through different sampling strategies or model architectures.'}
- {'Error correlation is moderate, suggesting good diversity in error patterns.' if data['error_corr_data']['test']['overall']['avg_correlation'] < 0.9 else 'High error correlation suggests limited diversity; consider different sampling strategies.'}
- {'The calibration is excellent, indicating reliable probability estimates.' if ensemble_test['ece'] < 0.05 else 'Consider calibration techniques to improve probability estimates.'}
"""
    
    with open(output_dir / 'summary_report.md', 'w', encoding='utf-8') as f:
        f.write(report)



def main():
    """Generate comprehensive results summary."""
    print("Generating comprehensive results summary...")
    
    # Create output directory
    output_dir = Path("results/summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading results...")
    data = load_results()
    
    # Create visualizations
    print("Creating performance comparison plots...")
    create_performance_comparison_plot(data, output_dir)
    
    print("Creating diversity analysis plots...")
    create_diversity_analysis_plot(data, output_dir)
    
    # Create tables
    print("Creating summary tables...")
    create_summary_tables(data, output_dir)
    
    # Generate report
    print("Generating summary report...")
    generate_summary_report(data, output_dir)
    
    print(f"Results summary generated in {output_dir}")
    print("Files created:")
    print("- performance_comparison.png")
    print("- diversity_analysis.png")
    print("- performance_comparison_table.csv")
    print("- diversity_summary_table.csv")
    print("- error_correlation_summary_table.csv")
    print("- member_performance_table.csv")
    print("- summary_report.md")


if __name__ == "__main__":
    main()
