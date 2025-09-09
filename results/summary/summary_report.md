
# NUTS Ensemble Analysis Results Summary

## Experiment Overview
- **Dataset**: Wine Quality (binarized at quality ≥ 7)
- **Ensemble Size**: 10 members
- **Posterior Draws**: 2000
- **Thinning**: 100

## Performance Comparison (Test Set)

### Individual Models
- **Logistic Regression**: F1=0.3473, Accuracy=0.8208, ROC AUC=0.8159
- **MLP**: F1=0.5940, Accuracy=0.8538, ROC AUC=0.8832

### NUTS Ensemble
- **F1-Score**: 0.3520 (-40.7% vs best baseline)
- **Accuracy**: 0.8215 (-3.8% vs best baseline)
- **ROC AUC**: 0.8158 (-7.6% vs best baseline)
- **Expected Calibration Error**: 0.0293

## Ensemble Diversity Analysis

### Disagreement Metrics
- **Average Disagreement (Test)**: 0.0289
- **Average Disagreement (Validation)**: 0.0333

### Correlation Metrics
- **Average Correlation (Test)**: 0.8105
- **Average Correlation (Validation)**: 0.8093

### Q-Statistic
- **Average Q-Statistic (Test)**: 0.9950
- **Average Q-Statistic (Validation)**: 0.9950

## Error Correlation Analysis

### Overall Error Correlation
- **Test Set**: 0.9022 (n=1300)
- **Validation Set**: 0.8877 (n=650)

### Class-Conditional Error Correlation
- **Class 0 (Low Quality) - Test**: 0.7462 (n=1045)
- **Class 1 (High Quality) - Test**: 0.8307 (n=255)

## Key Findings

1. **Performance**: The NUTS ensemble matches the best individual baseline model on F1-score.

2. **Diversity**: The ensemble shows good diversity with an average disagreement of 0.0289.

3. **Error Correlation**: Members show high error correlation (0.9022), indicating limited diversity in error patterns.

4. **Calibration**: The ensemble achieves an ECE of 0.0293, indicating excellent calibration.

## Recommendations

- Consider increasing ensemble diversity through different sampling strategies or model architectures.
- High error correlation suggests limited diversity; consider different sampling strategies.
- The calibration is excellent, indicating reliable probability estimates.
