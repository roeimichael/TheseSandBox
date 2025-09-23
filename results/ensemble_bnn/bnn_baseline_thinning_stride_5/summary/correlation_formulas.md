# Correlation and Diversity Metrics

## Pairwise probability correlation
Pearson correlation between two members’ positive-class probabilities across samples.\
Let p_i^A and p_i^B be the probabilities for sample i (i=1..N):\
r = cov(p^A, p^B) / (std(p^A) * std(p^B)) = \frac{\sum_i (p_i^A - \bar p^A)(p_i^B - \bar p^B)}{\sqrt{\sum_i (p_i^A - \bar p^A)^2} \sqrt{\sum_i (p_i^B - \bar p^B)^2}}

## Disagreement
Fraction of samples where hard predictions differ:\
D = (1/N) \sum_i 1[\hat y_i^A \neq \hat y_i^B]

## Q-statistic
Based on joint correctness counts: N11 (both correct), N00 (both incorrect), N10 (A correct, B incorrect), N01 (A incorrect, B correct).\
Q = (N11 \cdot N00 - N10 \cdot N01) / (N11 \cdot N00 + N10 \cdot N01)

## Error correlation (overall and conditional)
Let E_i^A = 1[\hat y_i^A \neq y_i], E_i^B = 1[\hat y_i^B \neq y_i].\
Error correlation is the Pearson correlation (phi coefficient) between E^A and E^B (overall or restricted to y=c):\
corr(E^A, E^B) = \frac{\operatorname{cov}(E^A, E^B)}{\sqrt{\operatorname{var}(E^A)} \sqrt{\operatorname{var}(E^B)}}
