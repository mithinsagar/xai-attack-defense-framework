# Explanation Drift Metric

**Author:** Mithin Sagar S ([github.com/mithinsagar](https://github.com/mithinsagar))

## Formal Definition

The Explanation Drift `D(x, epsilon)` is the mean absolute difference
between per-feature attribution vectors before and after perturbation:

```
D(x, epsilon) = (1 / d) * sum_i |E_i(x) - E_i(x + delta)|
```

where
- `d`           : number of features
- `E_i(x)`      : absolute attribution of feature `i` for input `x`
- `delta`       : adversarial perturbation with `||delta||_inf <= epsilon`

## Additional Drift Metrics

| Metric                | Definition                                             |
|-----------------------|--------------------------------------------------------|
| Max Drift             | `max_i |E_i(x) - E_i(x + delta)|`                      |
| Top-k Drift           | Mean drift over the k features with the largest drifts |
| Rank Correlation Drift| `1 - |Spearman(rank_before, rank_after)|` (higher = more manipulation) |

All metrics are implemented in `metrics/drift_metric.py`. The convenience
function `compute_all_drift_metrics(before, after, k=5)` returns a dict
with keys `Drift`, `MaxDrift`, `Top5Drift`, `RankCorrDrift`.
