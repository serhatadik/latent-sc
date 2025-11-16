"""Analysis modules for correlation and regression."""

from .correlation import (
    compute_pearson_correlation,
    compute_correlation_matrix,
    analyze_metric_correlations
)
from .regression import (
    train_variance_regression,
    predict_variance_map,
    evaluate_regression
)

__all__ = [
    'compute_pearson_correlation',
    'compute_correlation_matrix',
    'analyze_metric_correlations',
    'train_variance_regression',
    'predict_variance_map',
    'evaluate_regression'
]
