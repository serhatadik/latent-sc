"""
Metrics for evaluating localization performance.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def extract_locations_from_map(transmit_power_map, threshold=1e-10):
    """
    Extract transmitter locations from a sparse power map.

    Parameters
    ----------
    transmit_power_map : ndarray
        2D array of transmit powers.
    threshold : float, optional
        Minimum power threshold to consider a location active. Default is 1e-10.

    Returns
    -------
    ndarray
        (N, 2) array of [col, row] coordinates for active locations.
    """
    rows, cols = np.nonzero(transmit_power_map > threshold)
    # Return as (col, row) to match (x, y) convention usually used in this project
    return np.column_stack((cols, rows))

def compute_localization_metrics(true_locations, estimated_locations, scale=1.0, tolerance=200.0):
    """
    Compute localization performance metrics.

    Parameters
    ----------
    true_locations : ndarray
        (N, 2) array of true transmitter coordinates [col, row] or [x, y].
    estimated_locations : ndarray
        (M, 2) array of estimated transmitter coordinates [col, row] or [x, y].
    scale : float, optional
        Scaling factor to convert pixel/grid units to meters. Default is 1.0.
    tolerance : float, optional
        Distance threshold (in meters) for a correct detection. Default is 200.0.

    Returns
    -------
    dict
        Dictionary containing:
        - 'ale': Average Localization Error (meters) for matched pairs
        - 'n_true': Number of true transmitters
        - 'n_est': Number of estimated transmitters
        - 'tp': True Positives
        - 'fp': False Positives
        - 'fn': False Negatives
        - 'pd': Probability of Detection (Recall)
        - 'precision': Precision
        - 'far': False Alarm Rate (1 - Precision)
        - 'f1_score': F1 Score
    """
    n_true = len(true_locations)
    n_est = len(estimated_locations)
    
    metrics = {
        'n_true': n_true,
        'n_est': n_est,
        'ale': np.nan,
        'tp': 0,
        'fp': n_est,
        'fn': n_true,
        'pd': 0.0,
        'precision': 0.0,
        'far': 0.0,
        'f1_score': 0.0
    }

    if n_true == 0:
        if n_est > 0:
            metrics['far'] = 1.0
        return metrics

    if n_est == 0:
        return metrics

    # Compute distance matrix (in meters)
    # cdist computes distance between each pair
    # true_locations and estimated_locations should be in the same units (pixels)
    # We multiply by scale to get meters
    dists_pixels = cdist(true_locations, estimated_locations, metric='euclidean')
    dists_meters = dists_pixels * scale

    # 1. Compute ALE using Hungarian algorithm (linear_sum_assignment)
    # This finds the optimal 1-to-1 matching that minimizes total distance
    # Note: This forces a matching even if distances are large, which is standard for ALE
    # but we only match up to min(n_true, n_est)
    row_ind, col_ind = linear_sum_assignment(dists_meters)
    matched_distances = dists_meters[row_ind, col_ind]
    metrics['ale'] = np.mean(matched_distances)

    # 2. Compute Detection Metrics (TP, FP, FN) based on tolerance
    # A true transmitter is "detected" if there is at least one estimate within tolerance.
    # However, one estimate cannot detect multiple true transmitters (usually).
    # And one true transmitter cannot be detected by multiple estimates (double counting).
    # The Hungarian matching above gives us a unique pairing. We can use that.
    
    # Count matches within tolerance
    tp_count = np.sum(matched_distances <= tolerance)
    
    # Update metrics
    metrics['tp'] = int(tp_count)
    metrics['fp'] = int(n_est - tp_count)
    metrics['fn'] = int(n_true - tp_count)
    
    # Derived metrics
    if n_true > 0:
        metrics['pd'] = metrics['tp'] / n_true
        
    if n_est > 0:
        metrics['precision'] = metrics['tp'] / n_est
        metrics['far'] = metrics['fp'] / n_est
        
    if metrics['precision'] + metrics['pd'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['pd']) / (metrics['precision'] + metrics['pd'])

    return metrics
