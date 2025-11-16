"""Correlation analysis between spectrum occupancy metrics."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def compute_pearson_correlation(x, y):
    """
    Compute Pearson correlation coefficient and p-value.

    Parameters
    ----------
    x : array-like
        First variable
    y : array-like
        Second variable

    Returns
    -------
    tuple of float
        (correlation_coefficient, p_value)

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> r, p = compute_pearson_correlation(x, y)
    >>> abs(r - 1.0) < 1e-10  # Perfect positive correlation
    True
    """
    x = np.array(x)
    y = np.array(y)

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return np.nan, np.nan

    return pearsonr(x_clean, y_clean)


def compute_correlation_matrix(data_dict):
    """
    Compute correlation matrix for multiple variables.

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping variable names to arrays

    Returns
    -------
    pd.DataFrame
        Correlation matrix with variable names as index/columns

    Examples
    --------
    >>> data = {
    ...     'var1': np.array([1, 2, 3, 4]),
    ...     'var2': np.array([2, 4, 6, 8]),
    ...     'var3': np.array([1, 3, 5, 7])
    ... }
    >>> corr_matrix = compute_correlation_matrix(data)
    >>> corr_matrix.loc['var1', 'var2']  # High correlation
    1.0
    """
    variables = list(data_dict.keys())
    n = len(variables)

    # Initialize correlation matrix
    corr_matrix = np.zeros((n, n))

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                r, _ = compute_pearson_correlation(data_dict[var1],
                                                     data_dict[var2])
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r

    return pd.DataFrame(corr_matrix, index=variables, columns=variables)


def analyze_metric_correlations(rssi, variance, duty_cycle):
    """
    Analyze correlations between occupancy metrics.

    Generates data for Table III in the paper.

    Parameters
    ----------
    rssi : array-like
        Mean power (RSSI) values in dB
    variance : array-like
        Signal variation values in dB²
    duty_cycle : array-like
        Duty cycle values as percentages

    Returns
    -------
    dict
        Dictionary of correlation results:
        {
            'variance_mean_power': (r, p),
            'variance_duty_cycle': (r, p),
            'duty_cycle_mean_power': (r, p)
        }

    Examples
    --------
    >>> rssi = np.array([-85, -95, -96, -101, -93])
    >>> variance = np.array([21, 18, 26, 10, 23])
    >>> duty_cycle = np.array([45, 99, 96, 37, 45])
    >>> correlations = analyze_metric_correlations(rssi, variance, duty_cycle)
    >>> 'variance_mean_power' in correlations
    True
    """
    results = {}

    # Variance vs Mean Power
    r_var_power, p_var_power = compute_pearson_correlation(variance, rssi)
    results['variance_mean_power'] = {
        'r': r_var_power,
        'p_value': p_var_power
    }

    # Variance vs Duty Cycle
    r_var_dc, p_var_dc = compute_pearson_correlation(variance, duty_cycle)
    results['variance_duty_cycle'] = {
        'r': r_var_dc,
        'p_value': p_var_dc
    }

    # Duty Cycle vs Mean Power
    r_dc_power, p_dc_power = compute_pearson_correlation(duty_cycle, rssi)
    results['duty_cycle_mean_power'] = {
        'r': r_dc_power,
        'p_value': p_dc_power
    }

    return results


def create_correlation_table(band_correlations):
    """
    Create formatted correlation table for multiple frequency bands.

    Formats results for Table III in the paper.

    Parameters
    ----------
    band_correlations : dict
        Dictionary mapping band names to correlation dictionaries

    Returns
    -------
    pd.DataFrame
        Formatted correlation table

    Examples
    --------
    >>> correlations = {
    ...     '2160-2170': {
    ...         'variance_mean_power': {'r': -0.76, 'p_value': 0.01},
    ...         'variance_duty_cycle': {'r': 0.16, 'p_value': 0.65},
    ...         'duty_cycle_mean_power': {'r': 0.00, 'p_value': 0.99}
    ...     }
    ... }
    >>> table = create_correlation_table(correlations)
    >>> 'Variance-Mean Power' in table.index
    True
    """
    rows = []

    for band_name, corr_dict in band_correlations.items():
        rows.append({
            'Frequency Band': band_name,
            'Pairs of Metrics': 'Variance-Mean Power',
            'Pearson Correlation': f"{corr_dict['variance_mean_power']['r']:.2f}"
        })
        rows.append({
            'Frequency Band': band_name,
            'Pairs of Metrics': 'Variance-Duty Cycle',
            'Pearson Correlation': f"{corr_dict['variance_duty_cycle']['r']:.2f}"
        })
        rows.append({
            'Frequency Band': band_name,
            'Pairs of Metrics': 'Duty Cycle-Mean Power',
            'Pearson Correlation': f"{corr_dict['duty_cycle_mean_power']['r']:.2f}"
        })

    return pd.DataFrame(rows)
