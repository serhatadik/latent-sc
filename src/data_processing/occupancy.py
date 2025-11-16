"""Functions for calculating spectrum occupancy metrics."""

import numpy as np
import pandas as pd


def calculate_linear_threshold(freq, band_start, band_end,
                                threshold_start, threshold_end):
    """
    Calculate linearly varying threshold across frequency band.

    Parameters
    ----------
    freq : float or array-like
        Frequency in MHz
    band_start : float
        Start frequency of band in MHz
    band_end : float
        End frequency of band in MHz
    threshold_start : float
        Threshold value at band_start in dB
    threshold_end : float
        Threshold value at band_end in dB

    Returns
    -------
    float or ndarray
        Threshold value(s) in dB

    Examples
    --------
    >>> calculate_linear_threshold(3610, 3600, 3650, -105, -100)
    -104.0
    """
    if band_end == band_start:
        return threshold_start

    slope = (threshold_end - threshold_start) / (band_end - band_start)
    return threshold_start + slope * (freq - band_start)


def calculate_duty_cycle(df, threshold_column='threshold'):
    """
    Calculate duty cycle as percentage of time above threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'power' column and threshold information
    threshold_column : str, optional
        Name of column containing threshold values (default: 'threshold')

    Returns
    -------
    float
        Duty cycle as percentage (0-100)

    Examples
    --------
    >>> df = pd.DataFrame({'power': [-100, -90, -80], 'threshold': [-95, -95, -95]})
    >>> calculate_duty_cycle(df)
    66.66666666666666
    """
    if len(df) == 0:
        return 0.0

    if threshold_column in df.columns:
        occupied = df['power'] > df[threshold_column]
    else:
        raise ValueError(f"Column '{threshold_column}' not found in DataFrame")

    return (occupied.sum() / len(df)) * 100


def calculate_avg_power_occupied(df, threshold_column='threshold'):
    """
    Calculate average power when spectrum is occupied (above threshold).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'power' column and threshold information
    threshold_column : str, optional
        Name of column containing threshold values (default: 'threshold')

    Returns
    -------
    float
        Average power in dB when occupied, or NaN if never occupied

    Examples
    --------
    >>> df = pd.DataFrame({'power': [-100, -90, -80], 'threshold': [-95, -95, -95]})
    >>> calculate_avg_power_occupied(df)
    -85.0
    """
    if len(df) == 0:
        return np.nan

    if threshold_column in df.columns:
        occupied_df = df[df['power'] > df[threshold_column]]
    else:
        raise ValueError(f"Column '{threshold_column}' not found in DataFrame")

    if len(occupied_df) == 0:
        return np.nan

    return np.mean(occupied_df['power'])


def calculate_signal_variation(df, threshold_column='threshold'):
    """
    Calculate signal variation (variance) when spectrum is occupied.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'power' column and threshold information
    threshold_column : str, optional
        Name of column containing threshold values (default: 'threshold')

    Returns
    -------
    float
        Variance of power in dB² when occupied, or NaN if never occupied

    Examples
    --------
    >>> df = pd.DataFrame({'power': [-100, -90, -80], 'threshold': [-95, -95, -95]})
    >>> var = calculate_signal_variation(df)
    >>> abs(var - 25.0) < 0.1
    True
    """
    if len(df) == 0:
        return np.nan

    if threshold_column in df.columns:
        occupied_df = df[df['power'] > df[threshold_column]]
    else:
        raise ValueError(f"Column '{threshold_column}' not found in DataFrame")

    if len(occupied_df) == 0:
        return np.nan

    return np.var(occupied_df['power'])


def compute_occupancy_metrics(df, band_start, band_end,
                               threshold_start, threshold_end):
    """
    Compute all occupancy metrics for a frequency band.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'frequency' and 'power' columns
    band_start : float
        Start frequency of band in MHz
    band_end : float
        End frequency of band in MHz
    threshold_start : float
        Threshold at band_start in dB
    threshold_end : float
        Threshold at band_end in dB

    Returns
    -------
    dict
        Dictionary containing:
        - 'duty_cycle': Percentage (0-100)
        - 'avg_power_occupied': Mean power in dB
        - 'signal_variation': Variance in dB²
        - 'num_samples': Total number of samples
        - 'num_occupied': Number of occupied samples

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'frequency': [3620, 3625, 3630],
    ...     'power': [-100, -90, -80]
    ... })
    >>> metrics = compute_occupancy_metrics(df, 3610, 3650, -105, -105)
    >>> 'duty_cycle' in metrics
    True
    """
    if len(df) == 0:
        return {
            'duty_cycle': 0.0,
            'avg_power_occupied': np.nan,
            'signal_variation': np.nan,
            'num_samples': 0,
            'num_occupied': 0
        }

    # Apply linear threshold
    df = df.copy()
    df['threshold'] = df['frequency'].apply(
        lambda f: calculate_linear_threshold(f, band_start, band_end,
                                              threshold_start, threshold_end)
    )

    # Calculate metrics
    duty_cycle = calculate_duty_cycle(df)
    avg_power = calculate_avg_power_occupied(df)
    variation = calculate_signal_variation(df)

    occupied_df = df[df['power'] > df['threshold']]

    return {
        'duty_cycle': duty_cycle,
        'avg_power_occupied': avg_power,
        'signal_variation': variation,
        'num_samples': len(df),
        'num_occupied': len(occupied_df)
    }


def compute_chunk_metrics(df, band_start, band_end, chunk_size,
                           threshold_start, threshold_end):
    """
    Compute occupancy metrics for frequency chunks within a band.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'frequency' and 'power' columns
    band_start : float
        Start frequency of band in MHz
    band_end : float
        End frequency of band in MHz
    chunk_size : float
        Size of frequency chunks in MHz
    threshold_start : float
        Threshold at band_start in dB
    threshold_end : float
        Threshold at band_end in dB

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each chunk

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'frequency': np.linspace(3610, 3650, 1000),
    ...     'power': np.random.randn(1000) - 95
    ... })
    >>> chunk_metrics = compute_chunk_metrics(df, 3610, 3650, 10, -105, -105)
    >>> len(chunk_metrics) == 4  # 40 MHz / 10 MHz chunks
    True
    """
    results = []

    for start in np.arange(band_start, band_end, chunk_size):
        end = start + chunk_size
        chunk_df = df[(df['frequency'] >= start) & (df['frequency'] <= end)]

        metrics = compute_occupancy_metrics(chunk_df, start, end,
                                             threshold_start, threshold_end)
        metrics['chunk_start'] = start
        metrics['chunk_end'] = end
        results.append(metrics)

    return pd.DataFrame(results)
