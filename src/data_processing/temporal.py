"""Functions for temporal filtering and analysis of spectrum data."""

import pandas as pd
import numpy as np
from .occupancy import compute_occupancy_metrics


# Time-of-day definitions
TIME_OF_DAY_INTERVALS = {
    'morning': ('04:00:00', '12:00:00'),
    'afternoon': ('12:00:00', '20:00:00'),
    'night': ('20:00:00', '04:00:00')
}

# Season definitions (by month)
SEASON_INTERVALS = {
    'spring': (3, 6),
    'summer': (6, 9),
    'autumn': (9, 12),
    'winter': (0, 3)
}


def filter_by_time_of_day(df, period):
    """
    Filter DataFrame by time of day.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp' column
    period : str
        Time period: 'morning', 'afternoon', or 'night'

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 14:00']),
    ...     'power': [-90, -85]
    ... })
    >>> morning_df = filter_by_time_of_day(df, 'morning')
    >>> len(morning_df)
    1
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")

    if period not in TIME_OF_DAY_INTERVALS:
        raise ValueError(f"Period must be one of {list(TIME_OF_DAY_INTERVALS.keys())}")

    start_time, end_time = TIME_OF_DAY_INTERVALS[period]

    if period == 'night':
        # Night wraps around midnight
        mask = ((df['timestamp'].dt.time >= pd.to_datetime(start_time).time()) |
                (df['timestamp'].dt.time < pd.to_datetime(end_time).time()))
    else:
        mask = ((df['timestamp'].dt.time >= pd.to_datetime(start_time).time()) &
                (df['timestamp'].dt.time < pd.to_datetime(end_time).time()))

    return df.loc[mask]


def filter_by_season(df, season):
    """
    Filter DataFrame by season.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp' column
    season : str
        Season: 'spring', 'summer', 'autumn', or 'winter'

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'timestamp': pd.to_datetime(['2023-01-15', '2023-07-15']),
    ...     'power': [-90, -85]
    ... })
    >>> winter_df = filter_by_season(df, 'winter')
    >>> len(winter_df)
    1
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")

    if season not in SEASON_INTERVALS:
        raise ValueError(f"Season must be one of {list(SEASON_INTERVALS.keys())}")

    start_month, end_month = SEASON_INTERVALS[season]

    mask = ((df['timestamp'].dt.month >= start_month) &
            (df['timestamp'].dt.month < end_month))

    return df.loc[mask]


def compute_temporal_metrics(df, band_start, band_end,
                              threshold_start, threshold_end):
    """
    Compute occupancy metrics for all time-of-day periods and seasons.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp', 'frequency', and 'power' columns
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
        Nested dictionary with structure:
        {
            'time_of_day': {
                'morning': {'duty_cycle': ..., 'avg_power_occupied': ..., ...},
                'afternoon': {...},
                'night': {...}
            },
            'season': {
                'spring': {...},
                'summer': {...},
                'autumn': {...},
                'winter': {...}
            }
        }

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
    ...     'frequency': np.full(100, 3620.0),
    ...     'power': np.random.randn(100) - 90
    ... })
    >>> metrics = compute_temporal_metrics(df, 3610, 3650, -105, -105)
    >>> 'time_of_day' in metrics
    True
    >>> 'season' in metrics
    True
    """
    results = {
        'time_of_day': {},
        'season': {}
    }

    # Time-of-day analysis
    for period in TIME_OF_DAY_INTERVALS.keys():
        period_df = filter_by_time_of_day(df, period)
        metrics = compute_occupancy_metrics(period_df, band_start, band_end,
                                             threshold_start, threshold_end)
        results['time_of_day'][period] = metrics

    # Seasonal analysis
    for season in SEASON_INTERVALS.keys():
        season_df = filter_by_season(df, season)
        metrics = compute_occupancy_metrics(season_df, band_start, band_end,
                                             threshold_start, threshold_end)
        results['season'][season] = metrics

    return results


def analyze_all_monitors_temporal(monitor_names, band_start, band_end,
                                    threshold_start, threshold_end,
                                    base_path='./rfbaseline/',
                                    cutoff_date='2023-01-01'):
    """
    Analyze temporal patterns for all monitoring stations.

    Parameters
    ----------
    monitor_names : list of str
        Names of monitoring stations
    band_start : float
        Start frequency of band in MHz
    band_end : float
        End frequency of band in MHz
    threshold_start : float
        Threshold at band_start in dB
    threshold_end : float
        Threshold at band_end in dB
    base_path : str, optional
        Base path to rfbaseline folder
    cutoff_date : str, optional
        Date cutoff

    Returns
    -------
    dict
        Dictionary mapping monitor names to temporal metrics

    Examples
    --------
    >>> monitors = ['Bookstore', 'EBC']
    >>> results = analyze_all_monitors_temporal(monitors, 3610, 3650, -105, -105)
    >>> 'Bookstore' in results
    True
    """
    from .loader import load_monitoring_data

    all_results = {}

    for monitor in monitor_names:
        print(f"Processing {monitor}...")
        df = load_monitoring_data(monitor, band_start, band_end,
                                   base_path, cutoff_date)

        if df.empty:
            print(f"  Warning: No data for {monitor}")
            continue

        metrics = compute_temporal_metrics(df, band_start, band_end,
                                            threshold_start, threshold_end)
        all_results[monitor] = metrics

    return all_results
