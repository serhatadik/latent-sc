"""Functions for loading spectrum monitoring data from CSV files."""

import gzip
import os
import re
import pandas as pd
import numpy as np


def read_gzipped_csv(file_path):
    """
    Read a gzipped CSV file.

    Parameters
    ----------
    file_path : str
        Path to the .csv.gz file

    Returns
    -------
    pd.DataFrame
        DataFrame containing the CSV data

    Examples
    --------
    >>> df = read_gzipped_csv('monitor-1234567890.csv.gz')
    >>> 'frequency' in df.columns
    True
    """
    with gzip.open(file_path, 'rt') as file:
        return pd.read_csv(file)


def extract_timestamp_from_filename(filename):
    """
    Extract Unix timestamp from spectrum monitoring filename.

    Parameters
    ----------
    filename : str
        Filename in format 'monitor-{timestamp}.csv.gz'

    Returns
    -------
    int or None
        Unix timestamp if found, None otherwise

    Examples
    --------
    >>> extract_timestamp_from_filename('monitor-1234567890.csv.gz')
    1234567890
    >>> extract_timestamp_from_filename('invalid.csv') is None
    True
    """
    match = re.search(r'-(\d+)\.csv\.gz$', filename)
    return int(match.group(1)) if match else None


def concatenate_csvs_in_folder(folder_path, cutoff_date='2023-01-01'):
    """
    Concatenate all gzipped CSV files in a folder.

    Parameters
    ----------
    folder_path : str
        Path to folder containing .csv.gz files
    cutoff_date : str, optional
        Only include data after this date (default: '2023-01-01')

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all CSV data and timestamps

    Notes
    -----
    - Drops rows with NaN values
    - Filters data to only include dates after cutoff_date
    - Adds 'timestamp' column from filename

    Examples
    --------
    >>> df = concatenate_csvs_in_folder('./rfbaseline/Bookstore/')
    >>> 'timestamp' in df.columns
    True
    """
    dataframes = []
    cutoff = pd.Timestamp(cutoff_date)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for filename in os.listdir(folder_path):
        if filename.endswith('.gz'):
            file_path = os.path.join(folder_path, filename)
            timestamp = extract_timestamp_from_filename(filename)

            if timestamp is None:
                continue

            df = read_gzipped_csv(file_path)
            df['timestamp'] = pd.to_datetime(timestamp, unit='s')

            # Drop NaN values
            df = df.dropna()

            # Filter by date
            df = df[df['timestamp'] >= cutoff]

            dataframes.append(df)

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True)


def load_monitoring_data(monitor_name, band_start, band_end,
                         base_path='./rfbaseline/',
                         cutoff_date='2023-01-01'):
    """
    Load and filter spectrum monitoring data for a specific monitor and frequency band.

    Parameters
    ----------
    monitor_name : str
        Name of the monitoring station (e.g., 'Bookstore', 'EBC')
    band_start : float
        Start frequency of the band in MHz
    band_end : float
        End frequency of the band in MHz
    base_path : str, optional
        Base path to rfbaseline folder (default: './rfbaseline/')
    cutoff_date : str, optional
        Date cutoff (default: '2023-01-01')

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only the specified frequency band

    Examples
    --------
    >>> df = load_monitoring_data('Bookstore', 3610, 3650)
    >>> df['frequency'].min() >= 3610
    True
    >>> df['frequency'].max() <= 3650
    True
    """
    folder_path = os.path.join(base_path, monitor_name)

    # Load all data
    combined_df = concatenate_csvs_in_folder(folder_path, cutoff_date)

    if combined_df.empty:
        return combined_df

    # Drop center_freq column if it exists
    if 'center_freq' in combined_df.columns:
        combined_df = combined_df.drop(columns=['center_freq'])

    # Filter by frequency band
    filtered_df = combined_df[
        (combined_df['frequency'] >= band_start) &
        (combined_df['frequency'] <= band_end)
    ]

    return filtered_df


def min_max_frequencies(monitor_name, base_path='./rfbaseline/'):
    """
    Get the minimum and maximum frequencies observed at a monitor.

    Parameters
    ----------
    monitor_name : str
        Name of the monitoring station
    base_path : str, optional
        Base path to rfbaseline folder (default: './rfbaseline/')

    Returns
    -------
    tuple of float or (None, None)
        (min_frequency, max_frequency) in MHz

    Examples
    --------
    >>> min_freq, max_freq = min_max_frequencies('Bookstore')
    >>> min_freq < max_freq
    True
    """
    folder_path = os.path.join(base_path, monitor_name)
    combined_df = concatenate_csvs_in_folder(folder_path)

    if not combined_df.empty:
        return combined_df['frequency'].min(), combined_df['frequency'].max()
    else:
        return None, None


def median_time_diff_between_files(monitor_name, base_path='./rfbaseline/'):
    """
    Calculate median time difference between consecutive monitoring files.

    Parameters
    ----------
    monitor_name : str
        Name of the monitoring station
    base_path : str, optional
        Base path to rfbaseline folder (default: './rfbaseline/')

    Returns
    -------
    float or None
        Median time difference in seconds, or None if insufficient data

    Examples
    --------
    >>> diff = median_time_diff_between_files('Bookstore')
    >>> diff > 0
    True
    """
    folder_path = os.path.join(base_path, monitor_name)
    timestamps = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.gz'):
            timestamp = extract_timestamp_from_filename(filename)
            if timestamp is not None:
                timestamps.append(timestamp)

    if len(timestamps) < 2:
        return None

    timestamps.sort()
    time_differences = np.diff(timestamps)
    return np.median(time_differences) if len(time_differences) > 0 else None
