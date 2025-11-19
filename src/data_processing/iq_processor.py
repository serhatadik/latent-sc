"""
IQ Sample Processing Module for Raw Data to Monitoring Locations Conversion.

This module provides functionality to:
1. Load raw IQ samples from any directory under raw_data/
2. Compute Power Spectral Density (PSD) and extract channel power
3. Match power measurements with GPS coordinates
4. Aggregate measurements by receiver location

Based on signal processing logic from the data branch.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gpxpy
import gpxpy.gpx
from collections import defaultdict


# RF Channel Definitions (from data branch settings.py)
RF_CHANNELS = {
    "TX1": (3533903750, 3533931250),
    "TX2": (3533945000, 3533972500),
    "TX3": (3533986250, 3534013750),
    "TX4": (3534027500, 3534055000),
    "TX5": (3534068750, 3534096250),
}

# Transmitter to RF Channel Mapping
# Note: EBC and USTAR share TX1 frequency band but are separated by date:
# - EBC: Used until June 27, 2023 (inclusive)
# - USTAR: Used from June 28, 2023 onwards
TRANSMITTER_TO_CHANNEL = {
    "ebc": "TX1",        # Active: <= 2023-06-27
    "ustar": "TX1",      # Active: >= 2023-06-28
    "guesthouse": "TX2",
    "mario": "TX3",
    "moran": "TX4",
    "wasatch": "TX5",
}

# Date threshold for EBC/USTAR split (midnight on June 28, 2023)
TX1_SPLIT_DATE = np.datetime64('2023-06-28T00:00:00')

# Default RF parameters
DEFAULT_CENTER_FREQ = 3.534e9  # Hz
DEFAULT_SAMPLE_RATE = 0.22e6   # Hz (220 kHz)


def find_indices_outside(arr: np.ndarray, a: float, b: float) -> List[int]:
    """
    Find indices of array elements outside the range (a, b).

    Parameters
    ----------
    arr : np.ndarray
        Input array
    a : float
        Lower bound (exclusive)
    b : float
        Upper bound (exclusive)

    Returns
    -------
    List[int]
        List of indices where arr[i] is not in (a, b)
    """
    indices = [i for i, x in enumerate(arr) if not (a < x < b)]
    return indices


def compute_psd(iq_samples: np.ndarray, sample_rate: float, center_freq: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density (PSD) from IQ samples.

    This function applies a Hamming window, performs FFT, and computes PSD in dB.

    Parameters
    ----------
    iq_samples : np.ndarray
        Complex IQ samples
    sample_rate : float
        Sampling rate in Hz
    center_freq : float
        Center frequency in Hz

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (frequencies, PSD_shifted) where:
        - frequencies: Frequency array centered around center_freq
        - PSD_shifted: PSD in dB (10*log10(W/Hz)), frequency-shifted
    """
    nsamps = len(iq_samples)

    # Apply Hamming window
    x = iq_samples * np.hamming(nsamps)

    # Compute PSD
    PSD = np.abs(np.fft.fft(x)) ** 2 / (nsamps * sample_rate)
    PSD_log = 10.0 * np.log10(PSD)
    PSD_shifted = np.fft.fftshift(PSD_log)

    # Compute frequency array
    f = np.arange(sample_rate / -2.0, sample_rate / 2.0, sample_rate / nsamps)
    f += center_freq

    return f, PSD_shifted


def extract_channel_power(
    frequencies: np.ndarray,
    psd: np.ndarray,
    channel_min: float,
    channel_max: float
) -> float:
    """
    Extract maximum power for a specific RF channel.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array
    psd : np.ndarray
        Power spectral density array
    channel_min : float
        Minimum frequency of the channel
    channel_max : float
        Maximum frequency of the channel

    Returns
    -------
    float
        Maximum power (in dB) within the channel
    """
    # Find indices outside the channel range
    idx = find_indices_outside(frequencies, channel_min, channel_max)

    # Create a copy and mask out-of-channel values
    psd_channel = psd.copy()
    psd_channel[idx] = -200  # Very low value to exclude from max

    # Find maximum power in channel
    arg_max = np.argmax(psd_channel)

    return psd_channel[arg_max]


def process_iq_sample(
    iq_samples: np.ndarray,
    transmitter_name: str,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    center_freq: float = DEFAULT_CENTER_FREQ
) -> float:
    """
    Process IQ samples and extract power for a specific transmitter.

    Parameters
    ----------
    iq_samples : np.ndarray
        Complex IQ samples
    transmitter_name : str
        Name of transmitter (ustar, guesthouse, mario, moran, wasatch)
    sample_rate : float, optional
        Sampling rate in Hz (default: 220 kHz)
    center_freq : float, optional
        Center frequency in Hz (default: 3.534 GHz)

    Returns
    -------
    float
        Power value in dB for the specified transmitter's channel
    """
    # Map transmitter name to RF channel
    if transmitter_name not in TRANSMITTER_TO_CHANNEL:
        raise ValueError(f"Unknown transmitter: {transmitter_name}. "
                        f"Valid options: {list(TRANSMITTER_TO_CHANNEL.keys())}")

    channel_name = TRANSMITTER_TO_CHANNEL[transmitter_name]
    channel_min, channel_max = RF_CHANNELS[channel_name]

    # Compute PSD
    frequencies, psd = compute_psd(iq_samples, sample_rate, center_freq)

    # Extract power for the specific channel
    power = extract_channel_power(frequencies, psd, channel_min, channel_max)

    return power


def load_iq_samples_from_directory(
    directory: Path,
    samples_to_skip: int = 2
) -> Dict[np.datetime64, np.ndarray]:
    """
    Load IQ samples from a single directory.

    Parameters
    ----------
    directory : Path
        Directory path containing IQ sample .npy files
    samples_to_skip : int, optional
        Number of files to skip at the end (potentially incomplete)

    Returns
    -------
    Dict[np.datetime64, np.ndarray]
        Dictionary mapping timestamps to IQ sample arrays
    """
    data = {}

    if not directory.exists():
        print(f"Warning: Directory does not exist: {directory}")
        return data

    files = sorted(os.listdir(directory))
    print(f"Loading {len(files)} files from {directory.name}")

    for cnt, file in enumerate(files):
        # Skip last N files (potentially incomplete)
        if cnt < len(files) - samples_to_skip and file.endswith('.npy'):
            # Extract timestamp from filename (format: YYYY-MM-DD-HH-MM-SS-IQ...)
            try:
                time_str = file.split('-IQ')[0].split('.')[0]
                time = pd.to_datetime(time_str)
                time = np.datetime64(time).astype('datetime64[s]')

                # Load IQ samples (take first element [0] as files contain nested arrays)
                iq_data = np.load(directory / file)
                if isinstance(iq_data, np.ndarray) and len(iq_data) > 0:
                    data[time] = iq_data[0] if iq_data.ndim > 1 else iq_data
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}")
                continue

    return data


def load_iq_samples_from_directories(
    directories: List[Path],
    samples_to_skip: int = 2
) -> Dict[np.datetime64, np.ndarray]:
    """
    Load IQ samples from multiple directories.

    Parameters
    ----------
    directories : List[Path]
        List of directory paths containing IQ sample files
    samples_to_skip : int, optional
        Number of files to skip at end of each directory

    Returns
    -------
    Dict[np.datetime64, np.ndarray]
        Dictionary mapping timestamps to IQ sample arrays
    """
    all_data = {}

    for directory in directories:
        data = load_iq_samples_from_directory(directory, samples_to_skip)
        all_data.update(data)

    return all_data


def load_gps_from_csv(
    gps_files: List[Path],
    time_offset_hours: int = -6,
    min_latitude: float = 40
) -> Dict[np.datetime64, np.ndarray]:
    """
    Load GPS coordinates from CSV files.

    Parameters
    ----------
    gps_files : List[Path]
        List of paths to GPS CSV files
    time_offset_hours : int, optional
        Time offset to apply for timezone conversion (default: -6 for UTC-6)
    min_latitude : float, optional
        Minimum valid latitude for filtering (default: 40)

    Returns
    -------
    Dict[np.datetime64, np.ndarray]
        Dictionary mapping timestamps to [longitude, latitude] arrays
    """
    gps_datas = []
    for gps_file in gps_files:
        try:
            gps_datas.append(pd.read_csv(gps_file))
        except Exception as e:
            print(f"Warning: Failed to load GPS file {gps_file}: {e}")
            continue

    if not gps_datas:
        return {}

    gps_data = pd.concat(gps_datas, axis=0, ignore_index=True)

    # Use format='mixed' to handle timestamps with fractional seconds
    parsed_times = pd.to_datetime(gps_data['date time'], format='mixed')
    # Apply timezone offset (time_offset_hours is negative for UTC-6, so we add it)
    offset_times = parsed_times + pd.Timedelta(hours=time_offset_hours)

    # Convert to numpy datetime64[s]
    gps_times = offset_times.values.astype('datetime64[s]')

    latitude = np.array(gps_data['latitude'])
    longitude = np.array(gps_data['longitude'])

    coords = {
        gps_times[i]: np.array([longitude[i], latitude[i]])
        for i in range(len(gps_times))
        if latitude[i] > min_latitude
    }

    return coords


def match_power_with_gps(
    iq_data: Dict[np.datetime64, np.ndarray],
    gps_coords: Dict[np.datetime64, np.ndarray],
    transmitter_name: str,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    center_freq: float = DEFAULT_CENTER_FREQ,
    progress_interval: int = 100,
    time_tolerance_seconds: int = 10
) -> List[Dict]:
    """
    Process IQ samples, extract power, and match with GPS coordinates.

    Uses nearest-neighbor matching within a time tolerance window since
    GPS and IQ timestamps may not align exactly.

    For EBC/USTAR transmitters (both using TX1), applies date filtering:
    - EBC: Only processes samples from June 27, 2023 and earlier
    - USTAR: Only processes samples from June 28, 2023 and later

    Parameters
    ----------
    iq_data : Dict[np.datetime64, np.ndarray]
        Dictionary mapping timestamps to IQ samples
    gps_coords : Dict[np.datetime64, np.ndarray]
        Dictionary mapping timestamps to [lon, lat] coordinates
    transmitter_name : str
        Name of transmitter to extract power for (ebc, ustar, guesthouse, mario, moran, wasatch)
    sample_rate : float, optional
        Sampling rate in Hz
    center_freq : float, optional
        Center frequency in Hz
    progress_interval : int, optional
        Print progress every N samples
    time_tolerance_seconds : int, optional
        Maximum time difference for matching GPS coordinates (default: 10s)

    Returns
    -------
    List[Dict]
        List of dictionaries containing:
        - 'timestamp': np.datetime64
        - 'power': float (dB)
        - 'longitude': float
        - 'latitude': float
    """
    times = sorted(list(iq_data.keys()))
    gps_times = np.array(sorted(list(gps_coords.keys())))
    measurements = []

    # Determine date filter for EBC/USTAR
    apply_date_filter = transmitter_name in ['ebc', 'ustar']
    if apply_date_filter:
        if transmitter_name == 'ebc':
            date_filter_desc = f"samples before {TX1_SPLIT_DATE} (EBC period)"
        else:  # ustar
            date_filter_desc = f"samples from {TX1_SPLIT_DATE} onwards (USTAR period)"
    else:
        date_filter_desc = "all samples"

    print(f"Processing {len(times)} IQ samples for transmitter '{transmitter_name}'...")
    print(f"GPS time range: {gps_times[0]} to {gps_times[-1]}")
    print(f"IQ time range: {times[0]} to {times[-1]}")
    print(f"Time tolerance: ±{time_tolerance_seconds} seconds")
    print(f"Date filter: {date_filter_desc}")

    skipped_by_date = 0

    for i, time in enumerate(times):
        if i % progress_interval == 0:
            print(f"Progress: {i}/{len(times)} ({i/len(times)*100:.1f}%)")

        try:
            # Apply date filtering for EBC/USTAR
            if apply_date_filter:
                if transmitter_name == 'ebc':
                    # EBC: Only use samples from June 27, 2023 and earlier
                    if time >= TX1_SPLIT_DATE:
                        skipped_by_date += 1
                        continue
                else:  # ustar
                    # USTAR: Only use samples from June 28, 2023 and later
                    if time < TX1_SPLIT_DATE:
                        skipped_by_date += 1
                        continue

            # Find nearest GPS timestamp within tolerance
            time_diffs = np.abs((gps_times - time).astype('timedelta64[s]').astype(int))
            nearest_idx = np.argmin(time_diffs)
            min_diff = time_diffs[nearest_idx]

            if min_diff <= time_tolerance_seconds:
                # Process IQ sample to extract power
                power = process_iq_sample(
                    iq_data[time],
                    transmitter_name,
                    sample_rate,
                    center_freq
                )

                gps_time = gps_times[nearest_idx]
                lon, lat = gps_coords[gps_time]

                measurements.append({
                    'timestamp': time,
                    'power': power,
                    'longitude': lon,
                    'latitude': lat
                })
        except Exception as e:
            if i < 10:  # Only print first few errors to avoid spam
                print(f"Warning: Failed to process sample at {time}: {e}")
            continue

    if apply_date_filter and skipped_by_date > 0:
        print(f"Skipped {skipped_by_date} samples outside {transmitter_name.upper()} date range")

    print(f"Matched {len(measurements)} measurements with GPS coordinates")
    return measurements


def aggregate_measurements_by_location(
    measurements: List[Dict],
    dedup_threshold_meters: float = 20.0,
    min_samples_per_location: int = 10
) -> List[Dict]:
    """
    Aggregate measurements by receiver location.

    Groups measurements within dedup_threshold_meters and computes
    average power and location for each unique receiver position.

    Parameters
    ----------
    measurements : List[Dict]
        List of measurement dictionaries from match_power_with_gps()
    dedup_threshold_meters : float, optional
        Distance threshold in meters for grouping locations (default: 20m)
    min_samples_per_location : int, optional
        Minimum number of measurements required per location (default: 10)

    Returns
    -------
    List[Dict]
        List of aggregated location dictionaries containing:
        - 'name': str (auto-generated like "Location_1")
        - 'longitude': float (averaged)
        - 'latitude': float (averaged)
        - 'avg_power': float (averaged dB)
        - 'num_samples': int
        - 'std_power': float (standard deviation in dB)
    """
    from geopy import distance

    if not measurements:
        return []

    print(f"\nAggregating {len(measurements)} measurements by location...")
    print(f"Deduplication threshold: {dedup_threshold_meters} meters")

    # Convert to numpy arrays for easier processing
    coords = np.array([[m['latitude'], m['longitude']] for m in measurements])
    powers = np.array([m['power'] for m in measurements])

    # Group measurements by location
    location_groups = []
    used_indices = set()

    for i, coord1 in enumerate(coords):
        if i in used_indices:
            continue

        # Find all measurements within threshold distance
        group_indices = [i]
        for j, coord2 in enumerate(coords):
            if j <= i or j in used_indices:
                continue

            dist_m = distance.distance(tuple(coord1), tuple(coord2)).m
            if dist_m < dedup_threshold_meters:
                group_indices.append(j)
                used_indices.add(j)

        used_indices.add(i)
        location_groups.append(group_indices)

    print(f"Found {len(location_groups)} unique location groups")

    # Aggregate measurements for each location
    aggregated_locations = []

    for group_idx in location_groups:
        if len(group_idx) < min_samples_per_location:
            continue

        # Compute averages
        group_lats = coords[group_idx, 0]
        group_lons = coords[group_idx, 1]
        group_powers = powers[group_idx]

        aggregated_locations.append({
            'latitude': float(np.mean(group_lats)),
            'longitude': float(np.mean(group_lons)),
            'avg_power': float(np.mean(group_powers)),
            'std_power': float(np.std(group_powers)),
            'num_samples': len(group_idx)
        })

    # Sort by number of samples (descending)
    aggregated_locations.sort(key=lambda x: x['num_samples'], reverse=True)

    # Assign names
    for i, loc in enumerate(aggregated_locations):
        loc['name'] = f"Location_{i+1}"

    print(f"Aggregated to {len(aggregated_locations)} locations with >={min_samples_per_location} samples")

    return aggregated_locations
