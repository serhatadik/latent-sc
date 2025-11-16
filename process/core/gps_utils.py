"""
GPS data loading and coordinate utilities.
Extracted from read_samples.py, Fading_Analysis.py, and fading_analysis_rot.py
"""
import numpy as np
import pandas as pd
import gpxpy
import gpxpy.gpx
import os
from typing import Dict, List
from pathlib import Path


def load_gps_from_csv(
    gps_files: List[str],
    time_offset_hours: int = -6,
    min_latitude: float = 40
) -> Dict[np.datetime64, np.ndarray]:
    """
    Load GPS coordinates from CSV files.

    Args:
        gps_files: List of paths to GPS CSV files
        time_offset_hours: Time offset to apply (for timezone conversion)
        min_latitude: Minimum valid latitude (for filtering invalid coordinates)

    Returns:
        Dictionary mapping timestamps to [longitude, latitude] arrays
    """
    gps_datas = []
    for gps_file in gps_files:
        gps_datas.append(pd.read_csv(gps_file))

    gps_data = pd.concat(gps_datas, axis=0)

    gps_times = np.array(
        pd.to_datetime(gps_data['date time']) - np.timedelta64(time_offset_hours, 'h'),
        dtype='datetime64[s]'
    )
    latitude = np.array(gps_data['latitude'])
    longitude = np.array(gps_data['longitude'])

    coords = {
        gps_times[i]: np.array([longitude[i], latitude[i]])
        for i in range(len(gps_times))
    }

    return coords


def load_gps_from_gpx(
    gpx_files: List[str],
    time_offset_hours: int = -6,
    min_latitude: float = 40
) -> Dict[np.datetime64, np.ndarray]:
    """
    Load GPS coordinates from GPX files.

    Args:
        gpx_files: List of paths to GPX files
        time_offset_hours: Time offset to apply (for timezone conversion)
        min_latitude: Minimum valid latitude (for filtering invalid coordinates)

    Returns:
        Dictionary mapping timestamps to [longitude, latitude] arrays
    """
    coords = {}

    for gpx_file_path in gpx_files:
        with open(gpx_file_path, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        timestamp = np.datetime64(
                            pd.to_datetime(point.time) - np.timedelta64(time_offset_hours, 'h')
                        ).astype('datetime64[s]')

                        # Only add if latitude is above minimum threshold
                        if point.latitude > min_latitude:
                            coords[timestamp] = np.array([point.longitude, point.latitude])

    return coords


def merge_gps_coords(
    coords_main: Dict[np.datetime64, np.ndarray],
    coords_additional: Dict[np.datetime64, np.ndarray],
    min_latitude: float = 40
) -> Dict[np.datetime64, np.ndarray]:
    """
    Merge two GPS coordinate dictionaries, adding additional coords not in main.

    Args:
        coords_main: Main coordinate dictionary
        coords_additional: Additional coordinates to merge in
        min_latitude: Minimum valid latitude

    Returns:
        Merged coordinate dictionary
    """
    coords_merged = coords_main.copy()

    for timestamp, coord in coords_additional.items():
        if timestamp not in coords_merged:
            if coord[1] > min_latitude:  # coord[1] is latitude
                coords_merged[timestamp] = coord

    return coords_merged


def load_mobile_gps_data(
    gps_csv_dir: Path,
    gpx_dir: Path,
    time_offset_hours: int = -6,
    min_latitude: float = 40
) -> Dict[np.datetime64, np.ndarray]:
    """
    Load GPS data for mobile measurements (walking/driving).

    Combines CSV and GPX GPS data sources.

    Args:
        gps_csv_dir: Directory containing GPS CSV files
        gpx_dir: Directory containing GPX files
        time_offset_hours: Time offset for timezone conversion
        min_latitude: Minimum valid latitude

    Returns:
        Dictionary mapping timestamps to [longitude, latitude] arrays
    """
    # Load from CSV files
    gps_csv_files = [
        str(gps_csv_dir / name)
        for name in os.listdir(gps_csv_dir)
        if name.endswith('.txt')
    ]
    coords_csv = load_gps_from_csv(gps_csv_files, time_offset_hours, min_latitude)

    # Load from GPX files
    gpx_files = [
        str(gpx_dir / name)
        for name in os.listdir(gpx_dir)
        if name.startswith('2023')
    ]
    coords_gpx = load_gps_from_gpx(gpx_files, time_offset_hours, min_latitude)

    # Merge coordinates
    coords = merge_gps_coords(coords_csv, coords_gpx, min_latitude)

    return coords


def load_stationary_gps_data(
    gps_csv_dir: Path,
    file_pattern: str = "*Stat.txt",
    time_offset_hours: int = -6
) -> Dict[np.datetime64, np.ndarray]:
    """
    Load GPS data for stationary measurements.

    Args:
        gps_csv_dir: Directory containing GPS CSV files
        file_pattern: File pattern to match (default: "*Stat.txt")
        time_offset_hours: Time offset for timezone conversion

    Returns:
        Dictionary mapping timestamps to [longitude, latitude] arrays
    """
    gps_files = [
        str(gps_csv_dir / name)
        for name in os.listdir(gps_csv_dir)
        if name.endswith('Stat.txt')
    ]

    coords = load_gps_from_csv(gps_files, time_offset_hours)

    return coords


def deduplicate_coordinates(
    coordinates: np.ndarray,
    threshold_meters: float = 20
) -> np.ndarray:
    """
    Deduplicate coordinates by merging points within a threshold distance.

    This function groups coordinates that are within threshold_meters of each
    other and replaces them with their mean coordinate.

    Args:
        coordinates: Nx2 array of [longitude, latitude] coordinates
        threshold_meters: Distance threshold in meters for merging

    Returns:
        Nx2 array with deduplicated coordinates
    """
    from geopy import distance

    coord_unified = np.zeros_like(coordinates)
    skip_list = []

    for i, c1 in enumerate(coordinates):
        if i in skip_list:
            continue

        dummy = []
        dummy_idx = []
        flag = 1

        for j, c2 in enumerate(coordinates):
            # Convert [lon, lat] to (lat, lon) for distance calculation
            dist_m = distance.distance(tuple(c1[::-1]), tuple(c2[::-1])).m

            if dist_m < threshold_meters:
                flag = 0
                if j not in skip_list:
                    dummy.append(list(c2))
                    dummy_idx.append(j)
                    skip_list.append(j)

        if flag == 0:
            # Replace all grouped coordinates with their mean
            coord_unified[dummy_idx, :] = np.mean(dummy, axis=0)

    # Copy over coordinates that weren't grouped
    for i, c1 in enumerate(coordinates):
        if i not in skip_list:
            coord_unified[i, :] = list(c1)

    return coord_unified


def find_unique_coordinates(coordinates: np.ndarray) -> List[List[float]]:
    """
    Find unique coordinate locations from array.

    Args:
        coordinates: Nx2 array of coordinates

    Returns:
        List of unique [longitude, latitude] pairs
    """
    unique_coords = set(tuple(coord) for coord in coordinates)
    return [list(coord) for coord in unique_coords]
