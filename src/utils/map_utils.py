"""Utilities for loading and manipulating SLC map data."""

import os
import warnings
import numpy as np
import scipy.io as sio


def load_slc_map(map_folder_dir="./", downsample_factor=10):
    """
    Load SLC (Salt Lake City) map data from .mat file.

    Parameters
    ----------
    map_folder_dir : str, optional
        Directory containing the SLCmap file (default: "./")
    downsample_factor : int, optional
        Factor by which to downsample the map (default: 10)

    Returns
    -------
    dict
        Dictionary containing:
        - 'map': Combined DEM + building height array
        - 'dem': Digital elevation model
        - 'buildings': Building heights
        - 'UTM_lat': UTM latitude coordinates
        - 'UTM_long': UTM longitude coordinates
        - 'cellsize': Cell size in meters
        - 'axis': Map axis bounds

    Raises
    ------
    FileNotFoundError
        If no SLCmap file is found in the directory

    Examples
    --------
    >>> map_data = load_slc_map("./", downsample_factor=10)
    >>> map_data['map'].shape
    (520, 580)
    """
    # Find SLCmap file
    directory = os.listdir(map_folder_dir)
    map_file = None

    for fname in directory:
        if "SLCmap" in fname and fname.endswith('.mat'):
            map_file = os.path.join(map_folder_dir, fname)
            break

    if map_file is None:
        error_msg = f'Error: No SLCmap file found in folder: {map_folder_dir}'
        warnings.warn(error_msg)
        raise FileNotFoundError(error_msg)

    print(f'Loading map from: {map_file}')

    # Load .mat file
    mat_data = sio.loadmat(map_file)
    map_struct = mat_data['SLC']

    # Extract SLC structure
    SLC = map_struct[0][0]
    column_map = dict(zip(
        [name for name in SLC.dtype.names],
        [i for i in range(len(SLC.dtype.names))]
    ))

    # Extract data
    dem = SLC[column_map["dem"]]
    hybrid_bldg = SLC[column_map["hybrid_bldg"]]

    # Combine DEM with building heights (convert feet to meters: 0.3048)
    combined_map = dem + 0.3048 * hybrid_bldg

    # Downsample
    if downsample_factor > 1:
        combined_map = combined_map[::downsample_factor, ::downsample_factor]
        dem = dem[::downsample_factor, ::downsample_factor]
        hybrid_bldg = hybrid_bldg[::downsample_factor, ::downsample_factor]

    # Get UTM coordinates
    N1, N2 = combined_map.shape
    en = SLC[column_map["axis"]]
    cellsize = float(SLC[column_map["cellsize"]])

    # axis format: [easting_min, easting_max, northing_min, northing_max]
    # UTM_long should be Easting (x-axis), UTM_lat should be Northing (y-axis)
    UTM_long = np.linspace(en[0, 0], en[0, 1] - cellsize, N2)  # Easting (x)
    UTM_lat = np.linspace(en[0, 2], en[0, 3] - cellsize, N1)   # Northing (y)

    return {
        'map': combined_map,
        'dem': dem,
        'buildings': hybrid_bldg,
        'UTM_lat': np.squeeze(UTM_lat),
        'UTM_long': np.squeeze(UTM_long),
        'cellsize': cellsize * downsample_factor,
        'axis': en,
        'shape': combined_map.shape
    }


def get_utm_coordinates(map_data):
    """
    Extract UTM coordinates from map data.

    Parameters
    ----------
    map_data : dict
        Map data dictionary from load_slc_map()

    Returns
    -------
    tuple of ndarray
        (UTM_lat, UTM_long) coordinate arrays
    """
    return map_data['UTM_lat'], map_data['UTM_long']


def format_map_axes(ax, map_data, num_ticks=5):
    """
    Format matplotlib axes with proper UTM coordinate labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to format
    map_data : dict
        Map data dictionary from load_slc_map()
    num_ticks : int, optional
        Approximate number of ticks per axis (default: 5)

    Returns
    -------
    matplotlib.axes.Axes
        Formatted axes object

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> map_data = load_slc_map()
    >>> ax = format_map_axes(ax, map_data)
    """
    UTM_lat, UTM_long = get_utm_coordinates(map_data)

    # X-axis (longitude)
    interval_x = max(1, len(UTM_lat) // num_ticks)
    tick_values_x = list(range(0, len(UTM_lat), interval_x))
    tick_labels_x = [f'{lat:.1f}' for lat in UTM_lat[::interval_x]]
    ax.set_xticks(tick_values_x)
    ax.set_xticklabels(tick_labels_x, fontsize=14, rotation=0)

    # Y-axis (latitude)
    interval_y = max(1, len(UTM_long) // num_ticks)
    tick_values_y = list(range(0, len(UTM_long), interval_y))
    tick_labels_y = [f'{lon:.1f}' for lon in UTM_long[::interval_y]]

    # Set first y-tick label to empty to avoid overlap
    ax.set_yticks([0] + tick_values_y[1:])
    ax.set_yticklabels([""] + tick_labels_y[1:], fontsize=14, rotation=90)

    ax.set_xlabel('UTM$_E$ [m]', fontsize=18, labelpad=10)
    ax.set_ylabel('UTM$_N$ [m]', fontsize=18, labelpad=10)

    return ax
