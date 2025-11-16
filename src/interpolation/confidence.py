"""Confidence level calculation based on proximity to sensors."""

import numpy as np
from ..utils.coordinates import euclidean_distance


def calculate_confidence_level(dp, dmax, alpha):
    """
    Calculate confidence level at a location.

    Implements Equation (9) from the paper:
        γ_p = (1 - β(d_p)) * exp(-α * d_p)
        β(d_p) = min(d_p / d_max, 1)

    Parameters
    ----------
    dp : float
        Distance to nearest sensor in meters
    dmax : float
        Maximum distance threshold in meters
    alpha : float
        Steepness parameter in 1/meters

    Returns
    -------
    float
        Confidence level between 0 and 1

    Examples
    --------
    >>> conf = calculate_confidence_level(0, 1000, 0.01)
    >>> abs(conf - 1.0) < 1e-10  # Should be 1.0 at sensor location
    True
    >>> conf = calculate_confidence_level(500, 1000, 0.01)
    >>> 0 < conf < 1  # Should be between 0 and 1
    True
    """
    # Beta factor: linear increase from 0 to 1
    beta = min(dp / dmax, 1.0)

    # Exponential decay factor
    exp_decay = np.exp(-alpha * dp)

    # Combined confidence
    confidence = (1 - beta) * exp_decay

    return confidence


def compute_confidence_map(map_shape, sensor_locations, dmax=1000, alpha=0.01,
                            scale=1.0):
    """
    Compute confidence level at all map locations.

    Used for creating Figure 7 in the paper (signal variation & confidence).

    Parameters
    ----------
    map_shape : tuple of int
        (height, width) of the map in pixels
    sensor_locations : ndarray of shape (n_sensors, 2)
        Sensor coordinates in pixel space (col, row)
    dmax : float, optional
        Maximum distance threshold in meters (default: 1000)
    alpha : float, optional
        Steepness parameter in 1/meters (default: 0.01)
    scale : float, optional
        Scaling factor to convert pixels to meters (default: 1.0)

    Returns
    -------
    ndarray of shape (height, width)
        Confidence level at each pixel (0 to 1)

    Examples
    --------
    >>> sensors = np.array([[50, 50], [150, 150]])
    >>> conf_map = compute_confidence_map((200, 200), sensors, dmax=1000,
    ...                                    alpha=0.01, scale=5)
    >>> conf_map.shape
    (200, 200)
    >>> conf_map[50, 50]  # At sensor location
    1.0
    >>> conf_map[0, 0] < conf_map[50, 50]  # Lower confidence far from sensor
    True

    Notes
    -----
    Confidence is highest at sensor locations and decreases with distance.
    The rate of decrease is controlled by alpha and dmax.
    """
    height, width = map_shape
    confidence_map = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            # Find minimum distance to any sensor
            min_distance = float('inf')

            for sensor in sensor_locations:
                sensor_col, sensor_row = sensor
                dist = euclidean_distance([j, i], [sensor_col, sensor_row], scale)
                min_distance = min(min_distance, dist)

            # Calculate confidence
            confidence_map[i, j] = calculate_confidence_level(
                min_distance, dmax, alpha
            )

    return confidence_map


def compute_confidence_map_vectorized(map_shape, sensor_locations,
                                       dmax=1000, alpha=0.01, scale=1.0):
    """
    Vectorized version of confidence map computation (faster).

    Parameters
    ----------
    map_shape : tuple
        (height, width) of map
    sensor_locations : ndarray
        Sensor coordinates (col, row)
    dmax : float, optional
        Maximum distance (default: 1000)
    alpha : float, optional
        Steepness parameter (default: 0.01)
    scale : float, optional
        Pixel to meter conversion (default: 1.0)

    Returns
    -------
    ndarray
        Confidence map

    Examples
    --------
    >>> sensors = np.array([[50, 50]])
    >>> conf = compute_confidence_map_vectorized((100, 100), sensors)
    >>> conf.shape
    (100, 100)
    """
    height, width = map_shape
    n_sensors = len(sensor_locations)

    # Create coordinate grids
    rows, cols = np.mgrid[0:height, 0:width]

    # Initialize with large distances
    min_distances = np.full((height, width), np.inf)

    # Compute distance to each sensor
    for sensor in sensor_locations:
        sensor_col, sensor_row = sensor
        dx = (cols - sensor_col) * scale
        dy = (rows - sensor_row) * scale
        distances = np.sqrt(dx**2 + dy**2)

        # Update minimum distance
        min_distances = np.minimum(min_distances, distances)

    # Calculate confidence
    beta = np.minimum(min_distances / dmax, 1.0)
    exp_decay = np.exp(-alpha * min_distances)
    confidence_map = (1 - beta) * exp_decay

    return confidence_map
