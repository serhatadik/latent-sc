"""Path loss models and distance calculations for RF propagation."""

import numpy as np
from ..utils.coordinates import euclidean_distance


def log_distance_path_loss(ti, distances, pi0=0, np_exponent=2, di0=1):
    """
    Calculate received power using log-distance path loss model.

    Implements Equation (1) from the paper:
        p_rx = ti - pi0 - 10*np*log10(d/di0)

    Parameters
    ----------
    ti : float
        Transmit power in dB
    distances : ndarray
        Array of distances in meters
    pi0 : float, optional
        Reference power level at reference distance (default: 0)
    np_exponent : float, optional
        Path loss exponent (default: 2 for free space)
    di0 : float, optional
        Reference distance in meters (default: 1)

    Returns
    -------
    ndarray
        Received power levels in dB

    Examples
    --------
    >>> distances = np.array([10, 100, 1000])
    >>> powers = log_distance_path_loss(30, distances, np_exponent=2)
    >>> powers[0] > powers[1] > powers[2]  # Power decreases with distance
    True

    Notes
    -----
    This is a simplified log-distance path loss model. More complex models
    (e.g., ray-tracing) could be substituted for better accuracy.
    """
    distances = np.array(distances)
    # Avoid log(0) by ensuring minimum distance
    distances = np.maximum(distances, di0)

    path_loss = pi0 + 10 * np_exponent * np.log10(distances / di0)
    return ti - path_loss


def compute_path_loss_vector(ti, sensor_locations, transmitter_location,
                              scale=1.0, np_exponent=2):
    """
    Compute path loss from a transmitter to all sensors.

    Parameters
    ----------
    ti : float
        Transmit power in dB
    sensor_locations : ndarray of shape (n_sensors, 2)
        Sensor coordinates
    transmitter_location : array-like of length 2
        Transmitter coordinates (x, y) or (row, col)
    scale : float, optional
        Scaling factor to convert indices to meters (default: 1.0)
    np_exponent : float, optional
        Path loss exponent (default: 2)

    Returns
    -------
    ndarray of shape (n_sensors,)
        Predicted received power at each sensor

    Examples
    --------
    >>> sensors = np.array([[0, 0], [10, 0], [0, 10]])
    >>> tx_loc = [5, 5]
    >>> powers = compute_path_loss_vector(30, sensors, tx_loc, scale=5)
    >>> len(powers)
    3
    """
    n_sensors = len(sensor_locations)
    distances = np.zeros(n_sensors)

    for i, sensor in enumerate(sensor_locations):
        distances[i] = euclidean_distance(transmitter_location, sensor, scale)

    return log_distance_path_loss(ti, distances, np_exponent=np_exponent)


def compute_distance_matrix(sensor_locations, map_shape, scale=1.0):
    """
    Compute distance matrix from all map pixels to all sensors.

    Parameters
    ----------
    sensor_locations : ndarray of shape (n_sensors, 2)
        Sensor coordinates in pixel space (col, row)
    map_shape : tuple of int
        (height, width) of the map in pixels
    scale : float, optional
        Scaling factor to convert pixels to meters (default: 1.0)

    Returns
    -------
    ndarray of shape (height, width, n_sensors)
        Distance from each pixel to each sensor in meters

    Examples
    --------
    >>> sensors = np.array([[10, 20], [30, 40]])
    >>> distances = compute_distance_matrix(sensors, (100, 100), scale=5)
    >>> distances.shape
    (100, 100, 2)

    Notes
    -----
    This can be memory-intensive for large maps. Consider processing
    in batches if memory is limited.
    """
    height, width = map_shape
    n_sensors = len(sensor_locations)

    # Create coordinate grids
    rows, cols = np.mgrid[0:height, 0:width]

    # Initialize distance array
    distances = np.zeros((height, width, n_sensors))

    for i, sensor in enumerate(sensor_locations):
        sensor_col, sensor_row = sensor
        dx = (cols - sensor_col) * scale
        dy = (rows - sensor_row) * scale
        distances[:, :, i] = np.sqrt(dx**2 + dy**2)

        # Minimum distance of 1 meter to avoid singularities
        distances[:, :, i] = np.maximum(distances[:, :, i], 1.0)

    return distances


def compute_pairwise_sensor_distances(sensor_locations, scale=1.0):
    """
    Compute pairwise distances between all sensors.

    Parameters
    ----------
    sensor_locations : ndarray of shape (n_sensors, 2)
        Sensor coordinates
    scale : float, optional
        Scaling factor (default: 1.0)

    Returns
    -------
    ndarray of shape (n_sensors, n_sensors)
        Symmetric distance matrix

    Examples
    --------
    >>> sensors = np.array([[0, 0], [3, 0], [0, 4]])
    >>> dists = compute_pairwise_sensor_distances(sensors)
    >>> dists[0, 1]  # Distance from sensor 0 to 1
    3.0
    """
    n_sensors = len(sensor_locations)
    distances = np.zeros((n_sensors, n_sensors))

    for i in range(n_sensors):
        for j in range(i + 1, n_sensors):
            dist = euclidean_distance(sensor_locations[i],
                                       sensor_locations[j], scale)
            distances[i, j] = dist
            distances[j, i] = dist

    return distances
