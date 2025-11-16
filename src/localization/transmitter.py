"""Transmit power estimation using optimization."""

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from joblib import Parallel, delayed
from .path_loss import compute_path_loss_vector


def compute_error_vector(ti, sensor_locations, transmitter_location,
                          observed_powers, scale=1.0, np_exponent=2):
    """
    Compute error vector between observed and predicted powers.

    Implements Equation (2) from the paper:
        e_i(t_i) = p^th(t_i) - p

    Parameters
    ----------
    ti : float
        Transmit power to evaluate
    sensor_locations : ndarray
        Sensor coordinates
    transmitter_location : array-like
        Transmitter coordinates
    observed_powers : ndarray
        Observed power at each sensor in dB
    scale : float, optional
        Distance scaling factor (default: 1.0)
    np_exponent : float, optional
        Path loss exponent (default: 2)

    Returns
    -------
    ndarray
        Error vector (predicted - observed)

    Examples
    --------
    >>> sensors = np.array([[0, 0], [10, 0]])
    >>> observed = np.array([-50, -60])
    >>> error = compute_error_vector(30, sensors, [5, 0], observed, scale=5)
    >>> error.shape
    (2,)
    """
    predicted = compute_path_loss_vector(ti, sensor_locations,
                                          transmitter_location,
                                          scale, np_exponent)
    return predicted - observed_powers


def minimize_transmit_power(sensor_locations, transmitter_location,
                             observed_powers, scale=1.0, np_exponent=2,
                             method='BFGS'):
    """
    Minimize transmit power to best fit observed measurements.

    Implements Equation (3) from the paper:
        t_i_hat = argmin ||e_i(t_i)||_2

    Parameters
    ----------
    sensor_locations : ndarray of shape (n_sensors, 2)
        Sensor coordinates
    transmitter_location : array-like of length 2
        Potential transmitter coordinates
    observed_powers : ndarray of shape (n_sensors,)
        Observed power at each sensor in dB
    scale : float, optional
        Distance scaling factor (default: 1.0)
    np_exponent : float, optional
        Path loss exponent (default: 2)
    method : str, optional
        Optimization method (default: 'BFGS')

    Returns
    -------
    float
        Optimal transmit power in dB

    Examples
    --------
    >>> sensors = np.array([[0, 0], [10, 0], [0, 10]])
    >>> tx_loc = [5, 5]
    >>> observed = np.array([-50, -55, -55])
    >>> t_opt = minimize_transmit_power(sensors, tx_loc, observed, scale=5)
    >>> isinstance(t_opt, float)
    True

    Notes
    -----
    Uses scipy.optimize.minimize with BFGS algorithm by default.
    Initial guess is the mean of observed powers plus path loss estimate.
    """
    def objective(ti):
        """L2 norm of error vector."""
        error = compute_error_vector(ti, sensor_locations, transmitter_location,
                                       observed_powers, scale, np_exponent)
        return np.sum(error ** 2)

    # Initial guess: mean observed power (rough estimate)
    x0 = np.mean(observed_powers) + 40  # Add offset for typical path loss

    result = minimize(objective, x0, method=method)
    return result.x[0]


def estimate_transmit_power_single_pixel(pixel_coords, sensor_locations,
                                          observed_powers, scale=1.0,
                                          np_exponent=2):
    """
    Estimate transmit power for a single pixel location.

    Helper function for parallel processing.

    Parameters
    ----------
    pixel_coords : tuple
        (row, col) coordinates of pixel
    sensor_locations : ndarray
        Sensor coordinates
    observed_powers : ndarray
        Observed powers at sensors
    scale : float, optional
        Distance scaling factor
    np_exponent : float, optional
        Path loss exponent

    Returns
    -------
    tuple
        (row, col, transmit_power)
    """
    row, col = pixel_coords
    transmitter_location = [col, row]  # Note: (col, row) for (x, y)

    t_opt = minimize_transmit_power(sensor_locations, transmitter_location,
                                     observed_powers, scale, np_exponent)

    return (row, col, t_opt)


def estimate_transmit_power_map(map_shape, sensor_locations, observed_powers,
                                 scale=1.0, np_exponent=2, n_jobs=-1,
                                 batch_size=100, verbose=True):
    """
    Estimate transmit power for all pixels in the map.

    Creates Figure 3a from the paper: "Distribution of estimated tx power"

    Parameters
    ----------
    map_shape : tuple of int
        (height, width) of the map
    sensor_locations : ndarray of shape (n_sensors, 2)
        Sensor coordinates in pixel space (col, row)
    observed_powers : ndarray of shape (n_sensors,)
        Observed power at each sensor in dB
    scale : float, optional
        Scaling factor to convert pixels to meters (default: 1.0)
    np_exponent : float, optional
        Path loss exponent (default: 2)
    n_jobs : int, optional
        Number of parallel jobs, -1 for all cores (default: -1)
    batch_size : int, optional
        Batch size for parallel processing (default: 100)
    verbose : bool, optional
        Show progress bar (default: True)

    Returns
    -------
    ndarray of shape (height, width)
        Estimated transmit power at each pixel location

    Examples
    --------
    >>> sensors = np.array([[10, 20], [30, 40], [50, 60]])
    >>> observed = np.array([-80, -85, -90])
    >>> tx_map = estimate_transmit_power_map((100, 100), sensors, observed, scale=5)
    >>> tx_map.shape
    (100, 100)

    Notes
    -----
    This can be computationally expensive. Uses parallel processing by default.
    For a 500×500 map with 10 sensors, expect ~5-10 minutes on a modern CPU.
    """
    height, width = map_shape
    transmit_power_map = np.zeros((height, width))

    # Create list of all pixel coordinates
    all_pixels = [(i, j) for i in range(height) for j in range(width)]

    # Process in parallel with progress bar
    if verbose:
        print(f"Estimating transmit power for {len(all_pixels)} pixels...")

    results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
        delayed(estimate_transmit_power_single_pixel)(
            pixel, sensor_locations, observed_powers, scale, np_exponent
        )
        for pixel in (tqdm(all_pixels) if verbose else all_pixels)
    )

    # Fill in the map
    for row, col, t_opt in results:
        transmit_power_map[row, col] = t_opt

    return transmit_power_map
