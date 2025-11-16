"""Likelihood-based transmitter localization and signal estimation."""

import numpy as np
from .path_loss import compute_path_loss_vector, compute_pairwise_sensor_distances


def build_covariance_matrix(sensor_locations, sigma=4.5, delta_c=400, scale=1.0):
    """
    Build spatial covariance matrix with exponential decay.

    Implements Equation (5) from the paper:
        V_kl = σ² * exp(-d_kl / δ_c)

    Parameters
    ----------
    sensor_locations : ndarray of shape (n_sensors, 2)
        Sensor coordinates
    sigma : float, optional
        Standard deviation in dB (default: 4.5)
    delta_c : float, optional
        Correlation distance in meters (default: 400)
    scale : float, optional
        Scaling factor to convert coordinates to meters (default: 1.0)

    Returns
    -------
    ndarray of shape (n_sensors, n_sensors)
        Covariance matrix

    Examples
    --------
    >>> sensors = np.array([[0, 0], [10, 0], [0, 10]])
    >>> cov = build_covariance_matrix(sensors, sigma=4.5, delta_c=400, scale=5)
    >>> cov.shape
    (3, 3)
    >>> np.allclose(cov, cov.T)  # Should be symmetric
    True

    Notes
    -----
    The covariance matrix models spatial correlation in shadowing losses.
    Closer sensors have more correlated measurements.
    """
    # Compute pairwise distances
    distances = compute_pairwise_sensor_distances(sensor_locations, scale)

    # Build covariance matrix
    variance = sigma ** 2
    cov_matrix = variance * np.exp(-distances / delta_c)

    return cov_matrix


def compute_likelihood(error_vector, cov_matrix):
    """
    Compute Gaussian likelihood given error vector and covariance.

    Implements Equation (4) from the paper:
        L(p^th, V; p) = f(i|p) = (1 / sqrt(det(V)) * (2π)^m) *
                                   exp(-0.5 * e_i^T * V^(-1) * e_i)

    Parameters
    ----------
    error_vector : ndarray of shape (n_sensors,)
        Error between predicted and observed powers
    cov_matrix : ndarray of shape (n_sensors, n_sensors)
        Covariance matrix

    Returns
    -------
    float
        Likelihood value (unnormalized)

    Examples
    --------
    >>> error = np.array([1.0, -0.5, 0.3])
    >>> cov = np.eye(3) * 4.5**2  # Diagonal covariance
    >>> likelihood = compute_likelihood(error, cov)
    >>> likelihood > 0
    True

    Notes
    -----
    Returns unnormalized likelihood. Normalization happens when
    computing the probability mass function over all locations.
    """
    m = len(error_vector)

    # Compute inverse of covariance matrix
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        cov_inv = np.linalg.pinv(cov_matrix)

    # Determinant for normalization
    det_cov = np.linalg.det(cov_matrix)
    if det_cov <= 0:
        det_cov = 1e-10  # Avoid log(0)

    # Gaussian normalization factor
    normalization = 1.0 / (np.sqrt(det_cov) * (np.sqrt(2 * np.pi)) ** m)

    # Exponent term
    exponent = -0.5 * error_vector.T @ cov_inv @ error_vector

    likelihood = normalization * np.exp(exponent)

    return likelihood


def compute_transmitter_pmf(transmit_power_map, sensor_locations,
                             observed_powers, cov_matrix,
                             scale=1.0, np_exponent=2,
                             threshold=1e-10):
    """
    Compute probability mass function of transmitter location.

    Creates Figure 3b from the paper: "2D PMF of transmitter location"

    Computes likelihood for each potential transmitter location and
    normalizes to create a probability distribution.

    Parameters
    ----------
    transmit_power_map : ndarray of shape (height, width)
        Estimated transmit power at each pixel
    sensor_locations : ndarray of shape (n_sensors, 2)
        Sensor coordinates (col, row)
    observed_powers : ndarray of shape (n_sensors,)
        Observed power at each sensor in dB
    cov_matrix : ndarray of shape (n_sensors, n_sensors)
        Covariance matrix
    scale : float, optional
        Distance scaling factor (default: 1.0)
    np_exponent : float, optional
        Path loss exponent (default: 2)
    threshold : float, optional
        Minimum probability threshold (default: 1e-10)

    Returns
    -------
    ndarray of shape (height, width)
        Probability mass at each pixel (sums to 1)

    Examples
    --------
    >>> tx_map = np.ones((100, 100)) * 30
    >>> sensors = np.array([[10, 20], [30, 40]])
    >>> observed = np.array([-80, -85])
    >>> cov = np.eye(2) * 4.5**2
    >>> pmf = compute_transmitter_pmf(tx_map, sensors, observed, cov, scale=5)
    >>> abs(pmf.sum() - 1.0) < 1e-6  # Should sum to 1
    True
    """
    height, width = transmit_power_map.shape
    likelihood_map = np.zeros((height, width))

    print("Computing likelihood for each potential transmitter location...")

    for i in range(height):
        for j in range(width):
            transmitter_location = [j, i]  # (col, row) = (x, y)
            ti = transmit_power_map[i, j]

            # Compute predicted powers
            predicted = compute_path_loss_vector(ti, sensor_locations,
                                                  transmitter_location,
                                                  scale, np_exponent)

            # Compute error
            error = predicted - observed_powers

            # Compute likelihood
            likelihood_map[i, j] = compute_likelihood(error, cov_matrix)

    # Normalize to create PMF
    total_likelihood = np.sum(likelihood_map)

    if total_likelihood > 0:
        pmf = likelihood_map / total_likelihood
    else:
        # Uniform distribution if all likelihoods are zero
        pmf = np.ones_like(likelihood_map) / (height * width)

    # Apply threshold to avoid numerical issues
    pmf = np.maximum(pmf, threshold)

    # Renormalize after thresholding
    pmf = pmf / np.sum(pmf)

    return pmf


def estimate_received_power_map(transmit_power_map, pmf, sensor_locations,
                                 target_grid, scale=1.0, np_exponent=2,
                                 probability_threshold=1e-6):
    """
    Estimate received power at all target locations.

    Implements Equation (6) from the paper:
        p_est(j) = Σ_i p^th_j(t_i) * f(i|p)

    Creates Figure 3c from the paper: "Estimated signal strength"

    Parameters
    ----------
    transmit_power_map : ndarray of shape (height, width)
        Estimated transmit power at each potential transmitter location
    pmf : ndarray of shape (height, width)
        Probability mass function of transmitter location
    sensor_locations : ndarray of shape (n_sensors, 2)
        Not used in this version but kept for API consistency
    target_grid : ndarray of shape (height, width, 2)
        Grid of target coordinates (i, j) for each pixel
    scale : float, optional
        Distance scaling factor (default: 1.0)
    np_exponent : float, optional
        Path loss exponent (default: 2)
    probability_threshold : float, optional
        Only use transmitter locations with probability > threshold (default: 1e-6)

    Returns
    -------
    ndarray of shape (height, width)
        Estimated received power at each location (linear scale)

    Examples
    --------
    >>> tx_map = np.ones((50, 50)) * 30
    >>> pmf = np.ones((50, 50)) / 2500  # Uniform
    >>> sensors = np.array([[10, 10]])
    >>> # Create target grid
    >>> rows, cols = np.mgrid[0:50, 0:50]
    >>> target_grid = np.stack([rows, cols], axis=-1)
    >>> power_map = estimate_received_power_map(tx_map, pmf, sensors,
    ...                                          target_grid, scale=5)
    >>> power_map.shape
    (50, 50)

    Notes
    -----
    For efficiency, only considers transmitter locations with probability
    above the threshold. Returns power in linear scale (not dB).
    """
    height, width = transmit_power_map.shape
    power_estimate = np.zeros((height, width))

    # Find high-probability transmitter locations
    high_prob_indices = np.where(pmf > probability_threshold)
    n_locations = len(high_prob_indices[0])

    print(f"Estimating signal strength using {n_locations} high-probability transmitter locations...")

    # For each target pixel
    for target_row in range(height):
        for target_col in range(width):
            target_location = [target_col, target_row]  # (x, y)

            # Sum contributions from all probable transmitter locations
            power_sum = 0

            for idx in range(n_locations):
                tx_row = high_prob_indices[0][idx]
                tx_col = high_prob_indices[1][idx]

                transmitter_location = [tx_col, tx_row]
                ti = transmit_power_map[tx_row, tx_col]
                prob = pmf[tx_row, tx_col]

                # Compute path loss from this transmitter to target
                predicted_power = compute_path_loss_vector(
                    ti, np.array([target_location]),
                    transmitter_location, scale, np_exponent
                )[0]

                # Convert dB to linear and weight by probability
                power_linear = 10 ** (predicted_power / 10)
                power_sum += power_linear * prob

            power_estimate[target_row, target_col] = power_sum

    return power_estimate
