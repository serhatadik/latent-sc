"""
Propagation matrix computation for linear superposition model.

This module computes the propagation matrix A_model ∈ ℝ^(M×N) where:
- M = number of sensors
- N = number of grid points (potential transmitter locations)
- A_model[j, i] = linear path gain from grid point i to sensor j

The linear path gain is derived from the log-distance path loss model:
    PL[dB] = PL₀ + 10·nₚ·log₁₀(d/d₀)
    A[j,i] = 10^(-PL[dB]/10)

This allows linear superposition: p^th = A_model · t
"""

import numpy as np
from ..utils.coordinates import euclidean_distance


def compute_linear_path_gain(distance, pi0=0, np_exponent=2, di0=1):
    """
    Compute linear path gain (not dB) from distance.

    Converts log-distance path loss to linear scale:
        PL[dB] = pi0 + 10·np·log₁₀(d/d₀)
        Gain = 10^(-PL/10)

    Parameters
    ----------
    distance : float or ndarray
        Distance(s) in meters
    pi0 : float, optional
        Reference path loss at reference distance (dB), default: 0
    np_exponent : float, optional
        Path loss exponent, default: 2 (free space)
    di0 : float, optional
        Reference distance in meters, default: 1

    Returns
    -------
    float or ndarray
        Linear path gain (dimensionless, typically << 1)

    Examples
    --------
    >>> gain = compute_linear_path_gain(100, np_exponent=2)
    >>> gain < 1.0  # Path gain attenuates signal
    True
    >>> gain > 0.0  # But is always positive
    True

    Notes
    -----
    The returned gain is in linear scale (not dB). For typical RF propagation,
    this will be a very small number (e.g., 10^-8 at 100m with n_p=2).
    """
    distance = np.array(distance, dtype=float)
    distance = np.maximum(distance, di0)  # Avoid log(0)

    # Path loss in dB
    path_loss_dB = pi0 + 10 * np_exponent * np.log10(distance / di0)

    # Convert to linear scale
    linear_gain = 10 ** (-path_loss_dB / 10)

    return linear_gain


def compute_propagation_matrix(sensor_locations, map_shape, scale=1.0,
                                np_exponent=2, pi0=0, di0=1,
                                vectorized=True, verbose=True):
    """
    Build propagation matrix A_model ∈ ℝ^(M×N) for all sensor-grid pairs.

    For the linear superposition model:
        p^th = A_model · t

    where:
    - p^th ∈ ℝ^M: predicted received powers at sensors (linear scale)
    - t ∈ ℝ^N: transmit powers at grid points (linear scale)
    - A_model[j, i]: linear path gain from grid point i to sensor j

    Parameters
    ----------
    sensor_locations : ndarray of shape (M, 2)
        Sensor coordinates in pixel space (col, row)
    map_shape : tuple of (height, width)
        Shape of the grid
    scale : float, optional
        Scaling factor to convert pixels to meters, default: 1.0
    np_exponent : float, optional
        Path loss exponent, default: 2
    pi0 : float, optional
        Reference path loss (dB), default: 0
    di0 : float, optional
        Reference distance (m), default: 1
    vectorized : bool, optional
        Use vectorized computation (faster), default: True
    verbose : bool, optional
        Print progress information, default: True

    Returns
    -------
    A_model : ndarray of shape (M, N)
        Propagation matrix with linear path gains

    Examples
    --------
    >>> sensors = np.array([[10, 20], [30, 40]])  # 2 sensors
    >>> A = compute_propagation_matrix(sensors, (100, 100), scale=5)
    >>> A.shape
    (2, 10000)
    >>> np.all(A >= 0)  # All gains are non-negative
    True
    >>> np.all(A <= 1)  # All gains are ≤ 1 (attenuation)
    True

    Notes
    -----
    - Grid points are indexed row-major: i = row * width + col
    - For large maps, this matrix can be memory-intensive (M×N floats)
    - Vectorized version is ~10-100x faster than loop-based
    - Returns gains in linear scale (not dB)
    """
    M = len(sensor_locations)
    height, width = map_shape
    N = height * width

    if verbose:
        print(f"Building propagation matrix: {M} sensors × {N} grid points")
        print(f"Matrix size: {M}×{N} = {M*N:,} elements ({M*N*8/1e6:.1f} MB)")

    A_model = np.zeros((M, N), dtype=np.float64)

    if vectorized:
        # Vectorized computation: much faster for large grids
        # Create grid of all coordinates
        rows_grid, cols_grid = np.mgrid[0:height, 0:width]
        rows_flat = rows_grid.ravel()  # Shape: (N,)
        cols_flat = cols_grid.ravel()  # Shape: (N,)

        for j, sensor in enumerate(sensor_locations):
            sensor_col, sensor_row = sensor

            # Compute distances to all grid points at once
            dx = (cols_flat - sensor_col) * scale
            dy = (rows_flat - sensor_row) * scale
            distances = np.sqrt(dx**2 + dy**2)  # Shape: (N,)

            # Compute linear path gains
            A_model[j, :] = compute_linear_path_gain(
                distances, pi0, np_exponent, di0
            )

            if verbose and (j + 1) % max(1, M // 10) == 0:
                print(f"  Processed {j+1}/{M} sensors...")

    else:
        # Loop-based computation: slower but more memory-efficient
        for j, sensor in enumerate(sensor_locations):
            sensor_col, sensor_row = sensor

            for i in range(N):
                # Convert linear index to 2D coordinates
                row = i // width
                col = i % width

                # Grid point location
                grid_location = [col, row]

                # Compute distance
                distance = euclidean_distance(
                    grid_location, sensor, scale
                )

                # Compute linear path gain
                A_model[j, i] = compute_linear_path_gain(
                    distance, pi0, np_exponent, di0
                )

            if verbose and (j + 1) % max(1, M // 10) == 0:
                print(f"  Processed {j+1}/{M} sensors...")

    if verbose:
        print(f"Propagation matrix built successfully")
        print(f"  Min gain: {A_model.min():.2e}")
        print(f"  Max gain: {A_model.max():.2e}")
        print(f"  Mean gain: {A_model.mean():.2e}")

    return A_model


def propagation_matrix_to_map(A_model, map_shape):
    """
    Reshape a row of propagation matrix back to 2D map for visualization.

    Parameters
    ----------
    A_model : ndarray of shape (M, N)
        Propagation matrix
    map_shape : tuple of (height, width)
        Shape of the grid

    Returns
    -------
    ndarray of shape (M, height, width)
        Each slice [j, :, :] is the path gain map for sensor j

    Examples
    --------
    >>> A = np.random.rand(3, 10000)  # 3 sensors, 100×100 grid
    >>> maps = propagation_matrix_to_map(A, (100, 100))
    >>> maps.shape
    (3, 100, 100)
    """
    M, N = A_model.shape
    height, width = map_shape

    if N != height * width:
        raise ValueError(
            f"Matrix size {N} doesn't match map shape {map_shape} "
            f"({height}×{width}={height*width})"
        )

    return A_model.reshape(M, height, width)
