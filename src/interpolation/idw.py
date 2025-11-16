"""Inverse Distance Weighting (IDW) interpolation."""

import numpy as np
from ..utils.coordinates import euclidean_distance


def idw_weights(distances, power=2):
    """
    Calculate IDW interpolation weights.

    Implements Equation (8) from the paper:
        z_i = (1/d(x_i, x_0)) / Σ_j (1/d(x_j, x_0))

    Parameters
    ----------
    distances : ndarray
        Array of distances from known points to target point
    power : float, optional
        Power parameter for distance weighting (default: 2)

    Returns
    -------
    ndarray
        Normalized weights that sum to 1

    Examples
    --------
    >>> distances = np.array([1.0, 2.0, 4.0])
    >>> weights = idw_weights(distances, power=2)
    >>> abs(weights.sum() - 1.0) < 1e-10
    True
    >>> weights[0] > weights[1] > weights[2]  # Closer points have more weight
    True

    Notes
    -----
    Closer points receive higher weights. The power parameter controls
    how quickly weights decrease with distance.
    """
    # Avoid division by zero for very small distances
    distances = np.maximum(distances, 1e-10)

    # Compute weights
    weights = 1.0 / (distances ** power)

    # Normalize
    weights_sum = np.sum(weights)
    if weights_sum > 0:
        weights = weights / weights_sum
    else:
        # If all distances are zero, use uniform weights
        weights = np.ones_like(weights) / len(weights)

    return weights


def idw_interpolation(x_known, y_known, z_known, x_target, y_target, power=2):
    """
    Perform IDW interpolation at a single target point.

    Implements Equation (7) from the paper:
        ŝ_0 = Σ_i z_i * s_i

    Parameters
    ----------
    x_known : array-like
        X coordinates of known points
    y_known : array-like
        Y coordinates of known points
    z_known : array-like
        Values at known points
    x_target : float
        X coordinate of target point
    y_target : float
        Y coordinate of target point
    power : float, optional
        Power parameter (default: 2)

    Returns
    -------
    float
        Interpolated value at target point

    Examples
    --------
    >>> x = np.array([0, 1, 2])
    >>> y = np.array([0, 0, 0])
    >>> z = np.array([10, 20, 30])
    >>> value = idw_interpolation(x, y, z, 1.0, 0.0)
    >>> abs(value - 20.0) < 0.1  # Should be close to 20
    True
    """
    x_known = np.array(x_known)
    y_known = np.array(y_known)
    z_known = np.array(z_known)

    # Calculate distances from known points to target
    distances = np.sqrt((x_known - x_target)**2 + (y_known - y_target)**2)

    # Get weights
    weights = idw_weights(distances, power)

    # Interpolate
    interpolated_value = np.sum(weights * z_known)

    return interpolated_value


def interpolate_to_grid(x_known, y_known, z_known, grid_shape,
                        power=2, max_distance=None):
    """
    Interpolate values to a regular grid using IDW.

    Used for duty cycle interpolation in the paper (Section II.C).

    Parameters
    ----------
    x_known : array-like
        X coordinates of known points (in pixel coordinates)
    y_known : array-like
        Y coordinates of known points (in pixel coordinates)
    z_known : array-like
        Values at known points
    grid_shape : tuple of int
        (height, width) of output grid
    power : float, optional
        Power parameter (default: 2)
    max_distance : float, optional
        Maximum distance for interpolation. Points beyond this
        distance are set to NaN (default: None, no limit)

    Returns
    -------
    ndarray of shape grid_shape
        Interpolated values on grid

    Examples
    --------
    >>> x = np.array([10, 50, 90])
    >>> y = np.array([10, 50, 90])
    >>> z = np.array([100, 50, 0])
    >>> grid = interpolate_to_grid(x, y, z, (100, 100), power=2)
    >>> grid.shape
    (100, 100)
    >>> grid[50, 50]  # Should be close to 50
    50.0
    """
    height, width = grid_shape
    interpolated_grid = np.zeros((height, width))

    x_known = np.array(x_known)
    y_known = np.array(y_known)
    z_known = np.array(z_known)

    for i in range(height):
        for j in range(width):
            # Calculate distances to all known points
            distances = np.sqrt((x_known - j)**2 + (y_known - i)**2)

            # Check max_distance constraint
            if max_distance is not None:
                if np.all(distances > max_distance):
                    interpolated_grid[i, j] = np.nan
                    continue

            # Interpolate
            weights = idw_weights(distances, power)
            interpolated_grid[i, j] = np.sum(weights * z_known)

    return interpolated_grid


def idw_with_distance_threshold(x_known, y_known, z_known, grid_shape,
                                 max_distance, power=2):
    """
    IDW interpolation with radius-based search.

    Only uses points within max_distance for interpolation.
    Points with no neighbors within max_distance are set to NaN.

    Parameters
    ----------
    x_known : array-like
        X coordinates of known points
    y_known : array-like
        Y coordinates of known points
    z_known : array-like
        Values at known points
    grid_shape : tuple
        (height, width) of grid
    max_distance : float
        Maximum distance in pixels to search for neighbors
    power : float, optional
        Power parameter (default: 2)

    Returns
    -------
    ndarray
        Interpolated grid with NaN where no neighbors exist

    Examples
    --------
    >>> x = np.array([50])
    >>> y = np.array([50])
    >>> z = np.array([100])
    >>> grid = idw_with_distance_threshold(x, y, z, (100, 100), max_distance=10)
    >>> np.isnan(grid[0, 0])  # Far from known point
    True
    >>> not np.isnan(grid[50, 50])  # Close to known point
    True
    """
    height, width = grid_shape
    interpolated_grid = np.full((height, width), np.nan)

    x_known = np.array(x_known)
    y_known = np.array(y_known)
    z_known = np.array(z_known)

    for i in range(height):
        for j in range(width):
            # Calculate distances
            distances = np.sqrt((x_known - j)**2 + (y_known - i)**2)

            # Find neighbors within max_distance
            within_radius = distances <= max_distance

            if not np.any(within_radius):
                continue  # Leave as NaN

            # Use only nearby points
            local_distances = distances[within_radius]
            local_values = z_known[within_radius]

            # Interpolate
            weights = idw_weights(local_distances, power)
            interpolated_grid[i, j] = np.sum(weights * local_values)

    return interpolated_grid
