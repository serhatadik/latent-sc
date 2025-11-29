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

from src.propagation import LogDistanceModel, TiremModel
from src.propagation.log_distance import compute_linear_path_gain

def compute_propagation_matrix(sensor_locations, map_shape, scale=1.0,
                                model_type='log_distance', config_path=None,
                                np_exponent=2, pi0=0, di0=1,
                                vectorized=True, verbose=True):
    """
    Build propagation matrix A_model ∈ ℝ^(M×N) using the selected propagation model.

    Parameters
    ----------
    sensor_locations : ndarray of shape (M, 2)
        Sensor coordinates in pixel space (col, row)
    map_shape : tuple of (height, width)
        Shape of the grid
    scale : float, optional
        Scaling factor to convert pixels to meters, default: 1.0
    model_type : str, optional
        Propagation model to use: 'log_distance' (default) or 'tirem'
    config_path : str, optional
        Path to configuration file (required for 'tirem')
    np_exponent : float, optional
        Path loss exponent (for log_distance), default: 2
    pi0 : float, optional
        Reference path loss (dB) (for log_distance), default: 0
    di0 : float, optional
        Reference distance (m) (for log_distance), default: 1
    vectorized : bool, optional
        Use vectorized computation (for log_distance), default: True
    verbose : bool, optional
        Print progress information, default: True

    Returns
    -------
    A_model : ndarray of shape (M, N)
        Propagation matrix with linear path gains
    """
    if model_type == 'log_distance':
        model = LogDistanceModel(np_exponent=np_exponent, pi0=pi0, di0=di0, vectorized=vectorized)
        return model.compute_propagation_matrix(sensor_locations, map_shape, scale=scale, verbose=verbose)
    elif model_type == 'tirem':
        if config_path is None:
            # Default to standard location if not provided
            config_path = 'config/tirem_parameters.yaml'
        model = TiremModel(config_path)
        return model.compute_propagation_matrix(sensor_locations, map_shape, scale=scale, verbose=verbose)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
