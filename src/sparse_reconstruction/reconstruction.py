"""
Main reconstruction pipeline for joint sparse superposition method.

This module provides the high-level API for the sparse reconstruction approach,
tying together propagation matrix computation, whitening, and sparse solving.

Complete Pipeline:
-----------------
1. Convert observed powers from dBm to linear scale (mW)
2. Build covariance matrix V (reuse from existing localization module)
3. Compute whitening matrix W = V^(-1/2)
4. Build propagation matrix A_model with linear path gains
5. Solve: min_{t≥0} ‖W(A·t - p)‖₂² + λ‖t‖₁
6. Optionally convert result back to dBm for visualization
"""

import numpy as np
from .propagation_matrix import compute_propagation_matrix
from .whitening import compute_whitening_matrix
from .sparse_solver import solve_sparse_reconstruction
from ..localization.likelihood import build_covariance_matrix


def joint_sparse_reconstruction(sensor_locations, observed_powers_dBm, map_shape,
                                 scale=1.0, np_exponent=2, sigma=4.5, delta_c=400,
                                 lambda_reg=0.01, solver='auto',
                                 return_linear_scale=False, verbose=True,
                                 **solver_kwargs):
    """
    Perform joint sparse reconstruction to estimate transmit power field.

    This is the main entry point for the sparse superposition approach.

    Parameters
    ----------
    sensor_locations : ndarray of shape (M, 2)
        Sensor coordinates in pixel space (col, row)
    observed_powers_dBm : ndarray of shape (M,)
        Observed powers at sensors in dBm (e.g., -80 dBm)
    map_shape : tuple of (height, width)
        Shape of the reconstruction grid
    scale : float, optional
        Pixel-to-meter scaling factor, default: 1.0
    np_exponent : float, optional
        Path loss exponent, default: 2.0 (free space)
    sigma : float, optional
        Shadowing standard deviation (dB), default: 4.5
    delta_c : float, optional
        Correlation distance (meters), default: 400
    lambda_reg : float, optional
        Sparsity regularization parameter, default: 0.01
        - Larger λ → sparser solution (fewer transmitters)
        - Smaller λ → denser solution (more transmitters)
    solver : {'auto', 'cvxpy', 'sklearn', 'scipy'}, optional
        Optimization solver, default: 'auto'
    return_linear_scale : bool, optional
        Return power field in linear scale (mW), default: False (return dBm)
    verbose : bool, optional
        Print progress information, default: True
    **solver_kwargs : dict
        Additional arguments passed to solver

    Returns
    -------
    transmit_power_map : ndarray of shape (height, width)
        Estimated transmit power field
        - If return_linear_scale=False: in dBm
        - If return_linear_scale=True: in mW (linear scale)
    info : dict
        Reconstruction information including:
        - 'solver_info': details from sparse solver
        - 'n_nonzero': number of non-zero grid points
        - 'peak_location': (row, col) of strongest transmitter
        - 'peak_power': power at peak location
        - 'A_model': propagation matrix (if verbose=True)
        - 'W': whitening matrix (if verbose=True)

    Examples
    --------
    >>> # Setup
    >>> sensors = np.array([[10, 20], [30, 40], [50, 60]])
    >>> observed_dBm = np.array([-80, -85, -90])
    >>> map_shape = (100, 100)
    >>>
    >>> # Reconstruct
    >>> tx_map, info = joint_sparse_reconstruction(
    ...     sensors, observed_dBm, map_shape,
    ...     scale=5.0, lambda_reg=0.01, verbose=False
    ... )
    >>> tx_map.shape
    (100, 100)
    >>> info['n_nonzero'] < 10000  # Should be sparse
    True

    Notes
    -----
    Regularization Parameter Selection:
    - Start with λ ≈ 0.01 * ‖observed_powers‖
    - If solution is too dense (many non-zeros), increase λ
    - If solution is too sparse (all zeros), decrease λ
    - Typical range: 10^-4 to 10^-1

    Computational Complexity:
    - Propagation matrix: O(M·N)
    - Whitening matrix: O(M³)
    - Sparse solver: O(iterations × M·N)
    - Total: typically 10-60 seconds for 100×100 grid with 10 sensors
    """
    if verbose:
        print("\n" + "="*70)
        print("JOINT SPARSE SUPERPOSITION RECONSTRUCTION")
        print("="*70)

    M = len(sensor_locations)
    height, width = map_shape
    N = height * width

    if verbose:
        print(f"\nProblem Configuration:")
        print(f"  Sensors: M = {M}")
        print(f"  Grid points: N = {N} ({height}×{width})")
        print(f"  Scale: {scale} m/pixel")
        print(f"  Path loss exponent: n_p = {np_exponent}")
        print(f"  Sparsity parameter: λ = {lambda_reg:.4e}")

    # Step 1: Convert observed powers from dBm to linear scale (mW)
    if verbose:
        print(f"\nStep 1: Converting observed powers to linear scale...")
        print(f"  Input range: [{observed_powers_dBm.min():.1f}, {observed_powers_dBm.max():.1f}] dBm")

    observed_powers_linear = dbm_to_linear(observed_powers_dBm)

    if verbose:
        print(f"  Output range: [{observed_powers_linear.min():.2e}, {observed_powers_linear.max():.2e}] mW")

    # Step 2: Build covariance matrix
    if verbose:
        print(f"\nStep 2: Building covariance matrix...")
        print(f"  Shadowing σ = {sigma} dB")
        print(f"  Correlation distance δ_c = {delta_c} m")

    cov_matrix = build_covariance_matrix(
        sensor_locations, sigma=sigma, delta_c=delta_c, scale=scale
    )

    # Step 3: Compute whitening matrix
    if verbose:
        print(f"\nStep 3: Computing whitening matrix...")

    W = compute_whitening_matrix(cov_matrix, method='cholesky', verbose=verbose)

    # Step 4: Build propagation matrix
    if verbose:
        print(f"\nStep 4: Building propagation matrix...")

    A_model = compute_propagation_matrix(
        sensor_locations, map_shape, scale=scale,
        np_exponent=np_exponent, vectorized=True, verbose=verbose
    )

    # Step 5: Solve sparse reconstruction
    if verbose:
        print(f"\nStep 5: Solving sparse reconstruction...")

    t_est, solver_info = solve_sparse_reconstruction(
        A_model, W, observed_powers_linear, lambda_reg,
        solver=solver, verbose=verbose, **solver_kwargs
    )

    # Step 6: Reshape to map
    transmit_power_map_linear = t_est.reshape(height, width)

    # Step 7: Convert to dBm if requested
    if return_linear_scale:
        transmit_power_map = transmit_power_map_linear
        power_unit = "mW (linear)"
    else:
        transmit_power_map = linear_to_dbm(transmit_power_map_linear)
        power_unit = "dBm"

    # Compute statistics
    n_nonzero = solver_info['n_nonzero']
    peak_idx = np.argmax(t_est)
    peak_row = peak_idx // width
    peak_col = peak_idx % width
    peak_power_linear = t_est[peak_idx]
    peak_power_dBm = linear_to_dbm(peak_power_linear) if peak_power_linear > 0 else -np.inf

    if verbose:
        print(f"\n" + "="*70)
        print(f"RECONSTRUCTION COMPLETE")
        print(f"="*70)
        print(f"\nResults:")
        print(f"  Non-zero grid points: {n_nonzero}/{N} ({n_nonzero/N*100:.2f}%)")
        print(f"  Sparsity: {(1-n_nonzero/N)*100:.2f}%")
        print(f"  Peak location: (row={peak_row}, col={peak_col})")
        print(f"  Peak power: {peak_power_dBm:.1f} dBm ({peak_power_linear:.2e} mW)")
        print(f"  Output unit: {power_unit}")
        print(f"  Solver: {solver_info['solver_used']}")
        print(f"  Success: {solver_info['success']}")

    # Package info
    info = {
        'solver_info': solver_info,
        'n_nonzero': n_nonzero,
        'sparsity': 1 - n_nonzero / N,
        'peak_location': (peak_row, peak_col),
        'peak_power_linear': peak_power_linear,
        'peak_power_dBm': peak_power_dBm,
        'power_unit': power_unit,
    }

    if verbose:
        info['A_model'] = A_model
        info['W'] = W
        info['cov_matrix'] = cov_matrix

    return transmit_power_map, info


def reconstruct_signal_strength_map(transmit_power_map, A_model, map_shape,
                                     return_linear_scale=False, verbose=True):
    """
    Reconstruct signal strength at all locations using sparse transmit power field.

    Given the estimated sparse transmit power field t, compute received power at
    all grid locations: p_j = Σ_i A[j,i] * t[i]

    Parameters
    ----------
    transmit_power_map : ndarray of shape (height, width)
        Estimated transmit power field (in linear scale, mW)
    A_model : ndarray of shape (M, N)
        Propagation matrix (can be computed from any sensor to any grid point)
        Here we need all-to-all path gains
    map_shape : tuple of (height, width)
        Shape of the grid
    return_linear_scale : bool, optional
        Return in linear scale (mW), default: False (dBm)
    verbose : bool, optional
        Print progress, default: True

    Returns
    -------
    signal_strength_map : ndarray of shape (height, width)
        Estimated received signal strength at each location

    Notes
    -----
    This requires computing a full N×N propagation matrix, which can be
    memory-intensive for large grids. For 100×100 grid, this is 10,000×10,000 = 100M
    elements ≈ 800 MB.

    For large grids, use compute_signal_strength_at_points() to compute only at
    specific locations.
    """
    height, width = map_shape
    N = height * width

    if verbose:
        print(f"Reconstructing signal strength map...")
        print(f"  Warning: This requires N×N propagation matrix ({N}×{N} = {N*N:,} elements)")
        print(f"  Memory requirement: ~{N*N*8/1e9:.2f} GB")
        print(f"  Consider using compute_signal_strength_at_points() for large grids")

    # Flatten transmit power map
    t = transmit_power_map.ravel()

    # For signal strength reconstruction, we need path gains from all grid points
    # to all grid points. This is computationally expensive.
    # Instead, we can compute path gains on-the-fly for each target point.

    signal_strength_linear = np.zeros(N)

    for j in range(N):
        # For each target point j, compute path gain from all transmitter points i
        # and sum: p_j = Σ_i A[j,i] * t[i]
        # This is exactly what A_model does, but we need A_model to have N rows
        # (one for each target point), not just M rows (one for each sensor)

        # For now, return a placeholder
        # Full implementation would require computing full N×N matrix or
        # using on-the-fly computation
        pass

    raise NotImplementedError(
        "Full signal strength map reconstruction requires N×N propagation matrix. "
        "This is memory-intensive for large grids. "
        "Use compute_signal_strength_at_points() to compute at specific locations instead."
    )


def compute_signal_strength_at_points(transmit_power_map_linear, target_locations,
                                      scale=1.0, np_exponent=2,
                                      return_linear_scale=False, verbose=True):
    """
    Compute received signal strength at specific target locations.

    More memory-efficient than reconstruct_signal_strength_map() for evaluating
    at a subset of locations.

    Parameters
    ----------
    transmit_power_map_linear : ndarray of shape (height, width)
        Estimated transmit power field in linear scale (mW)
    target_locations : ndarray of shape (K, 2)
        Target coordinates (col, row) where to compute signal strength
    scale : float, optional
        Pixel-to-meter scaling, default: 1.0
    np_exponent : float, optional
        Path loss exponent, default: 2.0
    return_linear_scale : bool, optional
        Return in linear scale, default: False (dBm)
    verbose : bool, optional
        Print progress, default: True

    Returns
    -------
    signal_strengths : ndarray of shape (K,)
        Estimated received power at each target location

    Examples
    --------
    >>> tx_map = np.zeros((100, 100))
    >>> tx_map[50, 50] = 1.0  # Single transmitter at center
    >>> targets = np.array([[50, 50], [60, 50], [70, 50]])
    >>> powers = compute_signal_strength_at_points(tx_map, targets, scale=5.0)
    >>> powers[0] > powers[1] > powers[2]  # Decreasing with distance
    True
    """
    from .propagation_matrix import compute_linear_path_gain
    from ..utils.coordinates import euclidean_distance

    height, width = transmit_power_map_linear.shape
    K = len(target_locations)

    if verbose:
        print(f"Computing signal strength at {K} target locations...")

    signal_strengths_linear = np.zeros(K)

    # Find non-zero transmitter locations (for efficiency)
    nonzero_mask = transmit_power_map_linear > 1e-15
    nonzero_indices = np.argwhere(nonzero_mask)  # Shape: (n_nonzero, 2)
    n_nonzero = len(nonzero_indices)

    if verbose:
        print(f"  Found {n_nonzero} active transmitter locations")

    # For each target location
    for k, target in enumerate(target_locations):
        target_col, target_row = target

        # Sum contributions from all active transmitters
        for tx_row, tx_col in nonzero_indices:
            tx_location = [tx_col, tx_row]
            tx_power = transmit_power_map_linear[tx_row, tx_col]

            # Compute path gain
            distance = euclidean_distance(tx_location, target, scale)
            path_gain = compute_linear_path_gain(distance, np_exponent=np_exponent)

            # Add contribution
            signal_strengths_linear[k] += path_gain * tx_power

    # Convert to dBm if requested
    if return_linear_scale:
        return signal_strengths_linear
    else:
        return linear_to_dbm(signal_strengths_linear)


# ============================================================================
# Utility Functions
# ============================================================================

def dbm_to_linear(power_dBm):
    """
    Convert power from dBm to linear scale (mW).

    P[mW] = 10^(P[dBm] / 10)

    Parameters
    ----------
    power_dBm : float or ndarray
        Power in dBm

    Returns
    -------
    float or ndarray
        Power in mW (linear scale)

    Examples
    --------
    >>> dbm_to_linear(0)  # 0 dBm = 1 mW
    1.0
    >>> dbm_to_linear(-30)  # -30 dBm = 0.001 mW = 1 μW
    0.001
    >>> dbm_to_linear(30)  # 30 dBm = 1000 mW = 1 W
    1000.0
    """
    return 10 ** (power_dBm / 10)


def linear_to_dbm(power_mW):
    """
    Convert power from linear scale (mW) to dBm.

    P[dBm] = 10 * log₁₀(P[mW])

    Parameters
    ----------
    power_mW : float or ndarray
        Power in mW (linear scale)

    Returns
    -------
    float or ndarray
        Power in dBm

    Examples
    --------
    >>> linear_to_dbm(1.0)  # 1 mW = 0 dBm
    0.0
    >>> linear_to_dbm(0.001)  # 0.001 mW = -30 dBm
    -30.0
    >>> linear_to_dbm(1000.0)  # 1000 mW = 30 dBm
    30.0

    Notes
    -----
    Returns -inf for zero or negative input (mathematically undefined).
    """
    with np.errstate(divide='ignore'):
        return 10 * np.log10(np.maximum(power_mW, 1e-20))  # Avoid log(0)
