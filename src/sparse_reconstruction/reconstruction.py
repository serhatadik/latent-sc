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
from ..utils.coordinates import euclidean_distance


def joint_sparse_reconstruction(sensor_locations, observed_powers_dBm, map_shape,
                                 scale=1.0, np_exponent=2, sigma=4.5, delta_c=400,
                                 lambda_reg=0.01, norm_exponent=4, solver='auto',
                                 whitening_method='spatial_corr_exp_decay',
                                 sigma_noise=1e-13, eta=0.5,
                                 proximity_weight=50.0, proximity_decay=50.0,
                                 penalty_type='l1', penalty_param=0.5, sparsity_epsilon=1e-6,
                                 return_linear_scale=False,
                                 model_type='log_distance', model_config_path=None, feature_config_path=None, n_jobs=-1,
                                 selection_method='max', cluster_distance_m=100.0, cluster_threshold_fraction=0.1,
                                 cluster_max_candidates=100,
                                 dedupe_distance_m=60.0,
                                 use_power_filtering=False, power_density_sigma_m=200.0, power_density_threshold=0.3,
                                 max_tx_power_dbm=40.0, veto_margin_db=5.0,
                                 veto_threshold=1e-9, ceiling_penalty_weight=0.1,
                                 use_edf_penalty=False, edf_threshold=1.5,
                                 use_robust_scoring=False, robust_threshold=6.0,
                                 verbose=True,
                                 beam_width=1, pool_refinement=True, max_pool_size=50,
                                 input_is_linear=False, solve_in_linear_domain=None,
                                 observed_stds_dB=None, sigma_noise_floor=1e-15,
                                 **solver_kwargs):
    """
    Perform joint sparse reconstruction to estimate transmit power field.

    This is the main entry point for the sparse superposition approach.

    Parameters
    ----------
    sensor_locations : ndarray of shape (M, 2)
        Sensor coordinates in pixel space (col, row)
    observed_powers_dBm : ndarray of shape (M,)
        Observed powers at sensors.
    ...
    use_robust_scoring : bool, optional
        Enable Huber-like robust loss for GLRT residuals.
    robust_threshold : float, optional
         Threshold for robust clipping (in standardized units). Default: 6.0.
    """
    if verbose:
        print("\n" + "="*70)
        print("JOINT SPARSE SUPERPOSITION RECONSTRUCTION")
        print("="*70)

    # ... (rest of function body) ...

    # Determine solving domain
    if solve_in_linear_domain is None:
        solve_in_linear_domain = input_is_linear

    M = len(sensor_locations)
    height, width = map_shape
    N = height * width

    if verbose:
        print(f"\nProblem Configuration:")
        print(f"  Sensors: M = {M}")
        print(f"  Grid points: N = {N} ({height}×{width})")
        # ...
    # Step 1: Convert to linear mW (if necessary)
    if input_is_linear:
        observed_powers_linear = observed_powers_dBm
    else:
        from ..utils.conversions import dB_to_lin
        observed_powers_linear = dB_to_lin(observed_powers_dBm)

    # Step 2 & 3: Covariance and Whitening
    # Note: We pass model_type to handle different noise assumptions if needed,
    # but primarily whitening is about sensor noise/spatial correlation.
    if verbose:
        print(f"  Whitening Method: {whitening_method} (eta={eta})")

    spatial_weights = None
    # Handle diagonal whitening methods based on observations
    if whitening_method in ['hetero_diag', 'log_inv_power_diag']:
        W = compute_whitening_matrix(
            method=whitening_method,
            observed_powers=observed_powers_linear,
            sigma_noise=sigma_noise,
            eta=eta,
            verbose=verbose
        )
    elif whitening_method == 'hetero_diag_obs':
        if observed_stds_dB is None:
            raise ValueError("observed_stds_dB required for 'hetero_diag_obs' whitening method")
        if verbose:
            print(f"  Using observed std whitening (hetero_diag_obs)...")
        W = compute_whitening_matrix(
            method='hetero_diag_obs',
            observed_powers=observed_powers_linear,
            observed_stds_dB=observed_stds_dB,
            sigma_noise_floor=sigma_noise_floor,
            verbose=verbose
        )
    elif whitening_method == 'hetero_spatial':
        if verbose:
            print("  Using hetero_spatial whitening (heteroscedastic diag + spatial correlation)...")
            print(f"    sigma_noise: {sigma_noise:.2e}, eta: {eta:.2f}")
            print(f"    Correlation distance: {delta_c} m")

        # 1. Compute heteroscedastic standard deviations
        # V_ii = sigma_noise^2 + (eta * P_i)^2
        # std_i = sqrt(V_ii)
        v_diag_elements = sigma_noise**2 + (eta * observed_powers_linear)**2
        std_diag = np.sqrt(v_diag_elements)

        # 2. Compute spatial correlation matrix R
        # build_covariance_matrix with sigma=1.0 returns correlation matrix (diagonal = 1.0)
        R = build_covariance_matrix(
            sensor_locations, sigma=1.0, delta_c=delta_c, scale=scale
        )

        # 3. Compute Full Covariance V = D * R * D
        # V_ij = std_i * R_ij * std_j
        # We can compute this efficiently using broadcasting
        V = R * np.outer(std_diag, std_diag)

        if verbose:
            print(f"    Covariance V range: [{V.min():.2e}, {V.max():.2e}]")

        # 4. Compute Whitening Matrix W = V^(-1/2) using Cholesky
        # IMPORTANT: Set regularization=0.0 because V elements are extremely small (~1e-26) 
        # and default regularization (1e-10) would completely dominate the matrix structure,
        # leading to an effective Identity covariance and failed whitening.
        try:
            W = compute_whitening_matrix(V, method='cholesky', verbose=verbose, regularization=0.0)
        except np.linalg.LinAlgError:
            if verbose:
                print("  Cholesky failed with reg=0.0. Retrying with minimal regularization (1e-30)...")
            W = compute_whitening_matrix(V, method='cholesky', verbose=verbose, regularization=1e-30)

    # Default: Spatial covariance whitening (exponential decay) — homoscedastic
    else:
        if solve_in_linear_domain:
            # Linear-domain solving: sigma must match observation scale.
            # The dB-domain default (sigma=4.5) is ~10^16× too large for
            # linear powers of O(1e-9), making GLRT scores vanishingly small.
            # Use a single homoscedastic std derived from the observations.
            sigma_homo = np.sqrt(sigma_noise**2 + (eta * np.mean(observed_powers_linear))**2)

            if verbose:
                print(f"  Building homoscedastic spatial covariance (sigma_homo={sigma_homo:.2e}, delta_c={delta_c})")

            R = build_covariance_matrix(
                sensor_locations, sigma=1.0, delta_c=delta_c, scale=scale
            )
            V = sigma_homo**2 * R

            try:
                W = compute_whitening_matrix(V, method='cholesky', verbose=verbose, regularization=0.0)
            except np.linalg.LinAlgError:
                if verbose:
                    print("  Cholesky failed with reg=0.0. Retrying with minimal regularization (1e-30)...")
                W = compute_whitening_matrix(V, method='cholesky', verbose=verbose, regularization=1e-30)
        else:
            # dB-domain solving: use legacy sigma (dB-scale shadowing std)
            if verbose:
                print(f"  Building spatial covariance matrix (sigma={sigma}, delta_c={delta_c})")
            V = build_covariance_matrix(sensor_locations, sigma=sigma, delta_c=delta_c, scale=scale)
            W = compute_whitening_matrix(cov_matrix=V, method='cholesky', verbose=verbose)

    # Step 4: Compute Propagation Matrix (A)
    # This computes linear path gains from every pixel to every sensor
    if verbose:
        print(f"  Computing Propagation Matrix ({model_type})..." )

    A_model = compute_propagation_matrix(
        sensor_locations=sensor_locations,
        map_shape=map_shape,
        scale=scale,
        model_type=model_type,
        config_path=model_config_path,
        np_exponent=np_exponent,
        n_jobs=n_jobs,
        verbose=verbose
    )
    if solver == 'glrt':
        from .glrt_solver import solve_iterative_glrt
        t_est, solver_info = solve_iterative_glrt(
            A_model, W, observed_powers_linear,
            selection_method=selection_method,
            map_shape=map_shape,
            scale=scale,
            cluster_distance_m=cluster_distance_m,
            cluster_threshold_fraction=cluster_threshold_fraction,
            cluster_max_candidates=cluster_max_candidates,
            dedupe_distance_m=dedupe_distance_m,
            sensor_locations=sensor_locations,
            use_power_filtering=use_power_filtering,
            power_density_sigma_m=power_density_sigma_m,
            power_density_threshold=power_density_threshold,
            max_tx_power_dbm=max_tx_power_dbm,
            veto_margin_db=veto_margin_db,
            veto_threshold=veto_threshold,
            ceiling_penalty_weight=ceiling_penalty_weight,
            use_edf_penalty=use_edf_penalty,
            edf_threshold=edf_threshold,
            use_robust_scoring=use_robust_scoring,
            robust_threshold=robust_threshold,
            verbose=verbose,
            lambda_reg=lambda_reg,
            norm_exponent=norm_exponent,
            spatial_weights=spatial_weights,
            penalty_type=penalty_type,
            penalty_param=penalty_param,
            sparsity_epsilon=sparsity_epsilon,
            use_linear_objective=solve_in_linear_domain,
            beam_width=beam_width,
            pool_refinement=pool_refinement,
            max_pool_size=max_pool_size,
            **solver_kwargs
        )
    else:
        t_est, solver_info = solve_sparse_reconstruction(
            A_model, W, observed_powers_linear, lambda_reg,
            solver=solver, norm_exponent=norm_exponent,
            spatial_weights=spatial_weights,
            penalty_type=penalty_type, penalty_param=penalty_param,
            sparsity_epsilon=sparsity_epsilon,
            verbose=verbose,
            use_linear_objective=solve_in_linear_domain,
            **solver_kwargs
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

    # Always store A_model so downstream code (candidate analysis, validation)
    # can use the same propagation model that was used for localization
    info['A_model'] = A_model

    if verbose:
        info['W'] = W
        if 'V' in locals():
            info['cov_matrix'] = V

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
    # euclidean_distance is now imported at top level

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
