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
                                 dedupe_distance_m=25.0,
                                 power_density_sigma_m=200.0, power_density_threshold=0.3,
                                 verbose=True,
                                 input_is_linear=False, solve_in_linear_domain=None,
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
        - If input_is_linear=False (default): Values are in dBm (e.g., -80 dBm)
        - If input_is_linear=True: Values are in linear scale (mW)
    map_shape : tuple of (height, width)
        Shape of the reconstruction grid
    scale : float, optional
        Pixel-to-meter scaling factor, default: 1.0
    np_exponent : float, optional
        Path loss exponent, default: 2.0 (free space)
    sigma : float, optional
        Shadowing standard deviation (dB), default: 4.5.
        Only used if whitening_method='spatial_corr_exp_decay'.
    delta_c : float, optional
        Correlation distance (meters), default: 400.
        Only used if whitening_method='spatial_corr_exp_decay'.
    lambda_reg : float, optional
        Sparsity regularization parameter, default: 0.01
        - Larger λ → sparser solution (fewer transmitters)
        - Smaller λ → denser solution (more transmitters)
    norm_exponent : float, optional
        Exponent applied to column norms for L1 penalty weighting, default: 4
        Weight for column i is: (||a_i||_2^norm_exponent) / max(||a_j||_2^norm_exponent)
        Higher values increase emphasis on path gain differences
    solver : {'auto', 'cvxpy', 'sklearn', 'scipy'}, optional
        Optimization solver, default: 'auto'
    whitening_method : {'spatial_corr_exp_decay', 'log_inv_power_diag', 'hetero_diag'}, optional
        Method for computing whitening matrix, default: 'spatial_corr_exp_decay'
        - 'spatial_corr_exp_decay': Use covariance-based whitening W = V^(-1/2) with exponential decay correlation
        - 'log_inv_power_diag': Use diagonal matrix W_jj = log10(1/p_j) based on observed powers
        - 'hetero_diag': Use heteroscedastic diagonal matrix V_kk = sigma_noise^2 + eta^2 * p_k^2
    - 'hetero_geo_aware': Use geometry-aware covariance V_kl(i) = sigma(i) * K(f_k, f_l) * sigma(j)
    sigma_noise : float, optional
        Noise floor variance for 'hetero_diag' and 'hetero_geo_aware'. Default 1e-13.
    eta : float, optional
        Scaling factor for signal-dependent variance in 'hetero_diag'/'hetero_geo_aware'. Default 0.5.
    proximity_weight : float, optional
        Strength of soft penalty for transmitters near sensors.
        Penalty weight = 1 + proximity_weight * exp(-dist^2 / (2*decay^2))
        Default: 0.0 (disabled)
    proximity_decay : float, optional
        Distance scale (meters) for proximity penalty decay.
        Default: 50.0 meters
    penalty_type : {'l1', 'log_sum', 'lp'}, optional
        Type of sparsity penalty. Default: 'l1'
    penalty_param : float, optional
        Parameter for 'lp' penalty (p value). Default: 0.5
    sparsity_epsilon : float, optional
        Small constant for 'log_sum' and 'lp' penalties. Default: 1e-6
    return_linear_scale : bool, optional
        Return power field in linear scale (mW), default: False (return dBm)
    model_type : str, optional
        Propagation model to use:
        - 'log_distance': Simple log-distance path loss model (default)
        - 'tirem': TIREM terrain-aware propagation model
        - 'raytracing': Sionna ray-tracing propagation model
    model_config_path : str, optional
        Path to propagation model configuration file.
        Required for model_type='tirem' (TIREM config) or 'raytracing' (Sionna config).
    feature_config_path : str, optional
        Path to TIREM configuration file specifically for computing geometric features
        (LOS, obstacles, etc.) used in 'hetero_geo_aware' whitening.
        If None, defaults to 'config/tirem_parameters.yaml' to ensure consistent caching
        regardless of the selected propagation model.
    n_jobs : int, optional
        Number of parallel jobs for TIREM computation, default: -1
    selection_method : {'max', 'cluster', 'power_cluster'}, optional
        Method for selecting the best candidate in GLRT solver. Default: 'max'
        - 'max': Select the single location with maximum GLRT score
        - 'cluster': Identify clusters of high-scoring locations and select 
          the centroid of the strongest cluster (by sum of scores)
        - 'power_cluster': First filter candidates to high power-density regions,
          then apply cluster selection. Uses residual power to adapt across iterations.
    cluster_distance_m : float, optional
        Maximum distance in meters to consider two candidates as part of the same cluster.
        Default: 100.0 meters. Only used when selection_method='cluster' or 'power_cluster'.
    cluster_threshold_fraction : float, optional
        Fraction of max score for candidate inclusion in clustering (e.g., 0.1 = 10%).
        Default: 0.1. Only used when selection_method='cluster' or 'power_cluster'.
    cluster_max_candidates : int, optional
        Maximum number of top-scoring candidates to consider for clustering and to 
        store in history for visualization. Default: 100. This controls both the 
        clustering candidate pool and how many candidates are plotted at each iteration.
    dedupe_distance_m : float, optional
        Distance threshold in meters for deduplicating transmitters after GLRT iterations.
        Transmitters within this distance of each other are merged, keeping the one
        added earliest. Default: 25.0. Set to 0 or None to disable deduplication.
        Only used when solver='glrt'.
    power_density_sigma_m : float, optional
        Characteristic distance scale in meters for power density Gaussian kernel.
        Default: 200.0. Only used when selection_method='power_cluster'.
    power_density_threshold : float, optional
        Fraction of max density below which candidates are excluded. Default: 0.3.
        E.g., 0.3 means only candidates in regions with density >= 30% of max density 
        are considered. Only used when selection_method='power_cluster'.
    verbose : bool, optional
        Print progress information, default: True
    input_is_linear : bool, optional
        If True, observed_powers_dBm is treated as linear power (mW) and not converted.
        Default: False (treat as dBm)
    solve_in_linear_domain : bool, optional
        If True, solve the optimization problem in linear domain: min ||W(At - p)||^2.
        If False, solve in log domain: min ||W(log(At) - log(p))||^2.
        If None (default), infers from input_is_linear (True if input_is_linear else False).

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
    """
    if verbose:
        print("\n" + "="*70)
        print("JOINT SPARSE SUPERPOSITION RECONSTRUCTION")
        print("="*70)

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
        print(f"  Scale: {scale} m/pixel")
        print(f"  Propagation model: {model_type}")
        if model_type == 'log_distance':
            print(f"  Path loss exponent: n_p = {np_exponent}")
        print(f"  Sparsity parameter: λ = {lambda_reg:.4e}")
        if proximity_weight > 0:
            print(f"  Proximity penalty: weight={proximity_weight}, decay={proximity_decay} m")
        print(f"  Penalty type: {penalty_type}")
        print(f"  Input domain: {'Linear (mW)' if input_is_linear else 'Logarithmic (dBm)'}")
        print(f"  Solver domain: {'Linear' if solve_in_linear_domain else 'Logarithmic'}")

    # Step 1: Convert observed powers from dBm to linear scale (mW)
    if verbose:
        print(f"\nStep 1: Processing observed powers...")
        if input_is_linear:
             print(f"  Input range: [{observed_powers_dBm.min():.2e}, {observed_powers_dBm.max():.2e}] mW")
        else:
             print(f"  Input range: [{observed_powers_dBm.min():.1f}, {observed_powers_dBm.max():.1f}] dBm")

    if input_is_linear:
        observed_powers_linear = observed_powers_dBm
    else:
        observed_powers_linear = dbm_to_linear(observed_powers_dBm)

    if verbose:
        print(f"  Output range: [{observed_powers_linear.min():.2e}, {observed_powers_linear.max():.2e}] mW")

    # Step 2: Build covariance matrix (only if needed)
    cov_matrix = None
    if whitening_method == 'spatial_corr_exp_decay':
        if verbose:
            print(f"\nStep 2: Building covariance matrix...")
            print(f"  Shadowing σ = {sigma} dB")
            print(f"  Correlation distance δ_c = {delta_c} m")

        cov_matrix = build_covariance_matrix(
            sensor_locations, sigma=sigma, delta_c=delta_c, scale=scale
        )
    elif verbose:
        print(f"\nStep 2: Skipping covariance matrix build (not needed for {whitening_method})...")

    # Step 3: Compute whitening matrix
    if verbose:
        print(f"\nStep 3: Computing whitening matrix...")
        print(f"  Method: {whitening_method}")

    if whitening_method == 'spatial_corr_exp_decay':
        W = compute_whitening_matrix(cov_matrix, method='cholesky', verbose=verbose)
    elif whitening_method == 'log_inv_power_diag':
        W = compute_whitening_matrix(
            None, method='log_inv_power_diag',
            observed_powers=observed_powers_linear, verbose=verbose
        )
    elif whitening_method == 'hetero_diag':
        W = compute_whitening_matrix(
            None, method='hetero_diag',
            observed_powers=observed_powers_linear,
            sigma_noise=sigma_noise, eta=eta, verbose=verbose
        )
    elif whitening_method == 'hetero_geo_aware':
        if verbose:
            print("  Using hetero_geo_aware whitening (dynamic covariance)...")
        # We don't compute a static W here. The solver will handle it dynamically.
        W = None
        
        # Compute geometric features
        if feature_config_path is None:
             # Default fallback ensures we share cache across different model selections
             # Try ../config/ first (notebooks), then config/ (script root)
             import os
             if os.path.exists('../config/tirem_parameters.yaml'):
                 feature_config_path = '../config/tirem_parameters.yaml'
             else:
                 feature_config_path = 'config/tirem_parameters.yaml'

        if verbose:
            print(f"  Computing geometric features using TIREM (config: {feature_config_path})...")
            
        from ..propagation.tirem_wrapper import TiremModel
        # Initialize TIREM model for features
        # Note: We use a separate config path here so that feature computation
        # (which always uses TIREM) can share a consistent cache even if 
        # model_config_path points to a Sionna config.
        tirem_features_model = TiremModel(feature_config_path)
        
        # Compute features
        # Features: (M, N, 4)
        geometric_features = tirem_features_model.compute_geometric_features(
            sensor_locations, map_shape, scale=scale, n_jobs=n_jobs, verbose=verbose
        )
        
        solver_kwargs['geometric_features'] = geometric_features
        solver_kwargs['whitening_method'] = whitening_method
        solver_kwargs['sigma_noise'] = sigma_noise
        solver_kwargs['eta'] = eta
        
    else:
        raise ValueError(
            f"Unknown whitening_method '{whitening_method}'. "
            "Choose 'spatial_corr_exp_decay', 'log_inv_power_diag', 'hetero_diag', or 'hetero_geo_aware'"
        )

    # Step 4: Build propagation matrix
    if verbose:
        print(f"\nStep 4: Building propagation matrix...")

    A_model = compute_propagation_matrix(
        sensor_locations, map_shape, scale=scale,
        model_type=model_type, config_path=model_config_path,
        np_exponent=np_exponent, vectorized=True, n_jobs=n_jobs, verbose=verbose
    )


    # Step 4.6: Compute proximity weights (soft penalty)
    spatial_weights = None
    if proximity_weight > 0:
        if verbose:
            print(f"\nStep 4.6: Computing proximity weights (weight={proximity_weight}, decay={proximity_decay}m)...")
        
        spatial_weights = np.ones(N)
        decay_pixels = proximity_decay / scale
        decay_pixels_sq = decay_pixels ** 2
        
        # Grid coordinates
        grid_rows, grid_cols = np.indices((height, width))
        grid_points = np.column_stack((grid_cols.ravel(), grid_rows.ravel()))
        
        # For each grid point, find distance to nearest sensor
        # This can be slow for large grids. Use KDTree or simple broadcasting if M is small.
        # Since M is usually small (<100), broadcasting is fine.
        
        # sensor_locations: (M, 2)
        # grid_points: (N, 2)
        
        # Compute min distance squared to any sensor for all grid points
        # We can do this efficiently by iterating over sensors and taking min
        min_dist_sq = np.full(N, np.inf)
        
        for sensor in sensor_locations:
            # sensor is (col, row)
            sc, sr = sensor
            # Squared distance from this sensor to all grid points
            # (x - sc)^2 + (y - sr)^2
            d_sq = (grid_points[:, 0] - sc)**2 + (grid_points[:, 1] - sr)**2
            min_dist_sq = np.minimum(min_dist_sq, d_sq)
            
        # Compute weights: w = 1 + alpha * exp(-d^2 / (2*sigma^2))
        # sigma = decay_pixels
        # d^2 / (2*sigma^2) = min_dist_sq / (2 * decay_pixels_sq)
        
        gaussian_term = np.exp(-min_dist_sq / (2 * decay_pixels_sq))
        spatial_weights = 1.0 + proximity_weight * gaussian_term
        
        if verbose:
            print(f"  Spatial weights range: [{spatial_weights.min():.2f}, {spatial_weights.max():.2f}]")
    if verbose:
        print(f"\nStep 5: Solving sparse reconstruction...")

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
            power_density_sigma_m=power_density_sigma_m,
            power_density_threshold=power_density_threshold,
            verbose=verbose,
            lambda_reg=lambda_reg,
            norm_exponent=norm_exponent,
            spatial_weights=spatial_weights,
            penalty_type=penalty_type,
            penalty_param=penalty_param,
            sparsity_epsilon=sparsity_epsilon,
            use_linear_objective=solve_in_linear_domain,
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
