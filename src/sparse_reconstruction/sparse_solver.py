"""
Sparse reconstruction solver for joint transmitter localization.

Solves the constrained LASSO problem:
    min  ‖W(A·t - p)‖₂² + λ‖t‖₁
    t≥0

where:
- A ∈ ℝ^(M×N): propagation matrix
- W ∈ ℝ^(M×M): whitening matrix
- p ∈ ℝ^M: observed powers (linear scale)
- t ∈ ℝ^N: transmit power field (to be recovered)
- λ ≥ 0: sparsity regularization parameter

Provides multiple solver implementations with automatic fallback:
1. CVXPY: Most accurate, handles constraints elegantly
2. scikit-learn: Fast for standard LASSO, uses coordinate descent
3. scipy: General-purpose optimizer, always available
"""

import numpy as np
import warnings


def solve_sparse_reconstruction(A_model, W, observed_powers, lambda_reg,
                                 solver='scipy', verbose=True, norm_exponent=4,
                                 enable_reweighting=False, max_reweight_iter=5,
                                 reweight_epsilon=1e-12, reweight_epsilon_scale=1e-3,
                                 convergence_tol=1e-4,
                                 sparsity_threshold=0, gamma=0.0, max_l2_norm=None,
                                 exclusion_mask=None, spatial_weights=None,
                                 penalty_type='l1', penalty_param=0.5, sparsity_epsilon=1e-6,
                                 use_linear_objective=False,
                                 **solver_kwargs):
    """
    Solve sparse reconstruction problem with automatic solver selection.

    Solves:
        min  ‖W(log10(A·t) - log10(p))‖₂² + R(t)  (if use_linear_objective=False)
        min  ‖W(A·t - p)‖₂² + R(t)                (if use_linear_objective=True)
        t≥0
    
    where R(t) is the sparsity penalty:
    - 'l1': λ‖t‖₁
    - 'log_sum': λ∑log(t_i + ε)
    - 'lp': λ∑(t_i + ε)^p

    Parameters
    ----------
    A_model : ndarray of shape (M, N)
        Propagation matrix with linear path gains
    W : ndarray of shape (M, M)
        Whitening matrix (W = V^(-1/2))
    observed_powers : ndarray of shape (M,)
        Observed powers at sensors (linear scale, e.g., mW)
    lambda_reg : float or ndarray of shape (N,)
        Sparsity regularization parameter (λ ≥ 0)
        - If float: uniform regularization for all grid points
        - If ndarray: element-wise regularization weights (weighted L1)
        - λ = 0: standard least squares (dense solution)
        - λ > 0: encourages sparsity (fewer active transmitters)
        - λ → ∞: forces all-zero solution
    solver : {'scipy'}, optional
        Solver to use. Defaults to 'scipy'.
        Note: 'cvxpy' and 'sklearn' are NOT supported for the log-domain objective.
    verbose : bool, optional
        Print solver information, default: True
    norm_exponent : float, optional
        Exponent applied to column norms for L1 penalty weighting.
        Weight for column i is: (||a_i||_2^norm_exponent) / max(||a_j||_2^norm_exponent)
        Higher values increase emphasis on path gain differences. Default: 4
    enable_reweighting : bool, optional
        Enable iterative reweighting for enhanced sparsity. Default: False
    max_reweight_iter : int, optional
        Maximum number of reweighting iterations. Default: 5
    reweight_epsilon : float, optional
        Minimum damping factor for reweighting. Default: 1e-12
    reweight_epsilon_scale : float, optional
        Scaling factor for adaptive epsilon based on signal magnitude.
        epsilon = max(reweight_epsilon, max(|t|) * reweight_epsilon_scale)
        Default: 1e-3
    convergence_tol : float, optional
        Relative change threshold for convergence. Default: 1e-4
    sparsity_threshold : float, optional
        Threshold for hard sparsity - values below this are set to exactly zero.
        Applied to final solution. Default: 0 (no thresholding)
    gamma : float, optional
        Coefficient for negative L2 regularization term: -gamma * ||t||_2^2
        This encourages larger transmit power values. Default: 0.0 (disabled)
    max_l2_norm : float, optional
        Maximum L2 norm constraint: ||t||_2 ≤ max_l2_norm
        When set, automatically switches to 'trust-constr' solver. Default: None (no constraint)
    exclusion_mask : ndarray of shape (N,), bool, optional
        Mask indicating grid points that must be zero (True = excluded/zero).
        Used to prevent trivial solutions near sensors. Default: None
    spatial_weights : ndarray of shape (N,), optional
        Additional spatial regularization weights.
        Multiplied with sensitivity weights. Default: None (ones)
    penalty_type : {'l1', 'log_sum', 'lp'}, optional
        Type of sparsity penalty. Default: 'l1'
    penalty_param : float, optional
        Parameter for 'lp' penalty (p value). Default: 0.5
    sparsity_epsilon : float, optional
        Small constant for 'log_sum' and 'lp' penalties. Default: 1e-6
    use_linear_objective : bool, optional
        If True, solve in linear domain: min ||W(At - p)||^2.
        If False (default), solve in log domain: min ||W(log(At) - log(p))||^2.
    **solver_kwargs
        Additional keyword arguments passed to specific solver

    Returns
    -------
    t_est : ndarray of shape (N,)
        Estimated transmit powers (linear scale)
    info : dict
        Solver information:
        - 'solver_used': str, name of solver used
        - 'objective_value': float, final objective value
        - 'n_nonzero': int, number of non-zero transmit powers
        - 'sparsity': float, fraction of zero transmit powers
        - 'success': whether optimization succeeded

    Examples
    --------
    >>> M, N = 10, 100  # 10 sensors, 100 grid points
    >>> A = np.random.rand(M, N) * 0.01  # Small path gains
    >>> W = np.eye(M)  # Identity whitening
    >>> p = np.random.rand(M) * 1e-8  # Observed powers
    >>> t_est, info = solve_sparse_reconstruction(A, W, p, lambda_reg=0.01)
    >>> info['n_nonzero'] < N  # Solution should be sparse
    True
    >>> np.all(t_est >= 0)  # Solution should be non-negative
    True
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Sparse Reconstruction Solver ({'Linear' if use_linear_objective else 'Log'}-Domain)")
        print(f"{'='*60}")
        print(f"Problem size: M={A_model.shape[0]} sensors, N={A_model.shape[1]} grid points")
        if np.isscalar(lambda_reg):
            print(f"Sparsity parameter: lambda = {lambda_reg:.2e}")
        else:
            print(f"Sparsity parameter: lambda = [array of shape {lambda_reg.shape}]")
            print(f"  Mean lambda: {np.mean(lambda_reg):.2e}")
            print(f"  Max lambda: {np.max(lambda_reg):.2e}")
        print(f"Penalty type: {penalty_type}")
        if penalty_type == 'lp':
            print(f"  p = {penalty_param}")
        if exclusion_mask is not None:
            n_excluded = np.sum(exclusion_mask)
            print(f"Exclusion zone: {n_excluded} grid points forced to zero ({n_excluded/A_model.shape[1]*100:.1f}%)")

    # Validate inputs
    M, N = A_model.shape
    if W.shape != (M, M):
        raise ValueError(f"Whitening matrix shape {W.shape} incompatible with A_model shape {A_model.shape}")
    if observed_powers.shape != (M,):
        raise ValueError(f"Observed powers shape {observed_powers.shape} must be ({M},)")
    if np.any(lambda_reg < 0):
        raise ValueError(f"Regularization parameter must be non-negative")
    if exclusion_mask is not None and exclusion_mask.shape != (N,):
        raise ValueError(f"Exclusion mask shape {exclusion_mask.shape} must be ({N},)")

    if solver != 'scipy':
        if verbose:
            print(f"Warning: Solver '{solver}' is not supported for log-domain optimization. Falling back to 'scipy'.")
        solver = 'scipy'

    # Use scipy (L-BFGS-B)
    if solver == 'scipy':
        t_est, info = solve_sparse_reconstruction_scipy(
            A_model, W, observed_powers, lambda_reg,
            verbose=verbose,
            norm_exponent=norm_exponent,
            enable_reweighting=enable_reweighting,
            max_reweight_iter=max_reweight_iter,
            reweight_epsilon=reweight_epsilon,
            reweight_epsilon_scale=reweight_epsilon_scale,
            convergence_tol=convergence_tol,
            exclusion_mask=exclusion_mask,
            spatial_weights=spatial_weights,
            penalty_type=penalty_type,
            penalty_param=penalty_param,
            sparsity_epsilon=sparsity_epsilon,
            use_linear_objective=use_linear_objective,
            **solver_kwargs
        )
    else:
        raise ValueError(f"Unknown solver '{solver}'. Supported: 'scipy'")


    return t_est, info


def solve_sparse_reconstruction_cvxpy(A_model, W, observed_powers, lambda_reg,
                                       verbose=True, solver_name=None, **solver_options):
    """
    Deprecated: CVXPY solver does not support the non-convex log-domain objective.
    """
    raise NotImplementedError("CVXPY solver does not support the non-convex log-domain objective.")


def solve_sparse_reconstruction_sklearn(A_model, W, observed_powers, lambda_reg,
                                         verbose=True, max_iter=10000, tol=1e-6,
                                         **lasso_kwargs):
    """
    Deprecated: sklearn Lasso solver does not support the non-convex log-domain objective.
    """
    raise NotImplementedError("sklearn Lasso solver does not support the non-convex log-domain objective.")


def solve_sparse_reconstruction_scipy(A_model, W, observed_powers, lambda_reg,
                                       verbose=True, max_iter=1000, epsilon=1e-20, 
                                       norm_exponent=4, enable_reweighting=False,
                                       max_reweight_iter=5, reweight_epsilon=1e-12,
                                       reweight_epsilon_scale=1e-3,
                                       convergence_tol=1e-4, sparsity_threshold=0,
                                       gamma=0.0, max_l2_norm=None, exclusion_mask=None, 
                                       spatial_weights=None, 
                                       penalty_type='l1', penalty_param=0.5, sparsity_epsilon=1e-6,
                                       use_linear_objective=False,
                                       **scipy_kwargs):
    """
    Solve sparse reconstruction using scipy.optimize (L-BFGS-B).

    Objective:
        If use_linear_objective=False:
            min ‖W(log10(A·t + ε) - log10(p + ε))‖₂² + R(t)
        If use_linear_objective=True:
            min ‖W(A·t - p)‖₂² + R(t)
        
        t≥0

    Parameters
    ----------
    A_model : ndarray of shape (M, N)
        Propagation matrix
    W : ndarray of shape (M, M)
        Whitening matrix
    observed_powers : ndarray of shape (M,)
        Observed powers (linear scale)
    lambda_reg : float or ndarray
        Regularization parameter (scalar or vector)
    verbose : bool, optional
        Print solver information, default: True
    max_iter : int, optional
        Maximum L-BFGS iterations, default: 1000
    epsilon : float, optional
        Small constant to avoid log(0), default: 1e-20
    norm_exponent : float, optional
        Exponent for sensitivity weighting. Default: 4
    enable_reweighting : bool, optional
        Enable iterative reweighting for enhanced sparsity. Default: False
    max_reweight_iter : int, optional
        Maximum number of reweighting iterations. Default: 5
    reweight_epsilon : float, optional
        Minimum damping factor for reweighting. Default: 1e-12
    reweight_epsilon_scale : float, optional
        Scaling factor for adaptive epsilon. Default: 1e-3
    convergence_tol : float, optional
        Relative change threshold for convergence. Default: 1e-4
    sparsity_threshold : float, optional
        Threshold for hard sparsity - values below this are set to exactly zero.
        Applied to final solution. Default: 0 (no thresholding)
    gamma : float, optional
        Coefficient for negative L2 regularization term: -gamma * ||t||_2^2
        This encourages larger transmit power values. Default: 0.0 (disabled)
    max_l2_norm : float, optional
        Maximum L2 norm constraint: ||t||_2 ≤ max_l2_norm
        When set, automatically switches to 'trust-constr'. Default: None (no constraint)
    exclusion_mask : ndarray of shape (N,), bool, optional
        Mask indicating grid points that must be zero (True = excluded/zero).
    spatial_weights : ndarray of shape (N,), optional
        Additional spatial regularization weights.
    penalty_type : {'l1', 'log_sum', 'lp'}, optional
        Type of sparsity penalty. Default: 'l1'
    penalty_param : float, optional
        Parameter for 'lp' penalty (p value). Default: 0.5
    sparsity_epsilon : float, optional
        Small constant for 'log_sum' and 'lp' penalties. Default: 1e-6
    use_linear_objective : bool, optional
        If True, solve in linear domain. Default: False
    **scipy_kwargs
        Additional arguments for scipy.optimize.minimize

    Returns
    -------
    t_est : ndarray of shape (N,)
        Estimated transmit powers (linear scale)
    info : dict
        Solver information
    """
    from scipy.optimize import minimize, NonlinearConstraint, BFGS

    M, N = A_model.shape

    # Always use L-BFGS-B (most memory-efficient)
    # L2 constraint will be handled via barrier penalty method in objective
    scipy_method = 'L-BFGS-B'
    
    if verbose:
        if max_l2_norm is not None:
            print(f"Using scipy L-BFGS-B solver with barrier penalty for L2 constraint ||t||_2 ≤ {max_l2_norm}...")
        else:
            print(f"Using scipy L-BFGS-B solver ({'Linear' if use_linear_objective else 'Log'}-Domain)...")

    # Precompute constant terms
    if not use_linear_objective:
        # log10(p + epsilon)
        log_p = np.log10(observed_powers + epsilon)
        p_tilde = W @ log_p
    else:
        # Linear domain: W @ p
        p_whitened = W @ observed_powers

    # Precompute weights for L1 penalty (cancel path loss bias)
    # Omega_ii = ||a_i||_2^norm_exponent
    column_norms = np.linalg.norm(A_model, axis=0)
    column_norms_powered = column_norms ** norm_exponent
    if column_norms_powered.max() > 0:
        sensitivity_weights = column_norms_powered / column_norms_powered.max()
    else:
        sensitivity_weights = np.ones(N)

    # Apply spatial weights if provided
    if spatial_weights is not None:
        if spatial_weights.shape != (N,):
            raise ValueError(f"spatial_weights shape {spatial_weights.shape} must be ({N},)")
        sensitivity_weights *= spatial_weights
        if verbose:
            print(f"Applied spatial weights: range [{spatial_weights.min():.2f}, {spatial_weights.max():.2f}]")

    # Initialize for iterative reweighting
    if enable_reweighting:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Iterative Reweighting Enabled")
            print(f"{'='*60}")
            print(f"Max iterations: {max_reweight_iter}")
            print(f"Min damping factor (epsilon): {reweight_epsilon:.2e}")
            print(f"Damping scale: {reweight_epsilon_scale:.2e}")
            print(f"Convergence tolerance: {convergence_tol:.2e}")
            print(f"\nIteration 0: Initial solve with sensitivity weights only")
    
    # Track iterations
    t_prev = None
    actual_iterations = 0
    converged = False
    
    # Iterative reweighting loop
    for reweight_iter in range(max_reweight_iter + 1 if enable_reweighting else 1):
        # Update weights based on previous solution (skip for iteration 0)
        if reweight_iter == 0 or not enable_reweighting:
            # Iteration 0: Use only sensitivity weights
            weights = sensitivity_weights.copy()
        else:
            # Iterations 1+: Combine sensitivity and sparsity weights
            # Omega_ii^(k) = ||a_i||_2^c / (|t_i^(k-1)| + epsilon)
            
            # Adaptive epsilon: scale with signal magnitude
            max_t = np.max(np.abs(t_prev))
            current_epsilon = max(reweight_epsilon, max_t * reweight_epsilon_scale)
            
            sparsity_weights = 1.0 / (np.abs(t_prev) + current_epsilon)
            weights = sensitivity_weights * sparsity_weights
            
            # Normalize so that the minimum weight (for largest signal) is 1.0
            if weights.min() > 0:
                weights = weights / weights.min()
            elif weights.max() > 0:
                weights = weights / weights.max()
            
            if verbose:
                print(f"\nIteration {reweight_iter}: Refining sparsity")
                print(f"  Adaptive epsilon: {current_epsilon:.2e} (max_t={max_t:.2e})")
                print(f"  Previous nonzeros: {np.sum(np.abs(t_prev) > 1e-11)}")
                print(f"  Weight range: [{weights.min():.2e} / {weights.max():.2e}]")

        # Prepare regularization vector for this iteration
        if np.isscalar(lambda_reg):
            # Apply computed weights to scalar lambda
            lambda_vec = lambda_reg * weights
        else:
            # Use provided vector as is (assume user handled weighting)
            lambda_vec = lambda_reg

        def objective(t):
            """
            Objective function: Data Fidelity + Regularization + Constraints
            """
            # 1. Compute A·t
            At = A_model @ t
            
            if not use_linear_objective:
                # Log Domain
                # 2. Compute log10(A·t + ε)
                log_At = np.log10(At + epsilon)
                
                # 3. Compute residual vector in log domain: log(At) - log(p)
                log_diff = log_At - log_p
                
                # 4. Apply whitening: W(log_diff)
                whitened_diff = W @ log_diff
            else:
                # Linear Domain
                # 2. Compute residual: At - p
                diff = At - observed_powers
                
                # 3. Apply whitening: W(diff)
                whitened_diff = W @ diff
            
            # 5. Data fidelity term: squared L2 norm
            data_term = np.sum(whitened_diff**2)
            
            # 6. Regularization term: Weighted penalty
            if penalty_type == 'l1':
                regularization_term = np.sum(lambda_vec * np.abs(t))
            elif penalty_type == 'log_sum':
                regularization_term = np.sum(lambda_vec * np.log(t + sparsity_epsilon))
            elif penalty_type == 'lp':
                regularization_term = np.sum(lambda_vec * (t + sparsity_epsilon)**penalty_param)
            else:
                regularization_term = np.sum(lambda_vec * np.abs(t))
            
            # 7. Negative L2 regularization: -γ‖t‖₂² (with overflow protection)
            t_squared_sum = np.sum(t**2)
            t_squared_sum = np.clip(t_squared_sum, 0, 1e100)
            negative_l2_term = -gamma * t_squared_sum
            
            # 8. Exponential barrier for L2 constraint
            barrier_term = 0.0
            if max_l2_norm is not None:
                t_norm_squared = t_squared_sum  # Reuse computation
                max_norm_squared = max_l2_norm**2
                norm_ratio = t_norm_squared / max_norm_squared if max_norm_squared > 0 else 0
                
                if norm_ratio > 0.5:
                    barrier_coeff = max(1e3, 1e6 / (1 + gamma * 1e6))
                    exponent = 10 * (norm_ratio - 1)
                    exponent = np.clip(exponent, -50, 50)
                    barrier_term = barrier_coeff * (np.exp(exponent) - 1)
                
            return data_term + regularization_term + negative_l2_term + barrier_term

        def gradient(t):
            """
            Gradient of objective including exponential barrier.
            """
            # Forward pass (recompute needed parts)
            At = A_model @ t
            
            if not use_linear_objective:
                # Log Domain Gradient
                u = At + epsilon
                log_At = np.log10(u)
                log_diff = log_At - log_p
                whitened_diff = W @ log_diff  # This is r
                
                # Backprop
                # 1. W^T * r
                WTr = W.T @ whitened_diff
                
                # 2. diag(1/u) * W^T * r  => WTr / u (element-wise division)
                term2 = WTr / u
                
                # 3. A^T * term2
                grad_data_part = A_model.T @ term2
                
                # 4. Scale by 2 / ln(10)
                grad_data = (2 / np.log(10)) * grad_data_part
            else:
                # Linear Domain Gradient
                # f = ||W(At - p)||^2
                # grad = 2 A^T W^T W (At - p)
                diff = At - observed_powers
                whitened_diff = W @ diff # r_w
                
                # W^T * r_w
                WTr = W.T @ whitened_diff
                
                # 2 A^T * WTr
                grad_data = 2 * A_model.T @ WTr
            
            # Regularization gradient
            if penalty_type == 'l1':
                grad_reg = lambda_vec * np.sign(t)
            elif penalty_type == 'log_sum':
                grad_reg = lambda_vec / (t + sparsity_epsilon)
            elif penalty_type == 'lp':
                grad_reg = lambda_vec * penalty_param * (t + sparsity_epsilon)**(penalty_param - 1)
            else:
                grad_reg = lambda_vec * np.sign(t)
            
            # Negative L2 gradient: d/dt(-γ‖t‖₂²) = -2γt
            grad_negative_l2 = -2 * gamma * t
            
            # Exponential barrier gradient
            grad_barrier = np.zeros_like(t)
            if max_l2_norm is not None:
                t_norm_squared = np.sum(t**2)
                max_norm_squared = max_l2_norm**2
                norm_ratio = t_norm_squared / max_norm_squared if max_norm_squared > 0 else 0
                
                if norm_ratio > 0.5:
                    barrier_coeff = max(1e3, 1e6 / (1 + gamma * 1e6))
                    exponent = 10 * (norm_ratio - 1)
                    exponent = np.clip(exponent, -50, 50)
                    exp_val = np.exp(exponent)
                    grad_barrier = barrier_coeff * exp_val * 10 * (2 * t) / max_norm_squared
                
            return grad_data + grad_reg + grad_negative_l2 + grad_barrier

        # Initial guess
        # Use a small positive value to avoid log(0) issues at start
        if reweight_iter == 0 or t_prev is None:
            t0 = np.zeros(N) + 1e-12  # Start slightly away from zero (-80 dBm)
        else:
            # Warm start from previous iteration
            t0 = t_prev.copy()
            
        # Apply exclusion mask to initial guess
        if exclusion_mask is not None:
            t0[exclusion_mask] = 0.0

        # Box constraints: t ≥ 0
        # If exclusion_mask is provided, set bounds to (0, 0) for excluded points
        if exclusion_mask is None:
            bounds = [(0, None) for _ in range(N)]
        else:
            bounds = []
            for i in range(N):
                if exclusion_mask[i]:
                    bounds.append((0, 0))
                else:
                    bounds.append((0, None))

        # Solve with L-BFGS-B (barrier penalty handles L2 constraint)
        result = minimize(
            objective,
            t0,
            method='L-BFGS-B',
            jac=gradient,
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False},
            **scipy_kwargs
        )
        
        if verbose:
            print(f"  Solver iterations: {result.nit}")
            print(f"  Solver message: {result.message}")

        t_est = result.x
        actual_iterations = reweight_iter
        
        # Explicitly zero out excluded regions (critical for reweighting logic)
        if exclusion_mask is not None:
            t_est[exclusion_mask] = 0.0
        
        # Check convergence (skip for iteration 0)
        if enable_reweighting and reweight_iter > 0 and t_prev is not None:
            rel_change = np.linalg.norm(t_est - t_prev) / (np.linalg.norm(t_prev) + 1e-12)
            n_nonzero = np.sum(np.abs(t_est) > 1e-11)
            
            if verbose:
                print(f"  Current nonzeros: {n_nonzero}")
                print(f"  Relative change: {rel_change:.4e}")
                print(f"  Objective value: {result.fun:.4e}")
            
            if rel_change < convergence_tol:
                converged = True
                if verbose:
                    print(f"\n  * Converged after {reweight_iter} iterations")
                break
        
        # Save for next iteration
        t_prev = t_est.copy()
        
        # Heuristic: zero out very small values to help solver find exact zeros
        # This is critical for L-BFGS-B to escape small non-zero attractors
        if enable_reweighting:
            threshold = reweight_epsilon
            t_prev[t_prev < threshold] = 0.0
            if verbose:
                 print(f"  Zeroed out {np.sum(t_est < threshold)} values < {threshold:.2e}")
        
        # If not using reweighting, exit after first iteration
        if not enable_reweighting:
            break


    # Apply hard sparsity threshold if specified
    if sparsity_threshold > 0:
        n_before = np.sum(np.abs(t_est) > 1e-11)
        t_est[np.abs(t_est) < sparsity_threshold] = 0
        n_after = np.sum(np.abs(t_est) > 1e-11)
        if verbose and n_before != n_after:
            print(f"\nHard thresholding at {sparsity_threshold:.2e}:")
            print(f"  Eliminated {n_before - n_after} weak transmitters")
            print(f"  Remaining: {n_after}")
    
    # Compute statistics
    n_nonzero = np.sum(np.abs(t_est) > 1e-11) # Threshold slightly above floor
    sparsity = 1.0 - n_nonzero / N

    # Compute L2 norm of solution
    l2_norm_value = np.linalg.norm(t_est)

    # Compute breakdown of objective function components for final solution
    if verbose:
        # Recompute objective components for breakdown
        At_final = A_model @ t_est
        log_At_final = np.log10(At_final + epsilon)
        log_diff_final = log_At_final - log_p
        whitened_diff_final = W @ log_diff_final
        
        data_fidelity = np.sum(whitened_diff_final**2)
        l1_regularization = np.sum(lambda_vec * np.abs(t_est))
        
        t_squared_sum_final = np.sum(t_est**2)
        t_squared_sum_final = np.clip(t_squared_sum_final, 0, 1e100)
        negative_l2 = -gamma * t_squared_sum_final
        
        barrier_final = 0.0
        if max_l2_norm is not None:
            t_norm_squared_final = t_squared_sum_final
            max_norm_squared = max_l2_norm**2
            norm_ratio_final = t_norm_squared_final / max_norm_squared if max_norm_squared > 0 else 0
            if norm_ratio_final > 0.5:
                barrier_coeff = max(1e3, 1e6 / (1 + gamma * 1e6))
                exponent = 10 * (norm_ratio_final - 1)
                exponent = np.clip(exponent, -50, 50)
                barrier_final = barrier_coeff * (np.exp(exponent) - 1)
        
        total_objective = data_fidelity + l1_regularization + negative_l2 + barrier_final

    if verbose:
        print(f"  Status: {result.message}")
        print(f"  \n--- Objective Function Breakdown ---")
        print(f"  Data fidelity term:        {data_fidelity:.4e}")
        print(f"  L1 regularization (weighted): {l1_regularization:.4e}")
        if gamma != 0:
            print(f"  Negative L2 term (-gamma): {negative_l2:.4e}")
        if max_l2_norm is not None and barrier_final > 1e-10:
            print(f"  Barrier penalty:           {barrier_final:.4e}")
        print(f"  Total objective value:     {total_objective:.4e}")
        print(f"  (Reported by optimizer:    {result.fun:.4e})")
        print(f"  \n--- Solution Statistics ---")
        print(f"  Non-zero entries: {n_nonzero}/{N} ({n_nonzero/N*100:.1f}%)")
        print(f"  Sparsity: {sparsity*100:.1f}%")
        print(f"  Max transmit power: {t_est.max():.4e}")
        print(f"  L2 norm of solution: {l2_norm_value:.4f}")
        if max_l2_norm is not None:
            print(f"  L2 constraint: ||t||_2 ≤ {max_l2_norm:.4f} (satisfied: {l2_norm_value <= max_l2_norm + 1e-6})")

    info = {
        'solver_used': 'scipy_l-bfgs-b_log_domain',
        'objective_value': result.fun,
        'n_nonzero': n_nonzero,
        'sparsity': sparsity,
        'success': result.success,
        'n_iter': result.nit,
        'reweighting_enabled': enable_reweighting,
        'n_reweight_iter': actual_iterations if enable_reweighting else 0,
        'converged': converged if enable_reweighting else True,
        'l2_norm': l2_norm_value,
        'max_l2_norm': max_l2_norm
    }
    
    # Add objective breakdown to info if computed
    if verbose:
        info['objective_breakdown'] = {
            'data_fidelity': data_fidelity,
            'l1_regularization': l1_regularization,
            'negative_l2': negative_l2,
            'barrier': barrier_final,
            'total': total_objective
        }

    return t_est, info


def compute_reconstruction_error(A_model, t_est, observed_powers):
    """
    Compute reconstruction error: ‖A·t - p‖₂

    Parameters
    ----------
    A_model : ndarray of shape (M, N)
        Propagation matrix
    t_est : ndarray of shape (N,)
        Estimated transmit power field
    observed_powers : ndarray of shape (M,)
        Observed powers

    Returns
    -------
    float
        L2 reconstruction error

    Examples
    --------
    >>> A = np.random.rand(10, 100)
    >>> t = np.zeros(100); t[50] = 1.0  # Single transmitter
    >>> p = A @ t
    >>> error = compute_reconstruction_error(A, t, p)
    >>> error < 1e-10  # Perfect reconstruction
    True
    """
    predicted = A_model @ t_est
    error = np.linalg.norm(predicted - observed_powers)
    return error
