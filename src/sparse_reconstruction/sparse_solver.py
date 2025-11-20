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
                                 solver='scipy', verbose=True, **solver_kwargs):
    """
    Solve sparse reconstruction problem with automatic solver selection.

    Solves:
        min  ‖W(log10(A·t) - log10(p))‖₂² + λ‖t‖₁
        t≥0

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
    **solver_kwargs : dict
        Additional arguments passed to specific solver

    Returns
    -------
    t_est : ndarray of shape (N,)
        Estimated transmit power field (linear scale)
        Sparse: most entries are zero or near-zero
    info : dict
        Solver information including:
        - 'solver_used': which solver was used
        - 'objective_value': final objective value
        - 'n_nonzero': number of non-zero entries in solution
        - 'sparsity': fraction of zero entries
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
        print(f"Sparse Reconstruction Solver (Log-Domain)")
        print(f"{'='*60}")
        print(f"Problem size: M={A_model.shape[0]} sensors, N={A_model.shape[1]} grid points")
        if np.isscalar(lambda_reg):
            print(f"Sparsity parameter: λ = {lambda_reg:.2e}")
        else:
            print(f"Sparsity parameter: λ = [array of shape {lambda_reg.shape}]")
            print(f"  Mean λ: {np.mean(lambda_reg):.2e}")
            print(f"  Max λ: {np.max(lambda_reg):.2e}")

    # Validate inputs
    M, N = A_model.shape
    if W.shape != (M, M):
        raise ValueError(f"Whitening matrix shape {W.shape} incompatible with A_model shape {A_model.shape}")
    if observed_powers.shape != (M,):
        raise ValueError(f"Observed powers shape {observed_powers.shape} must be ({M},)")
    if np.any(lambda_reg < 0):
        raise ValueError(f"Regularization parameter must be non-negative")

    if solver != 'scipy':
        if verbose:
            print(f"Warning: Solver '{solver}' is not supported for log-domain optimization. Falling back to 'scipy'.")
        solver = 'scipy'

    # Use scipy (L-BFGS-B)
    t_est, info = solve_sparse_reconstruction_scipy(
        A_model, W, observed_powers, lambda_reg, verbose=verbose, **solver_kwargs
    )
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
                                       verbose=True, max_iter=1000, epsilon=1e-20, **scipy_kwargs):
    """
    Solve sparse reconstruction using scipy.optimize (L-BFGS-B) with log-domain objective.

    Objective:
        min ‖W(log10(A·t + ε) - log10(p + ε))‖₂² + λ‖t‖₁
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
        Maximum iterations, default: 1000
    epsilon : float, optional
        Small constant for numerical stability in log, default: 1e-20
    **scipy_kwargs : dict
        Additional arguments passed to scipy.optimize.minimize

    Returns
    -------
    t_est : ndarray of shape (N,)
        Estimated transmit power field
    info : dict
        Solver information
    """
    from scipy.optimize import minimize

    M, N = A_model.shape

    if verbose:
        print("Using scipy L-BFGS-B solver (Log-Domain)...")

    # Precompute constant terms
    # log10(p + epsilon)
    log_p = np.log10(observed_powers + epsilon)
    
    # Whitened log observed powers: W @ log_p
    # Note: W is applied to the difference of logs, so we can't pre-multiply W with log_p alone 
    # if we want to keep the structure W(log(At) - log(p)). 
    # Actually we can: W(a - b) = Wa - Wb. So yes, we can precompute W @ log_p.
    p_tilde = W @ log_p

    # Precompute W^T W for gradient calculation efficiency if needed, 
    # but W is small (MxM), so W.T @ (W @ r) is fast enough.

    # Precompute weights for L1 penalty (cancel path loss bias)
    # Omega_ii = ||a_i||_2
    column_norms = np.linalg.norm(A_model, axis=0)
    if column_norms.max() > 0:
        weights = column_norms / column_norms.max()
    else:
        weights = np.ones(N)

    # Prepare regularization vector
    if np.isscalar(lambda_reg):
        # Apply computed weights to scalar lambda
        lambda_vec = lambda_reg * weights
    else:
        # Use provided vector as is (assume user handled weighting)
        lambda_vec = lambda_reg

    def objective(t):
        """
        Objective function: ‖W(log10(A·t + ε) - log10(p + ε))‖₂² + ‖diag(λ)·t‖₁
        """
        # 1. Compute A·t
        At = A_model @ t
        
        # 2. Compute log10(A·t + ε)
        log_At = np.log10(At + epsilon)
        
        # 3. Compute residual vector in log domain: log(At) - log(p)
        # We already have log_p, so this is log_At - log_p
        log_diff = log_At - log_p
        
        # 4. Apply whitening: W(log_diff)
        # Equivalent to W @ log_At - p_tilde
        whitened_diff = W @ log_diff
        
        # 5. Data fidelity term: squared L2 norm
        data_term = np.sum(whitened_diff**2)
        
        # 6. Regularization term: Weighted L1 norm
        # lambda_vec already includes the weights
        regularization_term = np.sum(lambda_vec * np.abs(t))
            
        return data_term + regularization_term

    def gradient(t):
        """
        Gradient of objective.
        """
        # Forward pass (recompute needed parts)
        At = A_model @ t
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
        
        # Regularization gradient
        # Gradient of sum(lambda_vec * |t|) is lambda_vec * sign(t)
        grad_reg = lambda_vec * np.sign(t)
            
        return grad_data + grad_reg

    # Initial guess
    # Use a small positive value to avoid log(0) issues at start
    t0 = np.zeros(N) + 1e-8 # Start slightly away from zero (-80 dBm)

    # Box constraints: t ≥ 0
    bounds = [(0, None) for _ in range(N)]

    # Solve
    result = minimize(
        objective,
        t0,
        method='L-BFGS-B',
        jac=gradient,
        bounds=bounds,
        options={'maxiter': max_iter, 'disp': verbose},
        **scipy_kwargs
    )

    t_est = result.x

    # Compute statistics
    n_nonzero = np.sum(np.abs(t_est) > 1e-15) # Threshold slightly above floor
    sparsity = 1.0 - n_nonzero / N

    if verbose:
        print(f"  Status: {result.message}")
        print(f"  Objective value: {result.fun:.4e}")
        print(f"  Non-zero entries: {n_nonzero}/{N} ({n_nonzero/N*100:.1f}%)")
        print(f"  Sparsity: {sparsity*100:.1f}%")
        print(f"  Max transmit power: {t_est.max():.4e}")

    info = {
        'solver_used': 'scipy_log_domain',
        'objective_value': result.fun,
        'n_nonzero': n_nonzero,
        'sparsity': sparsity,
        'success': result.success,
        'n_iter': result.nit
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
