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
                                 solver='auto', verbose=True, **solver_kwargs):
    """
    Solve sparse reconstruction problem with automatic solver selection.

    Solves:
        min  ‖W(A·t - p)‖₂² + λ‖t‖₁
        t≥0

    Parameters
    ----------
    A_model : ndarray of shape (M, N)
        Propagation matrix with linear path gains
    W : ndarray of shape (M, M)
        Whitening matrix (W = V^(-1/2))
    observed_powers : ndarray of shape (M,)
        Observed powers at sensors (linear scale, e.g., mW)
    lambda_reg : float
        Sparsity regularization parameter (λ ≥ 0)
        - λ = 0: standard least squares (dense solution)
        - λ > 0: encourages sparsity (fewer active transmitters)
        - λ → ∞: forces all-zero solution
    solver : {'auto', 'cvxpy', 'sklearn', 'scipy'}, optional
        Solver to use, default: 'auto' (tries cvxpy, sklearn, then scipy)
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

    Notes
    -----
    Solver Selection:
    - 'cvxpy': Recommended for accuracy and flexibility (requires cvxpy package)
    - 'sklearn': Fast for large problems (requires scikit-learn)
    - 'scipy': Always available but slower, uses L-BFGS-B
    - 'auto': Tries cvxpy -> sklearn -> scipy in order

    Regularization Parameter Selection:
    - Start with λ = 0.01 * ‖W·p‖₂ as initial guess
    - Increase λ for sparser solutions (fewer transmitters)
    - Decrease λ for denser solutions (more transmitters)
    - Use cross-validation or information criteria for optimal λ
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Sparse Reconstruction Solver")
        print(f"{'='*60}")
        print(f"Problem size: M={A_model.shape[0]} sensors, N={A_model.shape[1]} grid points")
        print(f"Sparsity parameter: λ = {lambda_reg:.2e}")

    # Validate inputs
    M, N = A_model.shape
    if W.shape != (M, M):
        raise ValueError(f"Whitening matrix shape {W.shape} incompatible with A_model shape {A_model.shape}")
    if observed_powers.shape != (M,):
        raise ValueError(f"Observed powers shape {observed_powers.shape} must be ({M},)")
    if lambda_reg < 0:
        raise ValueError(f"Regularization parameter must be non-negative, got {lambda_reg}")

    # Try solvers in order
    if solver == 'auto':
        # Try cvxpy first (most accurate)
        try:
            t_est, info = solve_sparse_reconstruction_cvxpy(
                A_model, W, observed_powers, lambda_reg, verbose=verbose, **solver_kwargs
            )
            return t_est, info
        except ImportError:
            if verbose:
                print("  cvxpy not available, trying sklearn...")

        # Try sklearn second (fast)
        try:
            t_est, info = solve_sparse_reconstruction_sklearn(
                A_model, W, observed_powers, lambda_reg, verbose=verbose, **solver_kwargs
            )
            return t_est, info
        except ImportError:
            if verbose:
                print("  sklearn not available, falling back to scipy...")

        # Fall back to scipy (always available)
        t_est, info = solve_sparse_reconstruction_scipy(
            A_model, W, observed_powers, lambda_reg, verbose=verbose, **solver_kwargs
        )
        return t_est, info

    elif solver == 'cvxpy':
        return solve_sparse_reconstruction_cvxpy(
            A_model, W, observed_powers, lambda_reg, verbose=verbose, **solver_kwargs
        )
    elif solver == 'sklearn':
        return solve_sparse_reconstruction_sklearn(
            A_model, W, observed_powers, lambda_reg, verbose=verbose, **solver_kwargs
        )
    elif solver == 'scipy':
        return solve_sparse_reconstruction_scipy(
            A_model, W, observed_powers, lambda_reg, verbose=verbose, **solver_kwargs
        )
    else:
        raise ValueError(f"Unknown solver: {solver}. Choose 'auto', 'cvxpy', 'sklearn', or 'scipy'")


def solve_sparse_reconstruction_cvxpy(A_model, W, observed_powers, lambda_reg,
                                       verbose=True, solver_name=None, **solver_options):
    """
    Solve sparse reconstruction using CVXPY (convex optimization).

    Most accurate and flexible solver. Handles constraints elegantly.

    Parameters
    ----------
    A_model : ndarray of shape (M, N)
        Propagation matrix
    W : ndarray of shape (M, M)
        Whitening matrix
    observed_powers : ndarray of shape (M,)
        Observed powers (linear scale)
    lambda_reg : float
        Regularization parameter
    verbose : bool, optional
        Print solver information, default: True
    solver_name : str, optional
        CVXPY solver to use (e.g., 'ECOS', 'SCS', 'OSQP'), default: None (auto)
    **solver_options : dict
        Additional options passed to CVXPY solver

    Returns
    -------
    t_est : ndarray of shape (N,)
        Estimated transmit power field
    info : dict
        Solver information

    Raises
    ------
    ImportError
        If cvxpy is not installed
    """
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "CVXPY is not installed. Install with: pip install cvxpy"
        ) from e

    M, N = A_model.shape

    if verbose:
        print("Using CVXPY solver...")

    # Precompute whitened matrices
    A_tilde = W @ A_model  # Shape: (M, N)
    p_tilde = W @ observed_powers  # Shape: (M,)

    # Decision variable: transmit power field
    t = cp.Variable(N, nonneg=True)  # Non-negativity constraint

    # Objective: ‖A_tilde·t - p_tilde‖₂² + λ‖t‖₁
    residual = A_tilde @ t - p_tilde
    objective = cp.Minimize(
        cp.sum_squares(residual) + lambda_reg * cp.norm(t, 1)
    )

    # Solve
    problem = cp.Problem(objective)

    try:
        if solver_name is not None:
            problem.solve(solver=solver_name, verbose=verbose, **solver_options)
        else:
            problem.solve(verbose=verbose, **solver_options)
    except cp.error.SolverError as e:
        warnings.warn(f"CVXPY solver failed: {e}. Trying default solver...")
        problem.solve(verbose=verbose)

    # Extract solution
    if t.value is None:
        raise RuntimeError("CVXPY optimization failed to find a solution")

    t_est = t.value

    # Compute statistics
    n_nonzero = np.sum(np.abs(t_est) > 1e-10)
    sparsity = 1.0 - n_nonzero / N

    if verbose:
        print(f"  Status: {problem.status}")
        print(f"  Objective value: {problem.value:.4e}")
        print(f"  Non-zero entries: {n_nonzero}/{N} ({n_nonzero/N*100:.1f}%)")
        print(f"  Sparsity: {sparsity*100:.1f}%")
        print(f"  Max transmit power: {t_est.max():.4e}")

    info = {
        'solver_used': 'cvxpy',
        'objective_value': problem.value,
        'n_nonzero': n_nonzero,
        'sparsity': sparsity,
        'success': problem.status == 'optimal',
        'status': problem.status
    }

    return t_est, info


def solve_sparse_reconstruction_sklearn(A_model, W, observed_powers, lambda_reg,
                                         verbose=True, max_iter=10000, tol=1e-6,
                                         **lasso_kwargs):
    """
    Solve sparse reconstruction using scikit-learn's Lasso.

    Fast coordinate descent solver. Good for large problems.

    Parameters
    ----------
    A_model : ndarray of shape (M, N)
        Propagation matrix
    W : ndarray of shape (M, M)
        Whitening matrix
    observed_powers : ndarray of shape (M,)
        Observed powers (linear scale)
    lambda_reg : float
        Regularization parameter
    verbose : bool, optional
        Print solver information, default: True
    max_iter : int, optional
        Maximum iterations, default: 10000
    tol : float, optional
        Convergence tolerance, default: 1e-6
    **lasso_kwargs : dict
        Additional arguments passed to sklearn.linear_model.Lasso

    Returns
    -------
    t_est : ndarray of shape (N,)
        Estimated transmit power field
    info : dict
        Solver information

    Raises
    ------
    ImportError
        If scikit-learn is not installed

    Notes
    -----
    sklearn's Lasso minimizes:
        (1/2M) ‖X·w - y‖₂² + α‖w‖₁

    Our problem:
        ‖A_tilde·t - p_tilde‖₂² + λ‖t‖₁

    To match, we set: α = λ / (2M)
    """
    try:
        from sklearn.linear_model import Lasso
    except ImportError as e:
        raise ImportError(
            "scikit-learn is not installed. Install with: pip install scikit-learn"
        ) from e

    M, N = A_model.shape

    if verbose:
        print("Using scikit-learn Lasso solver...")

    # Precompute whitened matrices
    A_tilde = W @ A_model
    p_tilde = W @ observed_powers

    # Convert regularization parameter to sklearn format
    # sklearn: (1/2M) ‖X·w - y‖₂² + α‖w‖₁
    # Our problem: ‖A·t - p‖₂² + λ‖t‖₁
    # Therefore: α = λ / (2M)
    alpha_sklearn = lambda_reg / (2 * M)

    # Solve using Lasso with positive constraint
    lasso = Lasso(
        alpha=alpha_sklearn,
        positive=True,  # Non-negativity constraint
        max_iter=max_iter,
        tol=tol,
        fit_intercept=False,  # No intercept (physical constraint)
        **lasso_kwargs
    )

    lasso.fit(A_tilde, p_tilde)

    t_est = lasso.coef_

    # Compute statistics
    n_nonzero = np.sum(np.abs(t_est) > 1e-10)
    sparsity = 1.0 - n_nonzero / N

    # Compute objective value
    residual = A_tilde @ t_est - p_tilde
    objective = np.sum(residual**2) + lambda_reg * np.sum(np.abs(t_est))

    if verbose:
        print(f"  Iterations: {lasso.n_iter_}")
        print(f"  Objective value: {objective:.4e}")
        print(f"  Non-zero entries: {n_nonzero}/{N} ({n_nonzero/N*100:.1f}%)")
        print(f"  Sparsity: {sparsity*100:.1f}%")
        print(f"  Max transmit power: {t_est.max():.4e}")

    info = {
        'solver_used': 'sklearn',
        'objective_value': objective,
        'n_nonzero': n_nonzero,
        'sparsity': sparsity,
        'success': lasso.n_iter_ < max_iter,
        'n_iter': lasso.n_iter_
    }

    return t_est, info


def solve_sparse_reconstruction_scipy(A_model, W, observed_powers, lambda_reg,
                                       verbose=True, max_iter=1000, **scipy_kwargs):
    """
    Solve sparse reconstruction using scipy.optimize (L-BFGS-B).

    General-purpose optimizer. Always available but slower than specialized solvers.

    Parameters
    ----------
    A_model : ndarray of shape (M, N)
        Propagation matrix
    W : ndarray of shape (M, M)
        Whitening matrix
    observed_powers : ndarray of shape (M,)
        Observed powers (linear scale)
    lambda_reg : float
        Regularization parameter
    verbose : bool, optional
        Print solver information, default: True
    max_iter : int, optional
        Maximum iterations, default: 1000
    **scipy_kwargs : dict
        Additional arguments passed to scipy.optimize.minimize

    Returns
    -------
    t_est : ndarray of shape (N,)
        Estimated transmit power field
    info : dict
        Solver information

    Notes
    -----
    Uses L-BFGS-B with box constraints (t ≥ 0).
    Approximates L1 penalty with smooth approximation for gradient-based optimization.
    """
    from scipy.optimize import minimize

    M, N = A_model.shape

    if verbose:
        print("Using scipy L-BFGS-B solver...")

    # Precompute whitened matrices
    A_tilde = W @ A_model
    p_tilde = W @ observed_powers

    def objective(t):
        """Objective function: ‖A·t - p‖₂² + λ‖t‖₁"""
        residual = A_tilde @ t - p_tilde
        data_term = np.sum(residual**2)
        regularization_term = lambda_reg * np.sum(np.abs(t))
        return data_term + regularization_term

    def gradient(t):
        """Gradient of objective (using subgradient for L1)"""
        residual = A_tilde @ t - p_tilde
        grad_data = 2 * A_tilde.T @ residual
        grad_reg = lambda_reg * np.sign(t)  # Subgradient of L1
        return grad_data + grad_reg

    # Initial guess: zero
    t0 = np.zeros(N)

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
    n_nonzero = np.sum(np.abs(t_est) > 1e-10)
    sparsity = 1.0 - n_nonzero / N

    if verbose:
        print(f"  Status: {result.message}")
        print(f"  Objective value: {result.fun:.4e}")
        print(f"  Non-zero entries: {n_nonzero}/{N} ({n_nonzero/N*100:.1f}%)")
        print(f"  Sparsity: {sparsity*100:.1f}%")
        print(f"  Max transmit power: {t_est.max():.4e}")

    info = {
        'solver_used': 'scipy',
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
