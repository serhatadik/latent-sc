"""
Whitening matrix computation for decorrelating sensor measurements.

The whitening matrix W = V^(-1/2) transforms the correlated sensor measurements
into uncorrelated (white) measurements, allowing the use of standard L2 norm
in the optimization objective.

Mathematical Background:
-----------------------
Given covariance matrix V ∈ ℝ^(M×M):
    V = L·L^T  (Cholesky decomposition)
    V^(-1/2) = L^(-1)

Then the whitened measurements have identity covariance:
    Cov(W·x) = W·V·W^T = W·W^T·V·W^T·W = I

This transforms the weighted least squares problem into standard least squares:
    ‖x‖²_V = x^T·V^(-1)·x = ‖V^(-1/2)·x‖²₂ = ‖W·x‖²₂
"""

import numpy as np
import warnings


def compute_whitening_matrix(cov_matrix=None, method='cholesky', regularization=1e-10,
                              verbose=True, observed_powers=None,
                              sigma_noise=1e-13, eta=0.5):
    """
    Compute whitening matrix W = V^(-1/2) from covariance matrix.

    Parameters
    ----------
    cov_matrix : ndarray of shape (M, M), optional
        Covariance matrix (must be symmetric positive definite).
        Required for 'cholesky', 'svd', and 'eig' methods.
    method : {'cholesky', 'svd', 'eig', 'log_inv_power_diag', 'hetero_diag'}, optional
        Method for computing matrix square root, default: 'cholesky'
        - 'cholesky': Fast, requires positive definite matrix
        - 'svd': More robust, handles near-singular matrices
        - 'eig': Eigenvalue decomposition, middle ground
        - 'log_inv_power_diag': Diagonal matrix based on observed powers (W_jj = -log10(p_j))
        - 'hetero_diag': Heteroscedastic diagonal matrix V_kk = sigma_noise^2 + eta^2 * p_k^2
    regularization : float, optional
        Small value added to diagonal for numerical stability, default: 1e-10
    verbose : bool, optional
        Print diagnostic information, default: True
    observed_powers : ndarray of shape (M,), optional
        Observed sensor powers (linear scale, mW). Required for method='log_inv_power_diag' and 'hetero_diag'.
        When provided for 'log_inv_power_diag', diagonal whitening matrix W_jj = log10(1 / observed_powers_j) is computed.
    sigma_noise : float, optional
        Noise floor variance for 'hetero_diag' method. Default 1e-13.
    eta : float, optional
        Scaling factor for signal-dependent variance in 'hetero_diag'. Default 0.5.

    Returns
    -------
    W : ndarray of shape (M, M)
        Whitening matrix such that W·V·W^T ≈ I

    Raises
    ------
    np.linalg.LinAlgError
        If Cholesky decomposition fails (matrix not positive definite)
    ValueError
        If required inputs are missing for the selected method.

    Examples
    --------
    >>> V = np.array([[4, 1], [1, 2]])
    >>> W = compute_whitening_matrix(V, method='cholesky')
    >>> np.allclose(W @ V @ W.T, np.eye(2), atol=1e-6)
    True

    >>> # Near-singular matrix
    >>> V_singular = np.array([[1, 0.999], [0.999, 1]])
    >>> W = compute_whitening_matrix(V_singular, method='svd')
    >>> W.shape
    (2, 2)

    Notes
    -----
    - Cholesky is fastest but requires positive definite matrix
    - SVD is most robust for near-singular or ill-conditioned matrices
    - Small regularization (1e-10) helps with numerical stability
    """
    # Determine dimension M
    if cov_matrix is not None:
        M = cov_matrix.shape[0]
    elif observed_powers is not None:
        M = observed_powers.shape[0]
    else:
        raise ValueError("Either cov_matrix or observed_powers must be provided to determine dimension M")

    # Validate cov_matrix if provided
    if cov_matrix is not None:
        if cov_matrix.shape != (M, M):
            raise ValueError(f"Covariance matrix must be square, got shape {cov_matrix.shape}")

        if not np.allclose(cov_matrix, cov_matrix.T):
            warnings.warn("Covariance matrix is not symmetric, symmetrizing...")
            cov_matrix = (cov_matrix + cov_matrix.T) / 2

        # Add small regularization for numerical stability
        if regularization > 0:
            cov_matrix = cov_matrix + regularization * np.eye(M)

        if verbose:
            print(f"Computing whitening matrix using {method} method...")
            cond_num = np.linalg.cond(cov_matrix)
            print(f"  Covariance matrix: {M}×{M}")
            print(f"  Condition number: {cond_num:.2e}")
            if cond_num > 1e10:
                warnings.warn(
                    f"Covariance matrix is ill-conditioned (κ={cond_num:.2e}). "
                    "Consider using method='svd' for better stability."
                )
    elif verbose:
        print(f"Computing whitening matrix using {method} method...")

    # Compute W = V^(-1/2) using specified method
    if method == 'cholesky':
        if cov_matrix is None:
            raise ValueError("cov_matrix is required for method='cholesky'")
        W = _whitening_cholesky(cov_matrix, verbose)
    elif method == 'svd':
        if cov_matrix is None:
            raise ValueError("cov_matrix is required for method='svd'")
        W = _whitening_svd(cov_matrix, verbose)
    elif method == 'eig':
        if cov_matrix is None:
            raise ValueError("cov_matrix is required for method='eig'")
        W = _whitening_eig(cov_matrix, verbose)
    elif method == 'log_inv_power_diag':
        if observed_powers is None:
            raise ValueError("observed_powers must be provided for method='log_inv_power_diag'")
        W = _whitening_diagonal_observation(observed_powers, M, verbose)
    elif method == 'hetero_diag':
        if observed_powers is None:
            raise ValueError("observed_powers must be provided for method='hetero_diag'")
        W = _whitening_hetero_diag(observed_powers, M, sigma_noise, eta, verbose)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'cholesky', 'svd', 'eig', 'log_inv_power_diag', or 'hetero_diag'")

    # Validate result (only if cov_matrix was used/provided)
    if verbose and cov_matrix is not None and method not in ['log_inv_power_diag', 'hetero_diag']:
        whitened_cov = W @ cov_matrix @ W.T
        error = np.linalg.norm(whitened_cov - np.eye(M), 'fro') / M
        print(f"  Whitening error: ‖W·V·W^T - I‖_F / M = {error:.2e}")
        if error > 1e-6:
            warnings.warn(
                f"Whitening error is large ({error:.2e}). "
                "Check covariance matrix conditioning."
            )

    return W


def _whitening_hetero_diag(observed_powers, M, sigma_noise=1e-13, eta=0.5, verbose=True):
    """
    Compute heteroscedastic diagonal whitening matrix.

    [V_diag]_kk = sigma_noise^2 + eta^2 * (p_k)^2
    W = V^(-1/2) = diag(1 / sqrt(V_diag))

    Parameters
    ----------
    observed_powers : ndarray
        Observed powers in linear scale (mW)
    M : int
        Number of sensors
    sigma_noise : float
        Noise floor variance
    eta : float
        Scaling factor for signal-dependent variance
    verbose : bool
        Print info

    Returns
    -------
    W : ndarray
        Whitening matrix
    """
    # Diagonal elements of V
    v_diag = sigma_noise**2 + (eta * observed_powers)**2
    
    # W = V^(-1/2) is diagonal with elements 1/sqrt(v_diag)
    w_diag = 1.0 / np.sqrt(v_diag)
    W = np.diag(w_diag)
    
    if verbose:
        print(f"  Heteroscedastic diagonal whitening:")
        print(f"    sigma_noise: {sigma_noise:.2e}, eta: {eta:.2f}")
        print(f"    V_diag range: [{v_diag.min():.2e}, {v_diag.max():.2e}]")
        print(f"    W_diag range: [{w_diag.min():.2e}, {w_diag.max():.2e}]")
    
    return W


def _whitening_cholesky(cov_matrix, verbose=True):
    """
    Compute whitening matrix using Cholesky decomposition.

    V = L·L^T  =>  W = L^(-1)

    Fast but requires positive definite matrix.
    """
    try:
        L = np.linalg.cholesky(cov_matrix)
        W = np.linalg.inv(L)

        if verbose:
            print(f"  Cholesky decomposition successful")

        return W

    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "Cholesky decomposition failed. Matrix is not positive definite. "
            "Try method='svd' for more robust computation."
        ) from e


def _whitening_svd(cov_matrix, verbose=True):
    """
    Compute whitening matrix using SVD.

    V = U·S·V^T  =>  W = U·S^(-1/2)·U^T

    More robust for near-singular matrices.
    """
    U, s, Vt = np.linalg.svd(cov_matrix)

    # Threshold small singular values to avoid division by zero
    threshold = 1e-10 * s[0]  # Relative to largest singular value
    s_inv_sqrt = np.where(s > threshold, 1.0 / np.sqrt(s), 0.0)

    n_small = np.sum(s <= threshold)
    if verbose and n_small > 0:
        warnings.warn(
            f"Found {n_small} small singular values (< {threshold:.2e}), "
            "setting corresponding entries to zero."
        )

    W = U @ np.diag(s_inv_sqrt) @ U.T

    if verbose:
        print(f"  SVD decomposition: {len(s)} singular values")
        print(f"  Condition number: {s[0] / s[-1]:.2e}")

    return W


def _whitening_eig(cov_matrix, verbose=True):
    """
    Compute whitening matrix using eigenvalue decomposition.

    V = Q·Λ·Q^T  =>  W = Q·Λ^(-1/2)·Q^T

    Middle ground between speed and robustness.
    """
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Threshold small eigenvalues
    threshold = 1e-10 * eigvals[-1]  # Relative to largest eigenvalue
    eigvals_inv_sqrt = np.where(eigvals > threshold, 1.0 / np.sqrt(eigvals), 0.0)

    n_small = np.sum(eigvals <= threshold)
    if verbose and n_small > 0:
        warnings.warn(
            f"Found {n_small} small eigenvalues (< {threshold:.2e}), "
            "setting corresponding entries to zero."
        )

    W = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T

    if verbose:
        print(f"  Eigenvalue decomposition: {len(eigvals)} eigenvalues")
        print(f"  Condition number: {eigvals[-1] / eigvals[0]:.2e}")

    return W


def _whitening_diagonal_observation(observed_powers, M, verbose=True):
    """
    Compute diagonal whitening matrix based on observed sensor powers.

    W_jj = log10(1 / p_j) = -log10(p_j)

    This weights sensors inversely proportional to their observed power in log domain.
    Sensors with weaker signals (smaller p_j) get larger weights.

    Parameters
    ----------
    observed_powers : ndarray of shape (M,)
        Observed sensor powers in linear scale (mW)
    M : int
        Number of sensors
    verbose : bool, optional
        Print diagnostic information, default: True

    Returns
    -------
    W : ndarray of shape (M, M)
        Diagonal whitening matrix
    """
    if observed_powers.shape != (M,):
        raise ValueError(f"observed_powers shape {observed_powers.shape} must be ({M},)")
    
    if np.any(observed_powers <= 0):
        raise ValueError("All observed powers must be positive for diagonal_observation method")
    
    # Compute diagonal elements: W_jj = log10(1/p_j) = -log10(p_j)
    diagonal_elements = np.log10(1.0 / observed_powers)
    
    # Create diagonal matrix
    W = np.diag(diagonal_elements)
    
    if verbose:
        print(f"  Diagonal observation whitening:")
        print(f"    Observed power range: [{observed_powers.min():.2e}, {observed_powers.max():.2e}] mW")
        print(f"    Diagonal range: [{diagonal_elements.min():.2f}, {diagonal_elements.max():.2f}]")
        print(f"    Mean diagonal: {diagonal_elements.mean():.2f}")
    
    return W


def apply_whitening(W, measurements):
    """
    Apply whitening transformation to measurements.

    Parameters
    ----------
    W : ndarray of shape (M, M)
        Whitening matrix
    measurements : ndarray of shape (M,) or (M, K)
        Measurements to whiten (can be vector or matrix)

    Returns
    -------
    ndarray of shape (M,) or (M, K)
        Whitened measurements

    Examples
    --------
    >>> W = np.eye(3)  # Identity whitening
    >>> x = np.array([1, 2, 3])
    >>> apply_whitening(W, x)
    array([1, 2, 3])

    >>> # Multiple measurements
    >>> X = np.random.rand(3, 10)
    >>> W = np.eye(3)
    >>> apply_whitening(W, X).shape
    (3, 10)
    """
    return W @ measurements


def validate_whitening(W, cov_matrix, tolerance=1e-6):
    """
    Validate that W is a valid whitening matrix for V.

    Checks: W·V·W^T ≈ I

    Parameters
    ----------
    W : ndarray of shape (M, M)
        Whitening matrix
    cov_matrix : ndarray of shape (M, M)
        Original covariance matrix
    tolerance : float, optional
        Tolerance for identity check, default: 1e-6

    Returns
    -------
    bool
        True if W is a valid whitening matrix

    Examples
    --------
    >>> V = np.diag([4, 2, 1])
    >>> W = np.diag([0.5, 1/np.sqrt(2), 1])
    >>> validate_whitening(W, V, tolerance=1e-6)
    True
    """
    M = cov_matrix.shape[0]
    whitened_cov = W @ cov_matrix @ W.T
    identity = np.eye(M)

    error = np.linalg.norm(whitened_cov - identity, 'fro') / M

    is_valid = error < tolerance

    print(f"Whitening validation:")
    print(f"  ‖W·V·W^T - I‖_F / M = {error:.2e}")
    print(f"  Valid (< {tolerance}): {is_valid}")

    return is_valid
