"""
Whitened Bias-Invariant Iterative GLRT Solver.

This module implements the iterative Generalized Likelihood Ratio Test (GLRT)
for detecting and localizing transmitters in the presence of colored noise
and unknown additive bias.
"""

import numpy as np
from scipy import linalg
from scipy.optimize import lsq_linear

def solve_glrt(observed_powers, A_model, threshold, cov_matrix=None,
               max_iter=100, verbose=True, sigma_noise=1e-13, eta=0.5):
    """
    Whitened Bias-Invariant Iterative GLRT.

    Parameters
    ----------
    observed_powers : ndarray of shape (M,)
        Observed powers at sensors (p).
    A_model : ndarray of shape (M, N)
        Propagation matrix (A).
    threshold : float
        Detection threshold (gamma).
    cov_matrix : ndarray of shape (M, M) or str, optional
        Covariance matrix (V). If None, assumes Identity matrix.
        If 'hetero_diag', uses heteroscedastic diagonal covariance.
    max_iter : int, optional
        Maximum number of iterations (transmitters to detect).
    verbose : bool, optional
        Print progress.
    sigma_noise : float, optional
        Noise floor variance for 'hetero_diag' covariance. Default 1e-13.
    eta : float, optional
        Scaling factor for signal-dependent variance in 'hetero_diag'. Default 0.5.

    Returns
    -------
    active_set : list
        List of indices of detected transmitters.
    estimated_powers : ndarray of shape (len(active_set),)
        Estimated powers for the detected transmitters.
    estimated_bias : float
        Estimated constant bias.
    info : dict
        Additional information (residuals, scores, etc.).
    """
    M, N = A_model.shape
    
    # 1. Initialization
    r = observed_powers.copy()
    active_set = []
    
    # Whitening Matrix W = V^(-1/2)
    if cov_matrix is None:
        W = np.eye(M)
    elif isinstance(cov_matrix, str) and cov_matrix == 'hetero_diag':
        # Heteroscedastic Diagonal Covariance
        # [V_diag]_kk = sigma_noise^2 + eta^2 * (p_k)^2
        # sigma_noise and eta are now parameters
        
        # Diagonal elements of V
        v_diag = sigma_noise**2 + (eta * observed_powers)**2
        
        # W = V^(-1/2) is diagonal with elements 1/sqrt(v_diag)
        # Avoid division by zero if any v_diag is 0 (unlikely with sigma_noise)
        w_diag = 1.0 / np.sqrt(v_diag)
        W = np.diag(w_diag)
        
        if verbose:
            print(f"Using Heteroscedastic Diagonal Covariance (hetero_diag) with sigma_noise={sigma_noise}, eta={eta}.")
    else:
        # Using Cholesky: V = L L^T => W = L^(-1)
        try:
            L = linalg.cholesky(cov_matrix, lower=True)
            W = linalg.inv(L)
        except linalg.LinAlgError:
            # Fallback for singular covariance
            if verbose:
                print("Warning: Covariance matrix singular, using pseudo-inverse for whitening.")
            evals, evecs = linalg.eigh(cov_matrix)
            # Filter small eigenvalues
            mask = evals > 1e-10
            W = evecs[:, mask] @ np.diag(1.0 / np.sqrt(evals[mask])) @ evecs[:, mask].T
            # Note: This W might not be triangular, but W @ V @ W.T ~ I

    # Whitened Bias 1_w = W @ 1
    ones_vec = np.ones(M)
    ones_w = W @ ones_vec
    
    # Projection Matrix P_perp = I - (1_w 1_w^T) / ||1_w||^2
    norm_ones_w_sq = np.dot(ones_w, ones_w)
    if norm_ones_w_sq < 1e-10:
        # Should not happen unless W or 1 is zero
        P_perp = np.eye(M)
    else:
        P_perp = np.eye(M) - np.outer(ones_w, ones_w) / norm_ones_w_sq

    # Pre-compute Whitened & Projected Template Matrix: A_tilde = P_perp @ W @ A
    # This avoids recomputing it inside the loop for the scan step
    # Shape: (M, N)
    WA = W @ A_model
    A_tilde = P_perp @ WA
    
    # Pre-compute norms of columns of A_tilde for the denominator of the score
    # Shape: (N,)
    A_tilde_norms_sq = np.sum(A_tilde**2, axis=0)
    
    # Avoid division by zero for columns that are effectively zero (e.g. far away or masked)
    valid_cols_mask = A_tilde_norms_sq > 1e-20
    
    iteration = 0
    scores_history = []
    candidates_history = []
    
    while iteration < max_iter:
        # 2. Whiten & Project Residual
        # r_tilde = P_perp @ W @ r
        Wr = W @ r
        r_tilde = P_perp @ Wr
        
        # 3. Estimate Noise Variance
        # sigma_hat^2 = (1/M) * ||r_tilde||^2
        sigma_hat_sq = np.mean(r_tilde**2)
        
        if sigma_hat_sq < 1e-20:
            if verbose:
                print("Residual variance effectively zero. Stopping.")
            break

        # 4. Scan: Compute GLRT scores
        # l_i = (a_tilde_i^T r_tilde)^2 / (||a_tilde_i||^2 * sigma_hat^2)
        
        # Vectorized computation:
        # Numerator: (A_tilde^T @ r_tilde)^2
        correlations = A_tilde.T @ r_tilde
        numerator = correlations**2
        
        # Denominator: A_tilde_norms_sq * sigma_hat_sq
        denominator = A_tilde_norms_sq * sigma_hat_sq
        
        # Compute scores (handle invalid columns)
        scores = np.zeros(N)
        scores[valid_cols_mask] = numerator[valid_cols_mask] / denominator[valid_cols_mask]
        
        # Mask out already active indices
        if active_set:
            scores[active_set] = -1.0
            
        # 5. Select Best Candidate
        i_star = np.argmax(scores)
        l_max = scores[i_star]
        
        # Store top 10 candidates for visualization
        # Get indices of top 10 scores
        if len(scores) >= 10:
            top_10_indices = np.argpartition(scores, -10)[-10:]
            # Sort them by score descending
            top_10_indices = top_10_indices[np.argsort(scores[top_10_indices])[::-1]]
        else:
            top_10_indices = np.argsort(scores)[::-1]
            
        top_10_scores = scores[top_10_indices]
        
        candidates_history.append({
            'iteration': iteration,
            'top_indices': top_10_indices,
            'top_scores': top_10_scores,
            'selected_index': i_star,
            'selected_score': l_max
        })
        
        if verbose:
            print(f"Iter {iteration}: Max Score = {l_max:.4f} at index {i_star}")
            
        scores_history.append(l_max)

        # 6. Test Threshold
        if l_max > threshold:
            active_set.append(i_star)
            
            # 7. Joint Estimation (Update)
            # Estimate t_S and c to minimize || W(p - A_S t_S - c 1) ||^2
            # This is equivalent to minimizing || W [A_S, 1] [t_S; c] - W p ||^2
            
            # Construct design matrix X = [A_S, 1]
            # We use the original A_model columns corresponding to active_set
            A_S = A_model[:, active_set]
            X = np.column_stack([A_S, ones_vec])
            
            # Whitened design matrix and target
            WX = W @ X
            Wp = W @ observed_powers
            
            # Solve constrained least squares: t_S >= 0, c unconstrained
            # Variables: [t_S_1, ..., t_S_k, c]
            # Bounds: 
            #   t_S: [0, inf)
            #   c:   (-inf, inf)
            
            k = len(active_set)
            lb = np.concatenate([np.zeros(k), [-np.inf]])
            ub = np.concatenate([np.full(k, np.inf), [np.inf]])
            
            res = lsq_linear(WX, Wp, bounds=(lb, ub), method='bvls')
            
            params = res.x
            t_S = params[:k]
            c_hat = params[k]
            
            # 8. Update Residual
            # r = p - (A_S t_S + c 1)
            predicted_signal = A_S @ t_S + c_hat * ones_vec
            r = observed_powers - predicted_signal
            
            iteration += 1
        else:
            if verbose:
                print(f"Max score {l_max:.4f} < threshold {threshold}. Stopping.")
            break
            
    # Final results
    if active_set:
        # Re-run final estimation to be sure (though loop ends with update)
        # The last update in the loop is correct for the current active_set
        pass
    else:
        t_S = np.array([])
        c_hat = 0.0
        
    return active_set, t_S, c_hat, {'scores_history': scores_history, 'final_residual': r, 'candidates_history': candidates_history}

