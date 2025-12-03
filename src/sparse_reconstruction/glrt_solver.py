"""
Sequential "Add-One" Detection (GLRT) Solver for Sparse Reconstruction.

This module implements a greedy pursuit algorithm based on the Generalized Likelihood Ratio Test (GLRT).
Instead of optimizing the entire transmit power field jointly, it iteratively identifies the single
most likely transmitter location that explains the current residual and adds it to the support set.

Reference:
    Slide 41: "Generalized Likelihood Ratio Test (GLRT)"
"""

import numpy as np
from .sparse_solver import solve_sparse_reconstruction_scipy

def solve_iterative_glrt(A_model, W, observed_powers, 
                         glrt_max_iter=10, glrt_threshold=4.0,
                         verbose=True, **solver_kwargs):
    """
    Solve for transmit power using Sequential "Add-One" Detection (GLRT).

    Parameters
    ----------
    A_model : ndarray of shape (M, N)
        Propagation matrix (linear path gains)
    W : ndarray of shape (M, M)
        Whitening matrix
    observed_powers : ndarray of shape (M,)
        Observed powers (linear scale)
    glrt_max_iter : int, optional
        Maximum number of transmitters to find. Default: 10
    glrt_threshold : float, optional
        Minimum normalized GLRT score (R^2) to accept a new transmitter. 
        Range [0, 1]. Default: 4.0 (400% variance explained - usually requires unnormalized score)
    verbose : bool, optional
        Print progress. Default: True
    **solver_kwargs
        Additional arguments passed to the re-optimization solver (solve_sparse_reconstruction_scipy)

    Returns
    -------
    t_est : ndarray of shape (N,)
        Estimated transmit powers (linear scale)
    info : dict
        Solver information
    """
    M, N = A_model.shape
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Sequential GLRT Solver")
        print(f"{'='*60}")
        print(f"Max iterations (transmitters): {glrt_max_iter}")
        print(f"GLRT Threshold: {glrt_threshold:.2e}")

    # Initialize
    t_est = np.zeros(N)
    residual = observed_powers.copy() # r = p (initially, assuming t=0)
    support = [] # List of indices of found transmitters
    candidates_history = []
    
    # Precompute whitened propagation matrix for GLRT score calculation
    # A_w = W @ A_model
    # This might be large (M x N), but usually manageable if A_model is manageable.
    # If memory is tight, we can compute columns on the fly.
    # For M=10, N=10000, this is small (100k floats).
    A_w = W @ A_model
    
    # Precompute norms of whitened columns: ||W a_i||_2^2
    # Denominator of GLRT score
    A_w_norms_sq = np.sum(A_w**2, axis=0)
    
    # Avoid division by zero
    A_w_norms_sq[A_w_norms_sq < 1e-20] = 1e-20

    for k in range(1, glrt_max_iter + 1):
        if verbose:
            print(f"\nIteration {k}: Scanning for transmitter...")

        # 1. Calculate GLRT score for all i not in support
        # T_i = ( (W a_i)^T (W r) )^2 / ||W a_i||^2
        #     = ( a_i^w^T r^w )^2 / ||a_i^w||^2
        
        # Current whitened residual
        r_w = W @ residual
        
        # Numerator: (A_w^T r_w)^2
        # This is a matrix-vector multiplication: (N x M) @ (M x 1) -> (N x 1)
        correlations = A_w.T @ r_w
        numerator = correlations**2
        
        # GLRT Scores
        scores = numerator / A_w_norms_sq
        
        # Mask out already selected indices
        scores[support] = -1.0
        
        # Mask out excluded indices
        exclusion_mask = solver_kwargs.get('exclusion_mask', None)
        if exclusion_mask is not None:
            scores[exclusion_mask] = -1.0
        
        # 2. Select best candidate
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
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
            'iteration': k,
            'top_indices': top_10_indices,
            'top_scores': top_10_scores,
            'selected_index': best_idx,
            'selected_score': best_score
        })
        
        # Calculate residual energy for normalization
        # ||W r||^2 = r^T W^T W r = r_w^T r_w
        resid_energy = np.sum(r_w**2)
        
        # Normalized score (fraction of energy explained): 0 to 1
        if resid_energy > 1e-20:
            best_score_norm = best_score / resid_energy
        else:
            best_score_norm = 0.0
        
        if verbose:
            print(f"  Best candidate: Index {best_idx}")
            print(f"  Raw Score: {best_score:.4e}")
            print(f"  Normalized Score (R^2): {best_score_norm:.4e}")
            
        # 3. Test threshold
        # If threshold > 1.0, we assume it's a raw score threshold (e.g. 4.0)
        # If threshold <= 1.0, we assume it's a normalized score threshold (R^2)
        score_to_check = best_score if glrt_threshold > 1.0 else best_score_norm
        
        if score_to_check < glrt_threshold:
            if verbose:
                print(f"  Normalized score below threshold ({glrt_threshold:.2e}). Stopping.")
            break
            
        # 4. Update support
        support.append(best_idx)
        
        # 5. Re-optimize powers for found sources
        # We use the log-domain solver restricted to the support set
        if verbose:
            print(f"  Re-optimizing with {len(support)} sources...")
            
        # Extract sub-matrix for support
        A_sub = A_model[:, support]
        
        # We need to map the sub-solution back to the full t_est
        # The solver expects A_sub (M x K) and returns t_sub (K,)
        
        # IMPORTANT: During re-optimization of the support, we should NOT apply 
        # strong sparsity regularization, as we have already selected the sparse support.
        # We want to fit the data (Least Squares / MLE).
        # So we force lambda_reg to 0.0 (or very small) for this step.
        
        kwargs_copy = solver_kwargs.copy()
        kwargs_copy['lambda_reg'] = 0.0 # Disable sparsity penalty for support fitting
        
        # Also disable reweighting for the sub-problem as it's not needed
        kwargs_copy['enable_reweighting'] = False
        
        # Handle spatial_weights slicing
        spatial_weights = kwargs_copy.get('spatial_weights', None)
        if spatial_weights is not None:
            kwargs_copy['spatial_weights'] = spatial_weights[support]
            
        # Disable exclusion_mask for sub-problem (already handled in selection)
        kwargs_copy['exclusion_mask'] = None
        
        t_sub, sub_info = solve_sparse_reconstruction_scipy(
            A_sub, W, observed_powers, 
            verbose=False, # Keep it quiet
            **kwargs_copy
        )
        
        # Update full estimate
        t_est[:] = 0.0
        t_est[support] = t_sub
        
        # 6. Update residual
        # r = p - A t_est
        # Note: The user prompt says r = p - p_hat_th.
        # p_hat_th = A @ t_est
        p_hat = A_model @ t_est
        residual = observed_powers - p_hat
        
        if verbose:
            # Calculate current error
            # ||W(p - p_hat)||^2 is not exactly what GLRT minimizes (it minimizes log diff),
            # but it's a good metric for the residual power.
            resid_norm = np.linalg.norm(W @ residual)
            print(f"  Residual norm: {resid_norm:.4e}")

    # Final info
    info = {
        'solver_used': 'glrt',
        'n_iter': k,
        'n_nonzero': len(support),
        'support': support,
        'final_score': best_score if 'best_score' in locals() else 0.0,
        'candidates_history': candidates_history,
        'success': True
    }
    
    return t_est, info
