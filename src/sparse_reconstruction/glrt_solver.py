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


def _find_clusters_and_select_centroid(scores, map_shape, scale=1.0,
                                        threshold_fraction=0.1,
                                        cluster_distance_m=100.0,
                                        max_candidates=100,
                                        excluded_indices=None):
    """
    Identify clusters of high-scoring locations and return the centroid 
    of the strongest cluster. Uses KDTree for O(C log C) performance.
    
    Parameters
    ----------
    scores : ndarray of shape (N,)
        GLRT scores for all grid locations
    map_shape : tuple of (height, width)
        Shape of the reconstruction grid
    scale : float
        Pixel-to-meter scaling factor
    threshold_fraction : float
        Fraction of max score for candidate inclusion (e.g., 0.1 = 10%)
    cluster_distance_m : float
        Maximum distance in meters to consider two candidates as part of same cluster
    max_candidates : int
        Maximum number of top-scoring candidates to consider for clustering.
        Default: 100. Set to None or 0 to disable limit.
    excluded_indices : array-like, optional
        Indices to exclude from clustering (e.g., already selected support)
        
    Returns
    -------
    best_idx : int
        Index of the grid point at/near the centroid of the strongest cluster.
        Falls back to argmax if clustering fails.
    cluster_info : dict
        Information about the clustering result
    """
    from scipy.spatial import cKDTree
    
    height, width = map_shape
    
    # Work with a copy of scores for masking
    working_scores = scores.copy()
    if excluded_indices is not None and len(excluded_indices) > 0:
        working_scores[excluded_indices] = -1.0
    
    max_score = working_scores.max()
    if max_score <= 0:
        return np.argmax(scores), {'n_clusters': 0, 'cluster_size': 0, 'method': 'fallback_max'}
    
    # Get candidate indices: either top-K or threshold-based
    if max_candidates and max_candidates > 0:
        # Select top-K candidates by score (fast and focused on peaks)
        if len(working_scores) > max_candidates:
            # Use argpartition for O(N) selection of top-K
            top_k_indices = np.argpartition(working_scores, -max_candidates)[-max_candidates:]
            # Filter out negative scores (excluded indices)
            candidate_indices = top_k_indices[working_scores[top_k_indices] > 0]
        else:
            candidate_indices = np.where(working_scores > 0)[0]
    else:
        # Original threshold-based selection
        threshold = threshold_fraction * max_score
        candidate_indices = np.where(working_scores > threshold)[0]
    
    n_candidates = len(candidate_indices)
    if n_candidates == 0:
        return np.argmax(scores), {'n_clusters': 0, 'cluster_size': 0, 'method': 'fallback_max'}
    
    if n_candidates == 1:
        return candidate_indices[0], {'n_clusters': 1, 'cluster_size': 1, 'method': 'single_candidate'}
    
    # Convert flat indices to (row, col) coordinates
    candidate_rows = candidate_indices // width
    candidate_cols = candidate_indices % width
    candidate_coords = np.column_stack((candidate_rows, candidate_cols))
    
    # Convert distance threshold to pixel units
    cluster_distance_px = cluster_distance_m / scale
    
    # Use KDTree for O(C log C) neighbor finding
    tree = cKDTree(candidate_coords)
    neighbor_pairs = tree.query_pairs(r=cluster_distance_px, output_type='ndarray')
    
    # Union-Find with path compression
    parent = np.arange(n_candidates)
    
    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Build clusters from neighbor pairs
    for i, j in neighbor_pairs:
        union(i, j)
    
    # Group candidates by cluster root
    clusters = {}
    for i in range(n_candidates):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
    
    # Find the strongest cluster (by sum of scores)
    best_cluster_root = None
    best_cluster_score = -1.0
    candidate_scores = working_scores[candidate_indices]
    
    for root, members in clusters.items():
        cluster_score = candidate_scores[members].sum()
        if cluster_score > best_cluster_score:
            best_cluster_score = cluster_score
            best_cluster_root = root
    
    if best_cluster_root is None:
        return np.argmax(scores), {'n_clusters': 0, 'cluster_size': 0, 'method': 'fallback_max'}
    
    # Get members of the best cluster
    best_members = np.array(clusters[best_cluster_root])
    best_member_indices = candidate_indices[best_members]
    best_member_scores = candidate_scores[best_members]
    best_member_coords = candidate_coords[best_members]
    
    # Compute weighted centroid (scores as weights)
    total_weight = best_member_scores.sum()
    if total_weight > 0:
        centroid = (best_member_coords * best_member_scores[:, np.newaxis]).sum(axis=0) / total_weight
    else:
        centroid = best_member_coords.mean(axis=0)
    
    # Snap to nearest grid point (among the cluster members)
    distances_sq = ((best_member_coords - centroid) ** 2).sum(axis=1)
    nearest_idx = np.argmin(distances_sq)
    best_idx = best_member_indices[nearest_idx]
    
    cluster_info = {
        'n_clusters': len(clusters),
        'cluster_size': len(best_members),
        'n_candidates': n_candidates,
        'centroid_row': centroid[0],
        'centroid_col': centroid[1],
        'cluster_score_sum': best_cluster_score,
        'method': 'cluster_centroid'
    }
    
    return best_idx, cluster_info


def solve_iterative_glrt(A_model, W, observed_powers, 
                         glrt_max_iter=10, glrt_threshold=4.0,
                         selection_method='max', map_shape=None, scale=1.0,
                         cluster_distance_m=100.0, cluster_threshold_fraction=0.1,
                         cluster_max_candidates=100, dedupe_distance_m=25.0,
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
    selection_method : {'max', 'cluster'}, optional
        Method for selecting the best candidate. Default: 'max'
        - 'max': Select the single location with maximum GLRT score
        - 'cluster': Identify clusters of high-scoring locations and select 
          the centroid of the strongest cluster (by sum of scores)
    map_shape : tuple of (height, width), optional
        Shape of the reconstruction grid. Required if selection_method='cluster'.
    scale : float, optional
        Pixel-to-meter scaling factor. Default: 1.0. Used for cluster distance calculation.
    cluster_distance_m : float, optional
        Maximum distance in meters to consider two candidates as part of the same cluster.
        Default: 100.0 meters.
    cluster_threshold_fraction : float, optional
        Fraction of max score for candidate inclusion in clustering (e.g., 0.1 = 10%).
        Default: 0.1. Only used when max_candidates is None or 0.
    cluster_max_candidates : int, optional
        Maximum number of top-scoring candidates to consider for clustering.
        Default: 100. Set to None or 0 to use threshold_fraction instead.
    dedupe_distance_m : float, optional
        Distance threshold in meters for deduplicating transmitters after iterations.
        Transmitters within this distance of each other are merged, keeping the one
        added earliest. Default: 25.0. Set to 0 or None to disable deduplication.
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
        cluster_info = None
        if selection_method == 'cluster':
            if map_shape is None:
                raise ValueError("map_shape is required when selection_method='cluster'")
            best_idx, cluster_info = _find_clusters_and_select_centroid(
                scores, map_shape, scale=scale,
                threshold_fraction=cluster_threshold_fraction,
                cluster_distance_m=cluster_distance_m,
                max_candidates=cluster_max_candidates,
                excluded_indices=support + (list(np.where(exclusion_mask)[0]) if exclusion_mask is not None else [])
            )
        else:  # 'max' - original behavior
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
            'selected_score': best_score,
            'cluster_info': cluster_info
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
            if cluster_info is not None:
                print(f"  Selection method: {cluster_info['method']}")
                print(f"  Clusters found: {cluster_info['n_clusters']}, Selected cluster size: {cluster_info['cluster_size']}")
            
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

    # Deduplicate transmitters that are within dedupe_distance_m of each other
    # Keep the one added earliest (first in support list)
    if len(support) > 1 and map_shape is not None and dedupe_distance_m and dedupe_distance_m > 0:
        height, width = map_shape
        dedupe_distance_px = dedupe_distance_m / scale
        
        # Convert support indices to (row, col) coordinates
        support_rows = np.array([idx // width for idx in support])
        support_cols = np.array([idx % width for idx in support])
        support_coords = np.column_stack((support_rows, support_cols))
        
        # Track which indices to keep (earliest one in each group)
        keep_mask = np.ones(len(support), dtype=bool)
        
        # For each transmitter, check if any earlier transmitter is within dedupe distance
        for i in range(1, len(support)):
            if not keep_mask[i]:
                continue
            # Check distance to all earlier transmitters that are still kept
            for j in range(i):
                if not keep_mask[j]:
                    continue
                dist = np.sqrt(np.sum((support_coords[i] - support_coords[j])**2))
                dist_m = dist * scale
                if dist_m <= dedupe_distance_m:
                    # Remove the later one (i), keep the earlier one (j)
                    keep_mask[i] = False
                    if verbose:
                        print(f"\nDeduplication: Removing transmitter at index {support[i]} "
                              f"(iter {i+1}), within {dist_m:.1f}m of index {support[j]} (iter {j+1})")
                    break
        
        # Apply deduplication
        original_support = support.copy()
        support = [s for s, keep in zip(support, keep_mask) if keep]
        n_removed = len(original_support) - len(support)
        
        if n_removed > 0:
            if verbose:
                print(f"\nDeduplication: Removed {n_removed} duplicate transmitter(s), "
                      f"{len(support)} remaining")
            
            # Re-optimize with deduplicated support
            if len(support) > 0:
                A_sub = A_model[:, support]
                kwargs_copy = solver_kwargs.copy()
                kwargs_copy['lambda_reg'] = 0.0
                kwargs_copy['enable_reweighting'] = False
                spatial_weights = kwargs_copy.get('spatial_weights', None)
                if spatial_weights is not None:
                    kwargs_copy['spatial_weights'] = spatial_weights[support]
                kwargs_copy['exclusion_mask'] = None
                
                t_sub, _ = solve_sparse_reconstruction_scipy(
                    A_sub, W, observed_powers,
                    verbose=False,
                    **kwargs_copy
                )
                
                t_est[:] = 0.0
                t_est[support] = t_sub
    
    # Final info
    info = {
        'solver_used': 'glrt',
        'selection_method': selection_method,
        'n_iter': k,
        'n_nonzero': len(support),
        'support': support,
        'final_score': best_score if 'best_score' in locals() else 0.0,
        'candidates_history': candidates_history,
        'deduplication_applied': len(support) < k if 'k' in locals() else False,
        'success': True
    }
    
    return t_est, info
