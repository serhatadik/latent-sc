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
    
    whitening_method = solver_kwargs.get('whitening_method', None)
    
    if whitening_method == 'hetero_geo_aware':
        # Dynamic Whitening Integration
        if verbose:
            print("  Initializing hetero_geo_aware dynamic precomputation...")
            
        geometric_features = solver_kwargs.get('geometric_features') # Shape (M, N, 4)
        if geometric_features is None:
            raise ValueError("geometric_features required for hetero_geo_aware whitening")

        sigma_noise = solver_kwargs.get('sigma_noise', 1e-13)
        eta = solver_kwargs.get('eta', 0.5)
        # Default kernel widths RHO for [LOS, Elev, Obstacles, Dist]
        # LOS: binary, so small rho < 1 separates 0 and 1. 0.1?
        # Elev: degrees. 10-20 degrees?
        # Obs: count. 1-2?
        # Dist: meters. 100-500m?
        feature_rho = solver_kwargs.get('feature_rho', [0.1, 15.0, 1.0, 200.0])
        feature_rho = np.array(feature_rho)
        
        # We need to compute GLRT score for each candidate i:
        # T_i = ( (V_i^-1 a_i)^T r )^2 / ( a_i^T V_i^-1 a_i )
        # where V_i depends on i
        
        # V_i = D_i * C_i * D_i
        # D_i = diag( sigma(p_k) ) -> Diagonal matrix of std devs
        # C_i = Kernel matrix between sensors for transmitter i
        
        # Since M is small (~6-20), we can loop over N candidates?
        # Loop over N might be slow in Python if N is large (10k-100k).
        # We should try to vectorize or use numba/caching if possible.
        # But V_i^-1 needs to be computed for each i. Inverting M x M matrix N times.
        # If M=10, N=10000 -> 10000 inversions of 10x10. Very fast.
        
        # Let's precompute:
        # 1. Standard Deviations D(i) for each candidate i
        #    sigma_k(i) = sqrt(sigma_noise^2 + eta^2 * A_ki^2 * t_ref^2?) 
        #    Wait, slide says V = diag(sigma_noise^2 + eta^2 p_k^2). 
        #    What is p_k? It is the signal power from candidate i?
        #    Usually for GLRT we test if there is a signal.
        #    The variance model usually assumes the variance depends on the signal strength.
        #    In the "Add-One" test, we are testing a candidate with unknown amplitude `alpha`.
        #    But the covariance V is usually assumed fixed or estimated.
        #    If V depends on the signal we are estimating, it becomes iterative/complex.
        #    However, the request says: "V = diag(sigma_noise^2 + eta^2 p_k^2)". 
        #    Is p_k the OBSERVED power or the HYPOTHESIZED power?
        #    "Slide 15 defines ... based on signal power."
        #    Usually in this context (hetero_diag option implementation), p_k was the OBSERVED power?
        #    No, looking at reconstruction.py for hetero_diag:
        #    `W = compute_whitening_matrix(..., observed_powers=observed_powers_linear)`
        #    So for hetero_diag, it used observed powers.
        #    
        #    BUT, the prompt says: "feature vector ... between transmitter i and receiver k".
        #    "V_kl(i, g) = sigma^2(i) * K(...)". 
        #    Wait, "sigma^2(i)"?
        #    "The final element Sigma_ij of your merged matrix would be: D_ii * C_ij * D_jj"
        #    "D_ii = sqrt(sigma_noise^2 + eta^2 P_i^2)" where "P_i: The power at sensor i."
        #    If P_i is the observed power, then D is constant for all candidates!
        #    "Slide 15 defines the heteroscedastic diagonal elements based on signal power."
        #    If it's based on observed power, then D is fixed (M x M diagonal).
        
        #    Let's assume P_i is the OBSERVED power at sensor i (which drives the noise variance).
        #    In that case, D is independent of candidate i.
        #    ONLY C (Correlation) depends on candidate i (geometry).
        
        #    So, V_i = D @ C_i @ D.
        #    Inverse: V_i^-1 = D^-1 @ C_i^-1 @ D^-1.
        
        #    This simplifies things!
        
        # 1. Compute D and D^-1 (fixed)
        # observed_powers is linear.
        obs_var = sigma_noise**2 + (eta * observed_powers)**2
        D_diag = np.sqrt(obs_var)
        D_inv_diag = 1.0 / (D_diag + 1e-20)
        
        # 2. Precompute V_i inverse components for all i
        # We need term1[i] = a_i^T V_i^-1
        # And term2[i] = a_i^T V_i^-1 a_i  (scalar)
        
        # To avoid storing (N, M, M) matrices, we can just store what we need.
        # Actually, for GLRT score we need:
        # Score_i = ( (V_i^-1 a_i)^T r )^2 / ( a_i^T V_i^-1 a_i )
        #         = ( r^T V_i^-1 a_i )^2 / ...
        #         = ( r^T D^-1 C_i^-1 D^-1 a_i )^2 / ( a_i^T D^-1 C_i^-1 D^-1 a_i )
        
        # Let's define normalized vectors:
        # r_norm = D^-1 r
        # a_i_norm = D^-1 a_i  (vector for candidate i)
        
        # Score_i = ( r_norm^T C_i^-1 a_i_norm )^2 / ( a_i_norm^T C_i^-1 a_i_norm )
        
        # So we need to compute C_i and its inverse for each i.
        
        # Scale D^-1
        # A_model: (M, N)
        # r: (M,)
        
        D_inv_mat = np.diag(D_inv_diag)
        A_norm = D_inv_mat @ A_model  # (M, N)
        r_norm_init = D_inv_mat @ observed_powers # Initial r_norm
        
        # We need to function to compute C_i_inv @ vec efficiently?
        # Since M is small (~10), explicit inversion is fine.
        
        # Precompute C_i_inv for all i?
        # M^2 * N floats. 100 * N. If N=10000 -> 1M floats = 8MB. Cheap.
        
        # Compute C_i matrices
        # features: (M, N, 4) -> Transpose to (N, M, 4) for iteration?
        features_T = geometric_features.transpose(1, 0, 2) # (N, M, 4)
        
        C_inv_storage = np.zeros((N, M, M), dtype=np.float32)
        
        # rho for broadcasting: (1, 1, 4)
        rho_sq = feature_rho**2
        
        if verbose:
             print(f"  Precomputing correlation matrices for {N} candidates...")
             
        for i in range(N):
            # Features for sensors w.r.t candidate i: F_i (M, 4)
            F_i = features_T[i]
            
            # Compute Pairwise distances between rows of F_i (M x M)
            # Diff: (M, 1, 4) - (1, M, 4) -> (M, M, 4)
            diff = F_i[:, np.newaxis, :] - F_i[np.newaxis, :, :]
            
            # Weighted squared euclidean distance
            # Sum over features
            dist_sq = np.sum((diff**2) / rho_sq, axis=2) # (M, M)
            
            # Kernel K_ij = exp( - dist_sq / 2 )
            K = np.exp(-0.5 * dist_sq)
            
            # Normalize to Correlation Matrix C
            # C_ij = K_ij / sqrt(K_ii * K_jj)
            # But K_ii = exp(0) = 1. So C = K.
            C = K
            
            # Add small jitter for stability
            C += np.eye(M) * 1e-6
            
            # Invert
            try:
                # Use cholesky or inv? inv is safer for general C
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                 C_inv = np.linalg.pinv(C)
                 
            C_inv_storage[i] = C_inv
            
        # Precompute denominators: a_i_norm^T C_i^-1 a_i_norm
        # A_norm[:, i] is a_i_norm
        
        denom_storage = np.zeros(N)
        for i in range(N):
            a_vec = A_norm[:, i]
            # val = a^T C^-1 a
            val = a_vec @ C_inv_storage[i] @ a_vec
            denom_storage[i] = val
            
        denom_storage[denom_storage < 1e-20] = 1e-20
        
        # For the loop, we will use these precomputed terms
        # A_w, A_w_norms_sq are NOT used in the same way.
        # We will use A_norm, C_inv_storage, denom_storage.
        
        dynamic_whitening = True
        
    else:
        dynamic_whitening = False
        A_w = W @ A_model
        A_w_norms_sq = np.sum(A_w**2, axis=0)
        A_w_norms_sq[A_w_norms_sq < 1e-20] = 1e-20


    for k in range(1, glrt_max_iter + 1):
        if verbose:
            print(f"\nIteration {k}: Scanning for transmitter...")

        # 1. Calculate GLRT score for all i not in support
        # T_i = ( (W a_i)^T (W r) )^2 / ||W a_i||^2
        #     = ( a_i^w^T r^w )^2 / ||a_i^w||^2
        
        if dynamic_whitening:
             # Dynamic Score Calculation
             # r_current = residual (in linear power)
             
             # r_norm = D^-1 r
             r_norm = D_inv_mat @ residual
             
             # Numerator = ( r_norm^T C_i^-1 a_i_norm )^2
             # This is tricky because C_i^-1 depends on i.
             # We need to compute this for all i.
             # r_norm is (M,)
             # A_norm is (M, N)
             # C_inv_storage is (N, M, M)
             
             # Vectorized attempt:
             # Term = sum_j sum_k r_j * (C_inv_i)_jk * A_norm_k,i
             
             # Better: Compute vec_i = C_inv_i @ r_norm  -> (N, M)
             # Then: dot(vec_i, A_norm_i) -> (N,)
             
             # Einsum: 'nxy,y->nx' multiply (N,M,M) by (M,) -> (N,M)
             # C_inv_storage (N, M, M), r_norm (M) -> (N, M)
             # Requires N ops. C_inv_storage is N x M^2.
             # MatVec for each N.
             
             # weighted_r = np.einsum('nij,j->ni', C_inv_storage, r_norm)
             # BUT einsum might vary in performance. simple loop or tensordot?
             # tensordot over 1 axis? No, indices mismatch.
             # matmul: (N, M, M) @ (N, M, 1)?
             # r_val = r_norm.reshape(1, M, 1) -> broadcast?
             # (N, M, M) @ (1, M, 1) -> (N, M, 1)
             
             weighted_r_all = np.matmul(C_inv_storage, r_norm) # (N, M)
             
             # Now dot with a_i_norm column-wise
             # A_norm is (M, N). Transpose to (N, M)
             A_norm_T = A_norm.T # (N, M)
             
             # dot product per row
             correlations = np.sum(weighted_r_all * A_norm_T, axis=1) # (N,)
             
             numerator = correlations**2
             scores = numerator / denom_storage
             
        else:
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
        
        # Calculate residual energy for normalization
        if dynamic_whitening:
             # For dynamic whitening, the metric is slightly ambiguous because W changes.
             # But usually we normalize by the best candidate's whitening?
             # Or just the weighted residual energy with respect to the best candidate's covariance?
             # Let's use the selected best_idx's covariance.
             if best_idx is not None and best_idx >= 0:
                 # resid_energy = r^T V_best^-1 r
                 # = r_norm^T C_best^-1 r_norm
                 
                 C_inv_best = C_inv_storage[best_idx]
                 resid_energy = r_norm @ C_inv_best @ r_norm
             else:
                 resid_energy = 1.0 # fallback
        else:
            # ||W r||^2 = r^T W^T W r = r_w^T r_w
            resid_energy = np.sum(r_w**2)
        
        # Normalized score (fraction of energy explained): 0 to 1
        if resid_energy > 1e-20:
            best_score_norm = best_score / resid_energy
        else:
            best_score_norm = 0.0

        # Calculate Corrected Score for hetero_geo_aware
        geo_aware_score = 0.0
        normalized_val_for_history = best_score_norm # Default to R^2

        if whitening_method == 'hetero_geo_aware':
            # The raw GLRT score is scaled by the inverse correlation matrix C^-1.
            # If C has small eigenvalues (high correlation), C^-1 has large entries, inflating the score.
            # We normalize by the mean diagonal element of C^-1 (average amplification) 
            # to bring the score back to a scale comparable to the diagonal case.
            
            C_inv_best = C_inv_storage[best_idx]
            amplification = np.mean(np.diag(C_inv_best))
            if amplification < 1e-6:
                amplification = 1.0
                
            geo_aware_score = best_score / amplification
            normalized_val_for_history = geo_aware_score
            
            if verbose:
                print(f"  Covariance Amplification: {amplification:.2e}")
                print(f"  Geo-Aware Corrected Score: {geo_aware_score:.4e}")

        if verbose:
            print(f"  Best candidate: Index {best_idx}")
            print(f"  Raw Score: {best_score:.4e}")
            print(f"  Normalized Score (R^2): {best_score_norm:.4e}")
            if cluster_info is not None:
                print(f"  Selection method: {cluster_info['method']}")
                print(f"  Clusters found: {cluster_info['n_clusters']}, Selected cluster size: {cluster_info['cluster_size']}")

        # Store top-K candidates for visualization (uses cluster_max_candidates)
        # This ensures consistency: same number of candidates for clustering and visualization
        top_k = cluster_max_candidates if cluster_max_candidates and cluster_max_candidates > 0 else 100
        if len(scores) >= top_k:
            top_k_indices = np.argpartition(scores, -top_k)[-top_k:]
            # Sort them by score descending
            top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
        else:
            top_k_indices = np.argsort(scores)[::-1]
            
        top_k_scores = scores[top_k_indices]
        
        candidates_history.append({
            'iteration': k,
            'top_indices': top_k_indices,
            'top_scores': top_k_scores,
            'selected_index': best_idx,
            'selected_score': best_score,
            'normalized_score': normalized_val_for_history,
            'cluster_info': cluster_info
        })
        
        # 3. Test threshold
        # If whitening_method is hetero_geo_aware, use corrected score
        if whitening_method == 'hetero_geo_aware':
             score_to_check = geo_aware_score
        else:
            # Traditional logic for other methods
            # If threshold > 1.0, we assume it's a raw score threshold (e.g. 4.0)
            # If threshold <= 1.0, we assume it's a normalized score threshold (R^2)
            score_to_check = best_score if glrt_threshold > 1.0 else best_score_norm
        
        if score_to_check < glrt_threshold:
            if verbose:
                print(f"  Score ({score_to_check:.4e}) below threshold ({glrt_threshold:.2e}). Stopping.")
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
        
        # Remove partial whitening parameters that might confuse the sub-solver or scipy
        for key in ['geometric_features', 'feature_rho', 'whitening_method', 'sigma_noise', 'eta']:
            kwargs_copy.pop(key, None)
            
        W_for_sub = W
        if dynamic_whitening:
             # Construct W for the sub-problem based on the most recent candidate (best_idx)
             # This is an approximation if multiple sources are present, but better than nothing.
             # V_best^-1 = D^-1 C_best^-1 D^-1
             # We want W such that W^T W = V^-1
             # C_best^-1 = L L^T (Cholesky of inverse)
             # Then V^-1 = D^-1 L L^T D^-1 = (L^T D^-1)^T (L^T D^-1)
             # So W = L^T D^-1
             
             C_inv_best = C_inv_storage[best_idx]
             try:
                 L = np.linalg.cholesky(C_inv_best)
                 W_part = L.T
                 W_for_sub = W_part @ D_inv_mat
             except np.linalg.LinAlgError:
                 # Fallback to diagonal
                 W_for_sub = D_inv_mat 

        t_sub, sub_info = solve_sparse_reconstruction_scipy(
            A_sub, W_for_sub, observed_powers, 
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
            if dynamic_whitening:
                 # Use W from best_idx
                 C_inv_best = C_inv_storage[best_idx]
                 # r^T V^-1 r = r_norm^T C^-1 r_norm
                 resid_norm_sq = r_norm @ C_inv_best @ r_norm
                 resid_norm = np.sqrt(resid_norm_sq)
            else:
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
                # Disable exclusion_mask for sub-problem (already handled in selection)
                kwargs_copy['exclusion_mask'] = None
                
                # Remove partial whitening parameters for sub-solver
                for key in ['geometric_features', 'feature_rho', 'whitening_method', 'sigma_noise', 'eta']:
                    kwargs_copy.pop(key, None)
                
                W_for_dedupe = W
                if dynamic_whitening:
                    # Use the W from the primary support (first kept one?)
                    # rough approximation
                    idx_primary = support[0] if len(support) > 0 else 0
                    C_inv_best = C_inv_storage[idx_primary]
                    try:
                        L = np.linalg.cholesky(C_inv_best)
                        W_for_dedupe = L.T @ D_inv_mat
                    except np.linalg.LinAlgError:
                        W_for_dedupe = D_inv_mat

                t_sub, _ = solve_sparse_reconstruction_scipy(
                    A_sub, W_for_dedupe, observed_powers,
                    verbose=False,
                    **kwargs_copy
                )
                
                t_est[:] = 0.0
                t_est[support] = t_sub
    
    # Final info
    info = {
        'solver_used': 'glrt',
        'whitening_method': whitening_method,
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
