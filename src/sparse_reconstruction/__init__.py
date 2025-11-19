"""
Joint Sparse Superposition Reconstruction for Transmitter Localization.

This module implements an alternative approach to transmitter localization based on
sparse recovery and convex optimization. Unlike the likelihood-based two-stage approach
in src/localization/, this method performs joint optimization with sparsity constraints.

Mathematical Formulation:
------------------------
Given:
- M sensors with observed powers p ∈ ℝ^M
- N grid points (potential transmitter locations)
- Propagation matrix A_model ∈ ℝ^(M×N) with linear path gains
- Covariance matrix V ∈ ℝ^(M×M) capturing measurement uncertainty

Find sparse transmit power field t ∈ ℝ^N by solving:

    min  ‖W(A_model·t - p)‖₂² + λ‖t‖₁
    t≥0

where:
- W = V^(-1/2) is the whitening matrix
- λ ≥ 0 is the sparsity regularization parameter
- ‖·‖₁ encourages sparse solutions (few active transmitters)
- t≥0 enforces physical constraint (non-negative power)

Key Differences from Likelihood-Based Approach:
----------------------------------------------
1. Single-stage joint optimization (vs. two-stage)
2. Convex optimization with sparsity penalty (vs. per-pixel optimization + likelihood)
3. Linear superposition in power domain (vs. dB-scale path loss)
4. Explicit sparsity constraint (vs. implicit via likelihood concentration)

Modules:
--------
- propagation_matrix: Build A_model with linear path gains
- whitening: Compute W = V^(-1/2) via Cholesky decomposition
- sparse_solver: Solve the constrained LASSO problem
- reconstruction: Main pipeline tying everything together
"""

from .propagation_matrix import (
    compute_propagation_matrix,
    compute_linear_path_gain
)

from .whitening import (
    compute_whitening_matrix,
    apply_whitening
)

from .sparse_solver import (
    solve_sparse_reconstruction,
    solve_sparse_reconstruction_cvxpy,
    solve_sparse_reconstruction_sklearn
)

from .reconstruction import (
    joint_sparse_reconstruction,
    reconstruct_signal_strength_map,
    compute_signal_strength_at_points,
    dbm_to_linear,
    linear_to_dbm
)

__all__ = [
    # Propagation matrix
    'compute_propagation_matrix',
    'compute_linear_path_gain',

    # Whitening
    'compute_whitening_matrix',
    'apply_whitening',

    # Sparse solver
    'solve_sparse_reconstruction',
    'solve_sparse_reconstruction_cvxpy',
    'solve_sparse_reconstruction_sklearn',

    # Main reconstruction
    'joint_sparse_reconstruction',
    'reconstruct_signal_strength_map',
    'compute_signal_strength_at_points',

    # Utility functions
    'dbm_to_linear',
    'linear_to_dbm',
]
