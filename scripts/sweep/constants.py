"""Shared constants for the comprehensive parameter sweep."""

from pathlib import Path

# Project root: scripts/sweep/constants.py -> scripts/sweep -> scripts -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Known transmitter names (alphabetically sorted for canonical ordering)
KNOWN_TRANSMITTERS = ['guesthouse', 'mario', 'moran', 'ustar', 'wasatch']

# Whitening method configurations to sweep
# Format: (whitening_method, feature_rho)
# feature_rho: [LOS, Elevation Angle (deg), Obstacle Count, Distance (m)]
# Smaller rho = feature differences decorrelate sensor errors faster
AVAILABLE_WHITENING_CONFIGS = {
    'hetero_diag': ('hetero_diag', None),  # Diagonal heteroscedastic (baseline, no geometry)
    'hetero_diag_obs': ('hetero_diag_obs', None),  # Diagonal using observed std from data files
    'hetero_geo_aware': ('hetero_geo_aware', [0.5, 10.0, 1e6, 150.0]),  # Geometry-aware with physics-based rho
    'hetero_spatial': ('hetero_spatial', None),  # Heteroscedastic + Spatial Correlation (Exp Decay)
}

# Power density thresholds to sweep
POWER_DENSITY_THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.3]

# Desired column order for results CSV
DESIRED_COLUMN_ORDER = [
    'dir_name', 'tx_count', 'transmitters', 'seed', 'strategy', 'selection_method',
    'power_filtering', 'power_threshold', 'whitening_config', 'sigma_noise',
    'sigma_noise_dB', 'use_edf', 'edf_thresh', 'use_robust', 'robust_thresh', 'pooling_lambda', 'dedupe_dist',
    # Original metrics (from GLRT support)
    'ale', 'tp', 'fp', 'fn', 'pd', 'precision', 'f1_score', 'n_estimated',
    # Combination metrics
    'combo_n_tx', 'combo_ale', 'combo_tp', 'combo_fp', 'combo_fn', 'combo_pd', 'combo_precision',
    'combo_count_error', 'combo_rmse', 'combo_bic',
    # Reconstruction error metrics
    'recon_rmse', 'recon_mae', 'recon_bias', 'recon_max_error', 'recon_n_val_points', 'recon_noise_floor', 'recon_status',
]

# Cache directory for TIREM
TIREM_CACHE_DIR = _PROJECT_ROOT / "data" / "cache" / "tirem"
