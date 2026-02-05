"""
Comprehensive Reconstruction Parameter Sweep Script

Automatically processes all datasets in data/processed/, sweeping over various
reconstruction parameters (sigma_noise strategies, selection methods, feature_rho),
and generates comprehensive analysis reports by transmitter count (1-5) and universally.

This script:
1. Auto-discovers all data directories in data/processed/
2. Groups directories by TX count (1-5)
3. Runs reconstruction with various parameter combinations
4. Generates GLRT visualization for each unique data directory
5. Analyzes results per TX count and universally
6. Generates comprehensive reports and visualizations

Usage:
    python scripts/comprehensive_parameter_sweep.py
    python scripts/comprehensive_parameter_sweep.py --test  # Quick test mode
    python scripts/comprehensive_parameter_sweep.py --tx-counts 1,2,3  # Specific TX counts
    python scripts/comprehensive_parameter_sweep.py --nloc 30  # Only nloc30 directories
"""

# IMPORTANT: Set threading env vars BEFORE importing numpy to prevent deadlocks
# when using multiprocessing on Windows
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import yaml
import sys
import time
import argparse
import re
import os
import multiprocessing
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import utility functions
from src.utils import (
    load_slc_map,
    load_monitoring_locations,
    get_sensor_locations_array,
    load_transmitter_locations,
)

# Import sparse reconstruction
from src.sparse_reconstruction import (
    joint_sparse_reconstruction,
    dbm_to_linear,
)

# Import evaluation metrics
from src.evaluation.metrics import compute_localization_metrics
from src.evaluation.reconstruction_validation import (
    compute_reconstruction_error,
    check_validation_data_exists,
)

# Import candidate analysis functions
from scripts.candidate_analysis import (
    compute_candidate_power_rmse,
    filter_candidates_by_rmse,
    save_candidate_power_analysis,
    run_combinatorial_selection,
    recompute_powers_with_propagation_model,
    refit_with_per_tx_exponents,
)


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
    'hetero_spatial': ('hetero_spatial', None), # Heteroscedastic + Spatial Correlation (Exp Decay)
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
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
TIREM_CACHE_DIR = _PROJECT_ROOT / "data" / "cache" / "tirem"


def check_tirem_cache_exists(
    sensor_locations: np.ndarray,
    map_shape: Tuple[int, int],
    scale: float,
    tirem_config_path: str = 'config/tirem_parameters.yaml',
) -> Tuple[bool, bool]:
    """
    Check if TIREM cache files exist for the given sensor locations.
    
    Parameters
    ----------
    sensor_locations : ndarray
        Sensor locations in pixel coordinates
    map_shape : tuple
        Map shape (height, width)
    scale : float
        Pixel-to-meter scale
    tirem_config_path : str
        Path to TIREM config file
        
    Returns
    -------
    tuple of (bool, bool)
        (features_cached, prop_matrix_cached)
    """
    import hashlib
    import json
    
    # Load TIREM config to get map path
    with open(tirem_config_path, 'r') as f:
        tirem_config = yaml.safe_load(f)
    
    # Resolve map path (same logic as tirem_wrapper.py)
    map_file = tirem_config.get('map_file', 'SLCmap_5May2022.mat')
    if not Path(map_file).is_absolute():
        map_path = Path(tirem_config_path).parent.parent / map_file
        if not map_path.exists():
            map_path = Path.cwd() / map_file
        map_path = str(map_path.resolve())
    else:
        map_path = str(Path(map_file).resolve())
    
    # Build cache params (same as tirem_wrapper.py)
    cache_params = {
        'sensor_locations': sensor_locations.tolist() if isinstance(sensor_locations, np.ndarray) else sensor_locations,
        'map_shape': list(map_shape),
        'scale': float(scale),
        'tirem_config': tirem_config,
        'map_path': map_path
    }
    
    # Check propagation matrix cache
    cache_string = json.dumps(cache_params, sort_keys=True)
    cache_hash = hashlib.md5(cache_string.encode('utf-8')).hexdigest()
    prop_cache_file = TIREM_CACHE_DIR / f"tirem_prop_matrix_{cache_hash}.npy"
    prop_cached = prop_cache_file.exists()
    
    # Also check legacy cache (relative path)
    if not prop_cached:
        try:
            legacy_map_path = str(Path('..') / Path(map_path).name)
            legacy_params = cache_params.copy()
            legacy_params['map_path'] = legacy_map_path
            legacy_string = json.dumps(legacy_params, sort_keys=True)
            legacy_hash = hashlib.md5(legacy_string.encode('utf-8')).hexdigest()
            legacy_file = TIREM_CACHE_DIR / f"tirem_prop_matrix_{legacy_hash}.npy"
            prop_cached = legacy_file.exists()
        except:
            pass
    
    # Check feature cache
    feature_params = cache_params.copy()
    feature_params['type'] = 'geometric_features'
    feature_string = json.dumps(feature_params, sort_keys=True)
    feature_hash = hashlib.md5(feature_string.encode('utf-8')).hexdigest()
    feature_cache_file = TIREM_CACHE_DIR / f"tirem_features_{feature_hash}.npy"
    features_cached = feature_cache_file.exists()
    
    # Also check legacy feature cache
    if not features_cached:
        try:
            legacy_map_path = str(Path('..') / Path(map_path).name)
            legacy_params = feature_params.copy()
            legacy_params['map_path'] = legacy_map_path
            legacy_string = json.dumps(legacy_params, sort_keys=True)
            legacy_hash = hashlib.md5(legacy_string.encode('utf-8')).hexdigest()
            legacy_file = TIREM_CACHE_DIR / f"tirem_features_{legacy_hash}.npy"
            features_cached = legacy_file.exists()
        except:
            pass
    
    return features_cached, prop_cached


def parse_directory_name(dir_name: str) -> Tuple[List[str], Optional[int], Optional[int]]:
    """
    Parse a data directory name to extract transmitter names, num_locations, and seed.

    Examples:
        'mario_moran_nloc10_seed_32' -> (['mario', 'moran'], 10, 32)
        'guesthouse_wasatch_ustar_nloc10_seed_5' -> (['guesthouse', 'wasatch', 'ustar'], 10, 5)
        'mario_nloc10' -> (['mario'], 10, None)
        'mario_moran_seed_32' -> (['mario', 'moran'], None, 32)  # Legacy format (no nloc)
        'mario' -> (['mario'], None, None)
        'validation_mario' -> ([], None, None)  # Skip validation directories

    Parameters
    ----------
    dir_name : str
        Directory name to parse

    Returns
    -------
    tuple
        (list of transmitter names, num_locations or None, seed value or None)
    """
    # Skip validation directories
    if dir_name.startswith('validation_'):
        return [], None, None

    # Check for seed pattern
    seed_match = re.search(r'_seed_(\d+)$', dir_name)
    seed = int(seed_match.group(1)) if seed_match else None

    # Remove seed suffix if present
    name_part = re.sub(r'_seed_\d+$', '', dir_name)

    # Check for nloc pattern
    nloc_match = re.search(r'_nloc(\d+)$', name_part)
    num_locations = int(nloc_match.group(1)) if nloc_match else None

    # Remove nloc suffix if present
    name_part = re.sub(r'_nloc\d+$', '', name_part)

    # Split by underscore and filter for known transmitters
    parts = name_part.split('_')
    transmitters = [p for p in parts if p in KNOWN_TRANSMITTERS]

    return transmitters, num_locations, seed


def discover_data_directories(base_dir: Path) -> Dict[int, List[Dict]]:
    """
    Discover all valid data directories and group by TX count.

    Parameters
    ----------
    base_dir : Path
        Base directory to scan (data/processed/)

    Returns
    -------
    dict
        Dictionary mapping TX count -> list of directory info dicts
        Each info dict contains: 'path', 'transmitters', 'num_locations', 'seed', 'name'
    """
    grouped = defaultdict(list)

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue

        transmitters, num_locations, seed = parse_directory_name(item.name)

        # Skip empty or invalid directories
        if not transmitters:
            continue

        tx_count = len(transmitters)

        # Only include TX counts 1-5
        if tx_count < 1 or tx_count > 5:
            continue

        grouped[tx_count].append({
            'path': item,
            'transmitters': transmitters,
            'num_locations': num_locations,
            'seed': seed,
            'name': item.name,
        })

    # Sort each group by directory name for reproducibility
    for tx_count in grouped:
        grouped[tx_count].sort(key=lambda x: x['name'])

    return dict(grouped)


def define_sigma_noise_strategies(observed_powers_linear: np.ndarray, test_mode: bool = False) -> Dict[str, float]:
    """
    Define sigma_noise strategies to test.
    
    Parameters
    ----------
    observed_powers_linear : ndarray
        Observed powers in linear scale (mW)
    test_mode : bool
        If True, return a reduced set of strategies for quick testing
        
    Returns
    -------
    dict
        Dictionary of {strategy_name: sigma_noise_value}
    """
    min_power = np.min(observed_powers_linear)
    max_power = np.max(observed_powers_linear)
    mean_power = np.mean(observed_powers_linear)
    
    if test_mode:
        # Minimal set for quick testing
        return {
            'fixed_1e-9': 1e-9,
            'min_power': min_power,
            '5x_min': 5.0 * min_power,
        }
    
    # Full set of strategies
    # Reduced to top 5 based on parameter sweep analysis:
    # - fixed_1e-9: 28.1% times best (lowest ALE per directory)
    # - fixed_5e-9: 9.8% success rate, 14.1% times best
    # - 10x_min: 8.7% success rate, 10.9% times best
    # - 5x_min: Replaces 20x_min (User request)
    # - 0.1_mean: 11.5% success rate (highest), achieved best overall ALE (30.4m)
    strategies = {
        # Min-power based (dynamic, scaled to data)
        '10x_min': 10.0 * min_power,
        '5x_min': 5.0 * min_power,
        
        # Mean-power based (best success rate)
        '0.1_mean': 0.1 * mean_power,
    }
    
    return strategies


def save_glrt_visualization(
    info: Dict,
    map_data: Dict,
    sensor_locations: np.ndarray,
    observed_powers_dB: np.ndarray,
    tx_locations: Dict,
    output_dir: Path,
    experiment_name: str,
    rmse_filtered_support: Optional[List[int]] = None,
    save_iterations: bool = False,
):
    """
    Save GLRT iteration history visualization to files.
    
    Parameters
    ----------
    info : dict
        Reconstruction info containing solver_info with candidates_history
    map_data : dict
        Map data with shape and UTM coordinates
    sensor_locations : ndarray
        Sensor locations in pixel coordinates
    observed_powers_dB : ndarray
        Observed powers in dBm
    tx_locations : dict
        True transmitter locations
    output_dir : Path
        Directory to save visualization figures
    experiment_name : str
        Name for this experiment (used in filenames)
    rmse_filtered_support : list, optional
        List of grid indices that passed RMSE filtering. If provided, these
        are shown as magenta stars with numbers, while other candidates are
        shown as black stars (filtered out).
    save_iterations : bool
        If True, save visualization for each GLRT iteration. Default False.
    """

    if 'solver_info' not in info or 'candidates_history' not in info['solver_info']:
        return
    
    solver_info = info['solver_info']
    history = solver_info['candidates_history']
    
    if len(history) == 0:
        return
    
    # Create output directory
    vis_dir = output_dir / 'glrt_visualizations' / experiment_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    whitening_method = solver_info.get('whitening_method', 'unknown')
    
    # Get true TX coordinates
    tx_coords = np.array([tx['coordinates'] for tx in tx_locations.values()])
    
    # Only save iteration history if requested
    if save_iterations:
        for item in history:
            height, width = map_data['shape']
            score_map = np.zeros((height, width))

        
        top_indices = item['top_indices']
        top_scores = item['top_scores']
        
        # Fill scores
        rows, cols = np.unravel_index(top_indices, (height, width))
        score_map[rows, cols] = top_scores
        
        # Determine score label
        if whitening_method == 'hetero_geo_aware':
            display_score = item.get('normalized_score', item['selected_score'])
            score_label = "Corrected Score"
        else:
            display_score = item['selected_score']
            score_label = "GLRT Score"
        
        # Create figure
        fig = plt.figure(figsize=(13, 8))
        ax = fig.gca()
        
        # Plot sensors
        scatter = ax.scatter(sensor_locations[:, 0], sensor_locations[:, 1],
                             c=observed_powers_dB, s=150, edgecolor='green',
                             linewidth=2, cmap='hot', label='Monitoring Locations', zorder=6)
        
        # Plot true TX locations
        if len(tx_coords) > 0:
            ax.scatter(tx_coords[:, 0], tx_coords[:, 1],
                       marker='x', s=200, c='blue', linewidth=3,
                       label='True Transmitter Locations', zorder=10)
        
        # Plot candidates (sparse score map)
        nonzero_mask = score_map > 0
        if np.sum(nonzero_mask) > 0:
            nonzero_indices = np.argwhere(nonzero_mask)
            nonzero_row = nonzero_indices[:, 0]
            nonzero_col = nonzero_indices[:, 1]
            nonzero_values = score_map[nonzero_mask]
            
            sparse_scatter = ax.scatter(nonzero_col, nonzero_row,
                                        c=nonzero_values, s=300, marker='s',
                                        cmap='viridis', edgecolor='black', linewidth=1,
                                        label='Candidates', zorder=5)
            
            cbar = plt.colorbar(sparse_scatter, label=score_label)
            cbar.ax.tick_params(labelsize=18)
            cbar.set_label(label=score_label, size=18)
            cbar.ax.set_position([0.77, 0.1, 0.04, 0.8])
        
        # Highlight selected (All candidates in the beam for this iteration)
        # Check for new 'beam_selected_indices' key, fall back to single 'selected_index'
        beam_indices = item.get('beam_selected_indices', [item.get('selected_index')])
        beam_indices = [idx for idx in beam_indices if idx is not None]
        
        if beam_indices:
            sel_rows, sel_cols = np.unravel_index(beam_indices, map_data['shape'])
            ax.scatter(sel_cols, sel_rows, c='magenta', marker='*', s=400, label='Selected Candidate(s)', zorder=11)
        
        # UTM ticks
        UTM_lat = map_data['UTM_lat']
        UTM_long = map_data['UTM_long']
        interval = max(1, len(UTM_lat) // 5)
        tick_values = list(range(0, len(UTM_lat), interval))
        tick_labels = ['{:.1f}'.format(lat) for lat in UTM_lat[::interval]]
        plt.xticks(ticks=tick_values, labels=tick_labels, fontsize=14, rotation=0)
        
        interval = max(1, len(UTM_long) // 5)
        tick_values = list(range(0, len(UTM_long), interval))
        tick_labels = ['{:.1f}'.format(lat) for lat in UTM_long[::interval]]
        plt.yticks(ticks=[0] + tick_values[1:], labels=[""] + tick_labels[1:], fontsize=14, rotation=90)
        
        ax.set_xlim([0, map_data['shape'][1]])
        ax.set_ylim([0, map_data['shape'][0]])
        
        plt.xlabel('UTM$_E$ [m]', fontsize=18, labelpad=10)
        plt.ylabel('UTM$_N$ [m]', fontsize=18, labelpad=10)
        
        plt.title(f"GLRT Iteration {item['iteration']} ({score_label}: {display_score:.4f})", fontsize=20)
        
        scatter_cbar = plt.colorbar(scatter, label='Observed Signal [dBm]', location='left')
        scatter_cbar.ax.tick_params(labelsize=18)
        scatter_cbar.set_label(label='Observed Signal [dBm]', size=18)
        scatter_cbar.ax.set_position([0.18, 0.1, 0.04, 0.8])
        
        plt.legend(loc='upper right')
        
        # Save figure
        fig_path = vis_dir / f"iter_{item['iteration']:02d}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # === Power Density Visualization (separate figure) ===
        power_density_info = item.get('power_density_info')
        if power_density_info is not None and 'power_density' in power_density_info:
            # Create power density map visualization
            power_density = power_density_info['power_density']
            density_mask = power_density_info['density_mask']
            threshold = power_density_info['threshold']
            
            # Reshape to 2D
            density_map = power_density.reshape((height, width))
            mask_map = density_mask.reshape((height, width))
            
            # Create masked density (areas below threshold shown as NaN)
            density_thresholded = density_map.copy()
            density_thresholded[mask_map] = np.nan  # Mask out low-density areas
            
            fig = plt.figure(figsize=(14, 8))
            ax = fig.gca()
            
            # Plot the full density map with transparency
            im_full = ax.imshow(density_map, origin='lower', cmap='Blues', alpha=0.3,
                               vmin=0, vmax=1, aspect='auto')
            
            # Overlay the thresholded (valid) density regions
            im_valid = ax.imshow(density_thresholded, origin='lower', cmap='Reds',
                                vmin=threshold, vmax=1, aspect='auto')
            
            # Plot sensors with power indication
            scatter = ax.scatter(sensor_locations[:, 0], sensor_locations[:, 1],
                                c=observed_powers_dB, s=200, edgecolor='black',
                                linewidth=2, cmap='hot', label='Sensors (by power)', zorder=8)
            
            # Plot true TX locations
            if len(tx_coords) > 0:
                ax.scatter(tx_coords[:, 0], tx_coords[:, 1],
                          marker='x', s=250, c='blue', linewidth=4,
                          label='True TX Locations', zorder=10)
            
            # Highlight selected candidate
            sel_row, sel_col = np.unravel_index(item['selected_index'], (height, width))
            ax.scatter([sel_col], [sel_row], c='magenta', marker='*', s=500,
                      edgecolor='white', linewidth=2, label='Selected', zorder=11)
            
            # Colorbar for density
            cbar = plt.colorbar(im_valid, ax=ax, label='Power Density (thresholded)', shrink=0.8)
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('Power Density (above threshold)', size=14)
            
            # UTM ticks
            UTM_lat = map_data['UTM_lat']
            UTM_long = map_data['UTM_long']
            interval = max(1, len(UTM_lat) // 5)
            tick_values = list(range(0, len(UTM_lat), interval))
            tick_labels = ['{:.1f}'.format(lat) for lat in UTM_lat[::interval]]
            plt.xticks(ticks=tick_values, labels=tick_labels, fontsize=12, rotation=0)
            
            interval = max(1, len(UTM_long) // 5)
            tick_values = list(range(0, len(UTM_long), interval))
            tick_labels = ['{:.1f}'.format(lat) for lat in UTM_long[::interval]]
            plt.yticks(ticks=[0] + tick_values[1:], labels=[""] + tick_labels[1:], fontsize=12, rotation=90)
            
            ax.set_xlim([0, width])
            ax.set_ylim([0, height])
            
            plt.xlabel('UTM$_E$ [m]', fontsize=16, labelpad=10)
            plt.ylabel('UTM$_N$ [m]', fontsize=16, labelpad=10)
            
            n_masked = power_density_info['n_masked']
            n_total = len(power_density)
            pct_valid = (n_total - n_masked) / n_total * 100
            
            plt.title(f"Power Density Map - Iter {item['iteration']} | "
                     f"Threshold: {threshold:.0%} | Valid: {pct_valid:.1f}%", fontsize=16)
            
            plt.legend(loc='upper right', fontsize=11)
            
            # Save figure
            fig_path = vis_dir / f"power_density_iter_{item['iteration']:02d}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    # === Final Refined Selection Visualization ===
    if 'final_support' in solver_info and len(solver_info['final_support']) > 0:
        final_indices = solver_info['final_support']
        
        # Determine which candidates are kept vs filtered
        # Use rmse_filtered_support directly to preserve RMSE-sorted order (lowest to highest)
        if rmse_filtered_support is not None:
            kept_set = set(rmse_filtered_support)
            kept_indices = list(rmse_filtered_support)  # Already sorted by RMSE (lowest first)
            filtered_indices = [idx for idx in final_indices if idx not in kept_set]
        else:
            kept_indices = list(final_indices)
            filtered_indices = []

        
        fig = plt.figure(figsize=(13, 8))
        ax = fig.gca()
        
        # Plot sensors
        ax.scatter(sensor_locations[:, 0], sensor_locations[:, 1],
                   c=observed_powers_dB, s=150, edgecolor='green',
                   linewidth=2, cmap='hot', label='Monitoring Locations', zorder=6)
                   
        # Plot true TX
        if len(tx_coords) > 0:
            ax.scatter(tx_coords[:, 0], tx_coords[:, 1],
                       marker='x', s=200, c='blue', linewidth=3,
                       label='True Transmitter Locations', zorder=10)
        
        # Plot filtered candidates (black stars, no numbers)
        if len(filtered_indices) > 0:
            filt_rows, filt_cols = np.unravel_index(filtered_indices, map_data['shape'])
            ax.scatter(filt_cols, filt_rows, c='black', marker='*', s=400, 
                       edgecolor='white', linewidth=1.5,
                       label=f'Filtered by RMSE ({len(filtered_indices)})', zorder=10)
        
        # Plot kept candidates (magenta stars with numbers)
        if len(kept_indices) > 0:
            kept_rows, kept_cols = np.unravel_index(kept_indices, map_data['shape'])
            ax.scatter(kept_cols, kept_rows, c='magenta', marker='*', s=500, 
                       edgecolor='white', linewidth=1.5,
                       label=f'Kept ({len(kept_indices)})', zorder=11)
            
            # Add numbered annotations to kept candidates for association with power analysis plots
            for idx, (col, row) in enumerate(zip(kept_cols, kept_rows)):
                ax.annotate(f'{idx+1}', (col, row), textcoords='offset points', 
                           xytext=(8, 8), fontsize=12, fontweight='bold',
                           color='black', ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                    edgecolor='magenta', alpha=0.9),
                           zorder=12)
        
        # Formatting
        UTM_lat = map_data['UTM_lat']
        UTM_long = map_data['UTM_long']
        interval = max(1, len(UTM_lat) // 5)
        tick_values = list(range(0, len(UTM_lat), interval))
        tick_labels = ['{:.1f}'.format(lat) for lat in UTM_lat[::interval]]
        plt.xticks(ticks=tick_values, labels=tick_labels, fontsize=14, rotation=0)
        
        interval = max(1, len(UTM_long) // 5)
        tick_values = list(range(0, len(UTM_long), interval))
        tick_labels = ['{:.1f}'.format(lat) for lat in UTM_long[::interval]]
        plt.yticks(ticks=[0] + tick_values[1:], labels=[""] + tick_labels[1:], fontsize=14, rotation=90)
        
        ax.set_xlim([0, map_data['shape'][1]])
        ax.set_ylim([0, map_data['shape'][0]])
        
        # Title shows total, kept, and filtered counts
        title = f"Final Selection: {len(kept_indices)} Kept"
        if len(filtered_indices) > 0:
            title += f", {len(filtered_indices)} Filtered (RMSE > 20 dB)"
        plt.title(title, fontsize=18)
        plt.legend(loc='upper right')
        
        # Save figure
        fig_path = vis_dir / "final_selection.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)











def run_single_experiment(
    data_info: Dict,
    config: Dict,
    map_data: Dict,
    all_tx_locations: Dict,
    sigma_noise: float,
    selection_method: str,
    use_power_filtering: bool,
    whitening_method: str,
    feature_rho: Optional[List[float]],
    whitening_config_name: str,
    power_density_threshold: float = 0.3,
    strategy_name: str = '',
    model_type: str = 'tirem',
    recon_model_type: str = 'tirem',
    eta: float = 0.01,
    output_dir: Optional[Path] = None,
    save_visualization: bool = False,
    verbose: bool = False,
    beam_width: int = 1,
    max_pool_size: int = 50,
    use_edf_penalty: bool = False,
    edf_threshold: float = 1.5,
    use_robust_scoring: bool = False,
    robust_threshold: float = 6.0,
    save_iterations: bool = False,

    pooling_lambda: float = 0.01,
    dedupe_distance_m: float = 60.0,

    # Combinatorial selection parameters
    combo_min_distance_m: float = 100.0,
    combo_max_size: int = 5,
    combo_max_candidates: int = 10,
    combo_bic_weight: float = 0.2,
    combo_max_power_diff_dB: float = 20.0,
    combo_sensor_proximity_threshold_m: float = 100.0,
    combo_sensor_proximity_penalty: float = 10.0,
) -> Optional[Dict]:
    """
    Run a single reconstruction experiment.

    
    Returns
    -------
    dict or None
        Result dictionary with metrics, or None if failed
    """
    try:
        transmitters = data_info['transmitters']
        tx_underscore = "_".join(transmitters)
        data_dir = data_info['path']
        seed = data_info['seed']
        num_locations = data_info.get('num_locations')

        # Build config path (matching the directory naming convention)
        # Format: {transmitters}_nloc{N}_seed_{S} or legacy format without nloc
        config_id = tx_underscore
        if num_locations is not None:
            config_id = f"{config_id}_nloc{num_locations}"
        if seed is not None:
            config_id = f"{config_id}_seed_{seed}"
        config_path = f'config/monitoring_locations_{config_id}.yaml'
        
        # Check if config exists
        if not Path(config_path).exists():
            if verbose:
                print(f"    Config not found: {config_path}")
            return None
        
        # Load monitoring locations
        locations_config = load_monitoring_locations(
            config_path=config_path,
            map_data=map_data
        )
        sensor_locations = get_sensor_locations_array(locations_config)
        
        # Load power measurements
        powers_file = data_dir / f"{tx_underscore}_avg_powers.npy"
        if not powers_file.exists():
            if verbose:
                print(f"    Powers file not found: {powers_file}")
            return None
            
        observed_powers_dB = np.load(powers_file)
        observed_powers_linear = dbm_to_linear(observed_powers_dB)

        # Load observed standard deviations (for hetero_diag_obs whitening)
        stds_file = data_dir / f"{tx_underscore}_std_powers.npy"
        observed_stds_dB = np.load(stds_file) if stds_file.exists() else None

        # Get true transmitter locations
        tx_locations = {name: all_tx_locations[name] for name in transmitters if name in all_tx_locations}
        true_locs_pixels = np.array([tx['coordinates'] for tx in tx_locations.values()])
        
        start_time = time.time()
        
        # Determine model config paths (localization and reconstruction)
        if model_type == 'tirem':
            model_config_path = 'config/tirem_parameters.yaml'
        elif model_type == 'raytracing':
            model_config_path = 'config/sionna_parameters.yaml'
        else:
            model_config_path = None

        if recon_model_type == 'tirem':
            recon_model_config_path = 'config/tirem_parameters.yaml'
        elif recon_model_type == 'raytracing':
            recon_model_config_path = 'config/sionna_parameters.yaml'
        else:
            recon_model_config_path = None

        # Run reconstruction
        reconstruction_kwargs = {
            'sensor_locations': sensor_locations,
            'observed_powers_dBm': observed_powers_dB,
            'input_is_linear': False,
            'solve_in_linear_domain': True,
            'map_shape': map_data['shape'],
            'scale': config['spatial']['proxel_size'],
            'np_exponent': config['localization']['path_loss_exponent'],
            'lambda_reg': pooling_lambda,
            'norm_exponent': 0,
            'whitening_method': whitening_method,
            'sigma_noise': sigma_noise,
            'eta': eta,
            'solver': 'glrt',
            'selection_method': selection_method,
            'use_power_filtering': use_power_filtering,
            'power_density_threshold': power_density_threshold,
            'cluster_max_candidates': 30,
            'glrt_max_iter': max(10, len(transmitters) + 5),
            'glrt_threshold': 4.0,
            'glrt_max_iter': max(10, len(transmitters) + 5),
            'glrt_threshold': 4.0,
            'dedupe_distance_m': dedupe_distance_m,
            'return_linear_scale': False,
            'verbose': False,
            'model_type': model_type,
            'model_config_path': model_config_path,
            'n_jobs': -1,
            'beam_width': beam_width,
            'max_pool_size': max_pool_size,
            'max_pool_size': max_pool_size,
            'pool_refinement': True, # Always enable refinement for sweep
            'use_edf_penalty': use_edf_penalty,
            'edf_threshold': edf_threshold,
            'use_robust_scoring': use_robust_scoring,
            'robust_threshold': robust_threshold,
            # Observed std for hetero_diag_obs whitening (None for other methods)
            'observed_stds_dB': observed_stds_dB,
        }

        # Add feature_rho only for hetero_geo_aware
        if feature_rho is not None:
            reconstruction_kwargs['feature_rho'] = feature_rho
        
        tx_map, info = joint_sparse_reconstruction(**reconstruction_kwargs)

        elapsed = time.time() - start_time

        # Build experiment name for any file outputs
        pf_suffix = ''
        if use_power_filtering:
            pf_suffix = f'_pf_thresh{power_density_threshold}'
        experiment_name = f"{data_info['name']}_{strategy_name}_{whitening_config_name}_{selection_method}{pf_suffix}"

        # Always do RMSE-based candidate filtering and combinatorial selection (needed for BIC metrics)
        filtered_support = None
        if 'solver_info' in info and 'final_support' in info['solver_info']:
            final_support = info['solver_info']['final_support']

            if len(final_support) > 0:
                scale = config['spatial']['proxel_size']
                np_exponent = config['localization']['path_loss_exponent']

                # Step 1: Compute RMSE for all candidates (with bias correction)
                rmse_values, mae_values, max_error_values, optimal_tx_powers, slope_values = compute_candidate_power_rmse(
                    final_support=final_support,
                    tx_map=tx_map,
                    map_shape=map_data['shape'],
                    sensor_locations=sensor_locations,
                    observed_powers_dB=observed_powers_dB,
                    scale=scale,
                    np_exponent=np_exponent,
                )

                # Step 2: Filter candidates by RMSE threshold
                filtered_support, filtered_rmse, cutoff_rmse = filter_candidates_by_rmse(
                    final_support=final_support,
                    rmse_values=rmse_values,
                    max_error_values=max_error_values,
                    slope_values=slope_values,
                    output_dir=output_dir,
                    experiment_name=experiment_name,
                    min_candidates=1,
                    rmse_threshold=20.0,
                    max_error_threshold=38.0,
                    save_plot=save_visualization,  # Only save plot if visualizing
                )

                # Store filtered support for metrics computation
                info['solver_info']['rmse_filtered_support'] = filtered_support
                info['solver_info']['rmse_cutoff'] = cutoff_rmse
                info['solver_info']['n_filtered_by_rmse'] = len(final_support) - len(filtered_support)

                # Step 3: Generate power analysis plots only if visualizations requested
                if save_visualization and output_dir is not None:
                    save_candidate_power_analysis(
                        info=info,
                        tx_map=tx_map,
                        map_data=map_data,
                        sensor_locations=sensor_locations,
                        observed_powers_dB=observed_powers_dB,
                        tx_locations=tx_locations,
                        output_dir=output_dir,
                        experiment_name=experiment_name,
                        scale=scale,
                        np_exponent=np_exponent,
                        candidate_indices=filtered_support,
                    )

                # Step 4: Run combinatorial TX selection optimization (always, for BIC)
                # Find optimal combination of TXs that best explains observations
                # Only generate plots if save_visualization is True
                combination_result = run_combinatorial_selection(
                    info=info,
                    tx_map=tx_map,
                    map_data=map_data,
                    sensor_locations=sensor_locations,
                    observed_powers_dB=observed_powers_dB,
                    tx_locations=tx_locations,
                    output_dir=output_dir,
                    experiment_name=experiment_name,
                    filtered_support=filtered_support,
                    scale=scale,
                    np_exponent=np_exponent,
                    min_distance_m=combo_min_distance_m,
                    max_combination_size=combo_max_size,
                    max_candidates_to_consider=combo_max_candidates,
                    bic_penalty_weight=combo_bic_weight,
                    max_power_diff_dB=combo_max_power_diff_dB,
                    sensor_proximity_threshold_m=combo_sensor_proximity_threshold_m,
                    sensor_proximity_penalty=combo_sensor_proximity_penalty,
                    max_plots=10,
                    save_plots=save_visualization,  # Only generate plots if visualizing
                    verbose=False,
                )

                # Store combination result in info
                info['solver_info']['combination_result'] = combination_result
                info['solver_info']['optimal_combination'] = combination_result.get('best_combination', [])
                info['solver_info']['optimal_powers_dBm'] = combination_result.get('best_powers_dBm', np.array([]))
                info['solver_info']['combination_rmse'] = combination_result.get('best_rmse', np.inf)
                info['solver_info']['combination_bic'] = combination_result.get('best_bic', np.inf)

                # Step 5: Recompute optimal TX powers using the reconstruction
                # propagation model instead of the log-distance approximation
                # used during candidate selection.  The TX locations are kept
                # fixed; only powers are re-optimized so that reconstruction
                # uses power estimates consistent with the reconstruction
                # propagation model.
                optimal_combo = info['solver_info']['optimal_combination']
                if len(optimal_combo) > 0:
                    # Get the propagation matrix for reconstruction
                    if recon_model_type == model_type:
                        # Same model — reuse A_model from localization
                        A_recon = info.get('A_model')
                    else:
                        # Different model — compute propagation matrix
                        # for reconstruction model with sensor locations
                        from src.sparse_reconstruction.propagation_matrix import compute_propagation_matrix as _compute_prop_matrix
                        A_recon = _compute_prop_matrix(
                            sensor_locations=sensor_locations,
                            map_shape=map_data['shape'],
                            scale=scale,
                            model_type=recon_model_type,
                            config_path=recon_model_config_path,
                            np_exponent=np_exponent,
                            n_jobs=-1,
                            verbose=False,
                        )

                    if A_recon is not None:
                        recomp_powers, recomp_rmse, recomp_mae, recomp_max_err, recomp_total = \
                            recompute_powers_with_propagation_model(
                                combo_grid_indices=optimal_combo,
                                A_model=A_recon,
                                observed_powers_dB=observed_powers_dB,
                                max_power_diff_dB=combo_max_power_diff_dB,
                            )
                        info['solver_info']['optimal_powers_dBm'] = recomp_powers
                        info['solver_info']['combination_rmse'] = recomp_rmse

                # Step 5.5: Per-TX exponent refit (log_distance reconstruction only)
                # After localization, fit a per-TX path loss exponent from
                # observed sensor data, rebuild path gains, and re-optimize
                # powers.  This improves reconstruction when different TXs
                # experience different propagation conditions.
                if recon_model_type == 'log_distance' and len(optimal_combo) > 0:
                    per_tx_exp, _refit_gains, refit_powers, refit_rmse, \
                        refit_mae, refit_max_err, refit_total = \
                        refit_with_per_tx_exponents(
                            combo_grid_indices=optimal_combo,
                            map_shape=map_data['shape'],
                            sensor_locations=sensor_locations,
                            observed_powers_dB=observed_powers_dB,
                            current_powers_dBm=info['solver_info']['optimal_powers_dBm'],
                            scale=scale,
                            np_exponent_global=np_exponent,
                            max_power_diff_dB=combo_max_power_diff_dB,
                        )
                    info['solver_info']['per_tx_exponents'] = per_tx_exp.tolist()
                    info['solver_info']['optimal_powers_dBm'] = refit_powers
                    info['solver_info']['combination_rmse'] = refit_rmse

        # Save GLRT visualization if requested
        if save_visualization and output_dir is not None:
            save_glrt_visualization(
                info=info,
                map_data=map_data,
                sensor_locations=sensor_locations,
                observed_powers_dB=observed_powers_dB,
                tx_locations=tx_locations,
                output_dir=output_dir,
                experiment_name=experiment_name,
                rmse_filtered_support=filtered_support,
                save_iterations=save_iterations,
            )



        # Extract estimated locations
        if 'solver_info' in info and 'support' in info['solver_info']:
            support_indices = info['solver_info']['support']
            height, width = map_data['shape']
            
            n_est_raw = len(support_indices)
            
            valid_indices = []
            for idx in support_indices:
                r, c = idx // width, idx % width
                power_dbm = tx_map[r, c]
                if power_dbm > -190:
                    valid_indices.append(idx)
            
            n_est_diff = n_est_raw - len(valid_indices)
            
            est_rows = [idx // width for idx in valid_indices]
            est_cols = [idx % width for idx in valid_indices]
            est_locs_pixels = np.column_stack((est_cols, est_rows)) if valid_indices else np.empty((0, 2))
        else:
            from src.evaluation.metrics import extract_locations_from_map
            est_locs_pixels = extract_locations_from_map(tx_map, threshold=1e-10)
            n_est_raw = len(est_locs_pixels)
            n_est_diff = 0
        
        # Compute metrics
        metrics = compute_localization_metrics(
            true_locations=true_locs_pixels,
            estimated_locations=est_locs_pixels,
            scale=config['spatial']['proxel_size'],
            tolerance=200.0
        )

        # Compute metrics for optimal combination (if available)
        combo_metrics = {'combo_ale': np.nan, 'combo_tp': 0, 'combo_fp': 0, 'combo_fn': 0, 'combo_pd': 0.0, 'combo_precision': 0.0}
        if 'solver_info' in info and 'optimal_combination' in info['solver_info']:
            optimal_combo = info['solver_info']['optimal_combination']
            if len(optimal_combo) > 0:
                height, width = map_data['shape']
                combo_rows = [idx // width for idx in optimal_combo]
                combo_cols = [idx % width for idx in optimal_combo]
                combo_locs_pixels = np.column_stack((combo_cols, combo_rows))

                combo_metrics_raw = compute_localization_metrics(
                    true_locations=true_locs_pixels,
                    estimated_locations=combo_locs_pixels,
                    scale=config['spatial']['proxel_size'],
                    tolerance=200.0
                )
                combo_metrics = {
                    'combo_ale': combo_metrics_raw['ale'],
                    'combo_tp': combo_metrics_raw['tp'],
                    'combo_fp': combo_metrics_raw['fp'],
                    'combo_fn': combo_metrics_raw['fn'],
                    'combo_pd': combo_metrics_raw['pd'],
                    'combo_precision': combo_metrics_raw['precision'],
                }

        # Extract GLRT score history from solver info
        glrt_score_history = []
        glrt_n_iterations = 0
        glrt_initial_score = 0.0
        glrt_final_score = 0.0
        glrt_score_reduction = 0.0
        
        if 'solver_info' in info and 'candidates_history' in info['solver_info']:
            candidates_history = info['solver_info']['candidates_history']
            whitening_method = info['solver_info'].get('whitening_method', 'unknown')
            
            if len(candidates_history) > 0:
                glrt_n_iterations = len(candidates_history)
                
                # Extract scores - use normalized_score which is geo_aware_score for hetero_geo_aware
                # and R^2 normalized score for other methods
                for item in candidates_history:
                    # Use corrected score for hetero_geo_aware, raw score for others
                    if whitening_method == 'hetero_geo_aware':
                        score = item.get('normalized_score', item.get('selected_score', 0.0))
                    else:
                        score = item.get('selected_score', 0.0)
                    glrt_score_history.append(float(score))
                
                if glrt_score_history:
                    glrt_initial_score = glrt_score_history[0]
                    glrt_final_score = glrt_score_history[-1]
                    glrt_score_reduction = glrt_initial_score - glrt_final_score
            
            # Find closest iteration for each true transmitter
            height, width = map_data['shape']
            best_match_iterations = []
            
            if len(candidates_history) > 0 and len(true_locs_pixels) > 0:
                # Get locations of all added candidates in order
                candidate_locs = []
                for item in candidates_history:
                    idx = item['selected_index']
                    r, c = idx // width, idx % width
                    candidate_locs.append([c, r]) # x, y for distance calc (or r, c? true_locs_pixels are [col, row])
                    # Note: true_locs_pixels is usually [col, row] (x, y)
                    # Let's verify: In run_single_experiment: true_locs_pixels = np.array([tx['coordinates'] for tx ...])
                    # coordinates in load_transmitter_locations are [col_idx, row_idx]
                
                candidate_locs = np.array(candidate_locs)
                
                # For each true transmitter, find index of closest candidate
                for true_tx in true_locs_pixels:
                    # true_tx is [col, row]
                    # candidate_locs is [[c, r], [c, r]...]
                    
                    dists = np.sqrt(np.sum((candidate_locs - true_tx)**2, axis=1))
                    best_iter = np.argmin(dists) + 1 # 1-based iteration index
                    best_match_iterations.append(int(best_iter))
        
        # Convert history to JSON string for CSV storage
        import json
        glrt_score_history_str = json.dumps([round(s, 6) for s in glrt_score_history])

        # === RECONSTRUCTION ERROR COMPUTATION ===
        # Compute how well the estimated TX locations/powers predict RSS at validation points
        # Pass observation data for noise floor computation (clamping predictions)
        recon_metrics = compute_reconstruction_error(
            combo_indices=info.get('solver_info', {}).get('optimal_combination', []),
            combo_powers_dBm=[float(p) for p in info.get('solver_info', {}).get('optimal_powers_dBm', [])],
            map_data=map_data,
            transmitters=transmitters,
            project_root=Path(__file__).parent.parent,
            observed_powers_dB=observed_powers_dB,
            num_locations=num_locations,
            model_type=recon_model_type,
            model_config_path=recon_model_config_path,
            scale=config['spatial']['proxel_size'],
            auto_generate=False,  # Don't auto-generate during sweep
            verbose=False,
            output_dir=output_dir,
            experiment_name=experiment_name,
            save_plot=save_visualization,  # Generate validation plots when visualizations enabled
            true_tx_locations=tx_locations,  # For spatial plot visualization
            per_tx_exponents=info.get('solver_info', {}).get('per_tx_exponents'),
        )

        return {
            'ale': metrics['ale'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'pd': metrics['pd'],
            'precision': metrics['precision'],
            'f1_score': metrics['f1_score'],
            'n_estimated': metrics['n_est'],
            'n_est_raw': n_est_raw,
            'n_est_diff': n_est_diff,
            'count_error': metrics['n_est'] - len(true_locs_pixels),
            'runtime_s': elapsed,
            'obs_min_dbm': np.min(observed_powers_dB),
            'obs_mean_dbm': np.mean(observed_powers_dB),
            'obs_max_dbm': np.max(observed_powers_dB),
            'glrt_n_iterations': glrt_n_iterations,
            'glrt_initial_score': glrt_initial_score,
            'glrt_final_score': glrt_final_score,
            'glrt_score_reduction': glrt_score_reduction,
            'glrt_score_history': glrt_score_history_str,
            'best_match_iterations': json.dumps(best_match_iterations), # Store as JSON list
            # Combinatorial selection metrics
            'combo_n_tx': len(info.get('solver_info', {}).get('optimal_combination', [])),
            'combo_rmse': info.get('solver_info', {}).get('combination_rmse', np.nan),
            'combo_bic': info.get('solver_info', {}).get('combination_bic', np.nan),
            'combo_indices': json.dumps(info.get('solver_info', {}).get('optimal_combination', [])),
            'combo_powers_dBm': json.dumps(
                [float(p) for p in info.get('solver_info', {}).get('optimal_powers_dBm', [])]
            ),
            'per_tx_exponents': json.dumps(info.get('solver_info', {}).get('per_tx_exponents', [])),
            # Combinatorial selection localization metrics
            'combo_ale': combo_metrics['combo_ale'],
            'combo_tp': combo_metrics['combo_tp'],
            'combo_fp': combo_metrics['combo_fp'],
            'combo_fn': combo_metrics['combo_fn'],
            'combo_pd': combo_metrics['combo_pd'],
            'combo_precision': combo_metrics['combo_precision'],
            'combo_count_error': abs(len(info.get('solver_info', {}).get('optimal_combination', [])) - len(true_locs_pixels)),
            # Reconstruction error metrics
            'recon_rmse': recon_metrics['recon_rmse'],
            'recon_mae': recon_metrics['recon_mae'],
            'recon_bias': recon_metrics['recon_bias'],
            'recon_max_error': recon_metrics['recon_max_error'],
            'recon_n_val_points': recon_metrics['recon_n_val_points'],
            'recon_noise_floor': recon_metrics['recon_noise_floor'],
            'recon_status': recon_metrics['recon_status'],
        }

    except Exception as e:
        if verbose:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
        return None

def _worker_init():
    """Initialize worker process - disable numpy threading to prevent deadlocks."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


def process_single_directory(args: Tuple) -> Tuple[List[Dict], str]:
    """
    Process a single data directory with all parameter combinations.
    
    This is a top-level function for pickling compatibility with multiprocessing.
    All Path objects are converted to strings before being passed here.
    
    Parameters
    ----------
    args : tuple
        (data_info_serializable, config, map_data, all_tx_locations, output_dir_str,
         test_mode, model_type, recon_model_type, eta, save_visualizations,
         whitening_configs, selection_methods, power_thresholds, beam_width,
         max_pool_size, use_edf_penalty, edf_threshold, ...)

    Returns
    -------
    tuple
        (list of result dictionaries, skip_reason or None)
    """
    (data_info_serializable, config, map_data, all_tx_locations, output_dir_str,
     test_mode, model_type, recon_model_type, eta, save_visualizations, whitening_configs,
     selection_configs, power_thresholds, beam_width, max_pool_size,
     use_edf_penalty, edf_threshold, use_robust_scoring, robust_threshold, save_iterations,
     pooling_lambda, dedupe_distance_m,
     combo_min_distance_m, combo_max_size, combo_max_candidates, combo_bic_weight, combo_max_power_diff_dB,
     combo_sensor_proximity_threshold_m, combo_sensor_proximity_penalty) = args
    
    # Reconstruct data_info with Path object
    data_info = data_info_serializable.copy()
    data_info['path'] = Path(data_info['path_str'])
    output_dir = Path(output_dir_str)
    
    results = []
    dir_name = data_info['name']
    transmitters = data_info['transmitters']
    tx_underscore = "_".join(transmitters)
    tx_count = len(transmitters)
    
    # Load power data for strategy definition
    powers_file = data_info['path'] / f"{tx_underscore}_avg_powers.npy"
    if not powers_file.exists():
        return results, "no powers file"
    
    observed_powers_dB = np.load(powers_file)
    observed_powers_linear = dbm_to_linear(observed_powers_dB)
    
    # Check if TIREM cache exists (needed if either model uses TIREM)
    if model_type == 'tirem' or recon_model_type == 'tirem':
        seed = data_info['seed']
        num_locations = data_info.get('num_locations')
        # Build config path (matching the directory naming convention)
        config_id = tx_underscore
        if num_locations is not None:
            config_id = f"{config_id}_nloc{num_locations}"
        if seed is not None:
            config_id = f"{config_id}_seed_{seed}"
        config_path = f'config/monitoring_locations_{config_id}.yaml'

        if not Path(config_path).exists():
            return results, "no config file"

        locations_config = load_monitoring_locations(
            config_path=config_path,
            map_data=map_data
        )
        sensor_locations = get_sensor_locations_array(locations_config)

        features_cached, prop_cached = check_tirem_cache_exists(
            sensor_locations=sensor_locations,
            map_shape=map_data['shape'],
            scale=config['spatial']['proxel_size'],
            tirem_config_path='config/tirem_parameters.yaml',
        )

        # Features cache only needed for localization with hetero_geo_aware whitening
        needs_features = ('hetero_geo_aware' in whitening_configs) and (model_type == 'tirem')
        if not prop_cached or (needs_features and not features_cached):
            return results, f"no TIREM cache (features={features_cached}, prop={prop_cached}, needs_features={needs_features})"
    
    # Define strategies based on this dataset's observations
    strategies = define_sigma_noise_strategies(observed_powers_linear, test_mode=test_mode)
    
    attempted = 0
    failed = 0
    
    try:
        for strategy_name, sigma_noise in strategies.items():
            for sel_method, use_pf in selection_configs:
                # Determine thresholds to test for this selection config
                # If PF is enabled, test all thresholds. If disabled, test only one (value doesn't matter).
                if use_pf:
                    thresholds_to_test = power_thresholds
                else:
                    thresholds_to_test = [0.0] # Dummy value when PF is disabled
                
                for threshold in thresholds_to_test:
                    for config_name, (whitening_method, feature_rho) in whitening_configs.items():
                        attempted += 1
                        try:
                            # First pass: run WITHOUT visualizations to find best BIC
                            result = run_single_experiment(
                                data_info=data_info,
                                config=config,
                                map_data=map_data,
                                all_tx_locations=all_tx_locations,
                                sigma_noise=sigma_noise,
                                selection_method=sel_method,
                                use_power_filtering=use_pf,
                                whitening_method=whitening_method,
                                feature_rho=feature_rho,
                                whitening_config_name=config_name,
                                strategy_name=strategy_name,
                                model_type=model_type,
                                recon_model_type=recon_model_type,
                                eta=eta,
                                output_dir=output_dir,
                                save_visualization=False,  # Always False in first pass
                                verbose=False,
                                power_density_threshold=threshold,

                                beam_width=beam_width,
                                max_pool_size=max_pool_size,
                                use_edf_penalty=use_edf_penalty,
                                edf_threshold=edf_threshold,
                                use_robust_scoring=use_robust_scoring,
                                robust_threshold=robust_threshold,

                                save_iterations=save_iterations,

                                pooling_lambda=pooling_lambda,
                                dedupe_distance_m=dedupe_distance_m,

                                # Combinatorial selection parameters
                                combo_min_distance_m=combo_min_distance_m,
                                combo_max_size=combo_max_size,
                                combo_max_candidates=combo_max_candidates,
                                combo_bic_weight=combo_bic_weight,
                                combo_max_power_diff_dB=combo_max_power_diff_dB,
                                combo_sensor_proximity_threshold_m=combo_sensor_proximity_threshold_m,
                                combo_sensor_proximity_penalty=combo_sensor_proximity_penalty,
                            )


                            if result is not None:
                                result.update({
                                    'dir_name': dir_name,
                                    'tx_count': tx_count,
                                    'transmitters': ','.join(transmitters),
                                    'seed': data_info['seed'],
                                    'strategy': strategy_name,
                                    'selection_method': sel_method,
                                    'power_filtering': use_pf,
                                    'power_threshold': threshold if use_pf else float('nan'),
                                    'whitening_config': config_name,
                                    'sigma_noise': sigma_noise,
                                    'whitening_config': config_name,
                                    'sigma_noise': sigma_noise,
                                    'sigma_noise_dB': 10 * np.log10(sigma_noise) if sigma_noise > 0 else -np.inf,
                                    'use_edf': use_edf_penalty,
                                    'edf_thresh': edf_threshold if use_edf_penalty else float('nan'),
                                    'use_robust': use_robust_scoring,
                                    'robust_thresh': robust_threshold if use_robust_scoring else float('nan'),
                                    'robust_thresh': robust_threshold if use_robust_scoring else float('nan'),
                                    'pooling_lambda': pooling_lambda,
                                    'dedupe_dist': dedupe_distance_m,
                                    # Store whitening params for potential re-run
                                    '_whitening_method': whitening_method,
                                    '_feature_rho': feature_rho,
                                })
                                results.append(result)
                            else:
                                failed += 1
                        except Exception as inner_exc:
                            failed += 1
                        # Continue to next experiment
                        
                        # --- Intermediate Logging (every 5 experiments) ---
                        if attempted % 5 == 0:
                            n_curr = len(results)
                            best_curr = min(r['ale'] for r in results) if results else float('inf')
                            msg = f"[{dir_name}] Progress: {n_curr}/{attempted} exps | Best ALE: {best_curr:.2f}m"
                            print(f"  {msg}", flush=True)
                            try:
                                log_path = output_dir / "sweep_progress.log"
                                with open(log_path, "a", encoding="utf-8") as f:
                                    timestamp = datetime.now().strftime("%H:%M:%S")
                                    f.write(f"[{timestamp}] {msg}\n")
                            except:
                                pass
    except Exception as e:
        import traceback
        return results, f"EXCEPTION after {attempted} attempts ({failed} failed): {e}\n{traceback.format_exc()}"
    
    if failed > 0 and len(results) == 0:
        return results, f"all {attempted} experiments failed"

    # --- Logging Results (before visualization re-run) ---
    n_completed = len(results)
    best_ale = float('inf')
    max_prec = 0.0
    max_pd = 0.0
    best_count_err = 0.0

    if n_completed > 0:
        best_ale = min(r['ale'] for r in results)
        max_prec = max(r['precision'] for r in results)
        max_pd = max(r['pd'] for r in results)

        # for count error, we want the one with min absolute value
        errors = [r['count_error'] for r in results]
        best_count_err = min(errors, key=abs)

    # Format message
    # [Dir Name] Comp: X | Best ALE: Y.YYm | Max Pd: P.PP | Max Prec: Z.ZZ | Best C.Err: W.WW
    msg = (f"[{dir_name}] Comp: {n_completed}/{attempted} | "
           f"Best ALE: {best_ale:.1f}m | "
           f"Max Pd: {max_pd:.2f} | "
           f"Max Prec: {max_prec:.2f} | "
           f"Best C.Err: {best_count_err:.1f}")

    if n_completed > 0:
        # Identify best result
        best_result = min(results, key=lambda r: r['ale'])
        best_params = f"{best_result['strategy']}, {best_result['selection_method']}, {best_result['whitening_config']}"
        if best_result['power_filtering']:
             best_params += f", PF={best_result['power_threshold']}"
        msg += f" | Best Params: [{best_params}]"

    if failed > 0:
        msg += f" | Failed: {failed}"

    # 1. Print to stdout (will be captured by joblib and shown in main process)
    print(f"  {msg}", flush=True)

    # 2. Append to log file
    try:
        log_path = output_dir / "sweep_progress.log"
        # Use append mode, and simple locking by OS (hope for the best with concurrency)
        # For true safety we'd need a lock, but for simple logging this is usually fine
        with open(log_path, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass # Don't fail the worker just because logging failed

    # --- Re-run best BIC experiment with visualizations (if requested) ---
    if not save_visualizations:
        print(f"  [{dir_name}] Skipping visualizations (disabled)", flush=True)
    elif len(results) == 0:
        print(f"  [{dir_name}] Skipping visualizations (no results)", flush=True)
    if save_visualizations and len(results) > 0:
        # Find result with lowest combo_bic
        valid_bic_results = [r for r in results if 'combo_bic' in r and not np.isnan(r.get('combo_bic', np.nan))]
        if not valid_bic_results:
            # Debug: check what BIC values we have
            bic_values = [r.get('combo_bic', 'MISSING') for r in results[:3]]  # First 3
            print(f"  [{dir_name}] Warning: No valid BIC results for visualization. Sample BIC values: {bic_values}", flush=True)
        if valid_bic_results:
            best_bic_result = min(valid_bic_results, key=lambda r: r['combo_bic'])
            best_bic_strategy = best_bic_result['strategy']
            best_bic_whitening = best_bic_result['whitening_config']
            best_bic_sel_method = best_bic_result['selection_method']
            best_bic_pf = best_bic_result['power_filtering']
            best_bic_threshold = best_bic_result['power_threshold']
            best_bic_sigma = best_bic_result['sigma_noise']
            best_bic_whitening_method = best_bic_result.get('_whitening_method', 'none')
            best_bic_feature_rho = best_bic_result.get('_feature_rho', None)

            print(f"  [{dir_name}] Re-running best BIC experiment for visualizations: "
                  f"{best_bic_strategy}, {best_bic_whitening}, BIC={best_bic_result['combo_bic']:.2f}", flush=True)

            try:
                # Re-run with save_visualization=True
                _ = run_single_experiment(
                    data_info=data_info,
                    config=config,
                    map_data=map_data,
                    all_tx_locations=all_tx_locations,
                    sigma_noise=best_bic_sigma,
                    selection_method=best_bic_sel_method,
                    use_power_filtering=best_bic_pf,
                    whitening_method=best_bic_whitening_method,
                    feature_rho=best_bic_feature_rho,
                    whitening_config_name=best_bic_whitening,
                    strategy_name=best_bic_strategy,
                    model_type=model_type,
                    recon_model_type=recon_model_type,
                    eta=eta,
                    output_dir=output_dir,
                    save_visualization=True,  # NOW enable visualizations
                    verbose=False,
                    power_density_threshold=best_bic_threshold if best_bic_pf else 0.0,
                    beam_width=beam_width,
                    max_pool_size=max_pool_size,
                    use_edf_penalty=use_edf_penalty,
                    edf_threshold=edf_threshold,
                    use_robust_scoring=use_robust_scoring,
                    robust_threshold=robust_threshold,
                    save_iterations=save_iterations,
                    pooling_lambda=pooling_lambda,
                    dedupe_distance_m=dedupe_distance_m,
                    combo_min_distance_m=combo_min_distance_m,
                    combo_max_size=combo_max_size,
                    combo_max_candidates=combo_max_candidates,
                    combo_bic_weight=combo_bic_weight,
                    combo_max_power_diff_dB=combo_max_power_diff_dB,
                    combo_sensor_proximity_threshold_m=combo_sensor_proximity_threshold_m,
                    combo_sensor_proximity_penalty=combo_sensor_proximity_penalty,
                )
            except Exception as viz_exc:
                print(f"  [{dir_name}] Warning: Failed to generate visualizations: {viz_exc}", flush=True)

    return results, None  # None = no skip reason, processing succeeded


def append_results_to_csv(results: List[Dict], output_dir: Path):
    """
    Append a batch of results to the main results CSV file.
    Handles header creation if file doesn't exist.
    """
    if not results:
        return
        
    csv_path = output_dir / 'all_results.csv'
    df = pd.DataFrame(results)
    
    # Reorder columns to ensure consistent header
    existing_cols = list(df.columns)
    ordered_cols = []
    
    # Add desired columns if they are present in the dataframe
    for col in DESIRED_COLUMN_ORDER:
        if col in existing_cols:
            ordered_cols.append(col)
            
    # Add remaining columns
    for col in existing_cols:
        if col not in ordered_cols:
            ordered_cols.append(col)
            
    # Apply reordering
    df = df[ordered_cols]
    
    # Check if file exists to determine if header is needed
    file_exists = csv_path.exists()
    
    # Use append mode 'a'
    # Locking is not implemented here, assuming main thread calls this sequentially
    try:
        df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
    except Exception as e:
        print(f"Warning: Failed to append results to CSV: {e}")


def save_bic_results_csv(results_df: pd.DataFrame, output_dir: Path):
    """
    Save a simplified BIC-only results CSV with key metrics from combinatorial selection.

    The CSV contains:
    - dir_name: Directory name
    - transmitters: TX identifiers
    - seed: Random seed
    - strategy: GLRT strategy
    - whitening_config: Whitening configuration
    - tx_count: True TX count
    - combo_n_tx: Number of TXs in optimal combination
    - combo_ale: Average Localization Error from BIC selection
    - combo_pd: Probability of Detection from BIC selection
    - combo_precision: Precision from BIC selection
    - combo_count_error: |true_tx_count - estimated_tx_count|

    Parameters
    ----------
    results_df : pd.DataFrame
        Full results dataframe
    output_dir : Path
        Output directory
    """
    # Select only BIC-relevant columns (including strategy and whitening config)
    bic_columns = [
        'dir_name',
        'transmitters',
        'seed',
        'strategy',
        'whitening_config',
        'tx_count',
        'combo_n_tx',
        'combo_ale',
        'combo_pd',
        'combo_precision',
        'combo_count_error',
    ]

    # Filter to columns that exist
    available_columns = [col for col in bic_columns if col in results_df.columns]

    if len(available_columns) == 0:
        print("Warning: No BIC columns found in results")
        return

    bic_df = results_df[available_columns].copy()

    # Sort by dir_name, transmitters, seed, strategy, whitening_config
    sort_cols = [col for col in ['dir_name', 'transmitters', 'seed', 'strategy', 'whitening_config'] if col in bic_df.columns]
    if sort_cols:
        bic_df = bic_df.sort_values(sort_cols)

    # Save to CSV
    csv_path = output_dir / 'all_results_bic.csv'
    bic_df.to_csv(csv_path, index=False)
    print(f"BIC results saved to: {csv_path}")
    print(f"  Total rows: {len(bic_df)}")

    # Print summary statistics
    if 'combo_ale' in bic_df.columns:
        valid_ale = bic_df['combo_ale'].dropna()
        if len(valid_ale) > 0:
            print(f"  Mean ALE: {valid_ale.mean():.2f} m")
    if 'combo_pd' in bic_df.columns:
        valid_pd = bic_df['combo_pd'].dropna()
        if len(valid_pd) > 0:
            print(f"  Mean Pd: {valid_pd.mean()*100:.1f}%")
    if 'combo_precision' in bic_df.columns:
        valid_prec = bic_df['combo_precision'].dropna()
        if len(valid_prec) > 0:
            print(f"  Mean Precision: {valid_prec.mean()*100:.1f}%")
    if 'combo_count_error' in bic_df.columns:
        valid_ce = bic_df['combo_count_error'].dropna()
        if len(valid_ce) > 0:
            print(f"  Mean Count Error: {valid_ce.mean():.2f}")

    return bic_df


def generate_bic_analysis_report(bic_df: pd.DataFrame, output_dir: Path):
    """
    Generate a markdown analysis report for BIC-based combinatorial selection results.

    Parameters
    ----------
    bic_df : pd.DataFrame
        BIC results dataframe
    output_dir : Path
        Output directory
    """
    report_lines = []
    report_lines.append("# BIC Combinatorial Selection Analysis Report")
    report_lines.append("")
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Overall Summary
    report_lines.append("## Overall Summary")
    report_lines.append("")
    report_lines.append(f"- **Total experiments**: {len(bic_df)}")

    if 'tx_count' in bic_df.columns:
        report_lines.append(f"- **TX count range**: {int(bic_df['tx_count'].min())} - {int(bic_df['tx_count'].max())}")

    report_lines.append("")
    report_lines.append("### Aggregate Metrics")
    report_lines.append("")
    report_lines.append("| Metric | Mean | Std | Min | Max |")
    report_lines.append("|--------|------|-----|-----|-----|")

    for col, name in [('combo_ale', 'ALE (m)'), ('combo_pd', 'Pd'), ('combo_precision', 'Precision'), ('combo_count_error', 'Count Error')]:
        if col in bic_df.columns:
            valid = bic_df[col].dropna()
            if len(valid) > 0:
                if col in ['combo_pd', 'combo_precision']:
                    report_lines.append(f"| {name} | {valid.mean()*100:.1f}% | {valid.std()*100:.1f}% | {valid.min()*100:.1f}% | {valid.max()*100:.1f}% |")
                else:
                    report_lines.append(f"| {name} | {valid.mean():.2f} | {valid.std():.2f} | {valid.min():.2f} | {valid.max():.2f} |")

    # Analysis by Strategy
    if 'strategy' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Analysis by Strategy")
        report_lines.append("")
        report_lines.append("| Strategy | Count | Mean ALE | Mean Pd | Mean Precision | Mean Count Error |")
        report_lines.append("|----------|-------|----------|---------|----------------|------------------|")

        for strategy in sorted(bic_df['strategy'].unique()):
            subset = bic_df[bic_df['strategy'] == strategy]
            count = len(subset)
            ale = subset['combo_ale'].mean() if 'combo_ale' in subset.columns else np.nan
            pd_val = subset['combo_pd'].mean() if 'combo_pd' in subset.columns else np.nan
            prec = subset['combo_precision'].mean() if 'combo_precision' in subset.columns else np.nan
            ce = subset['combo_count_error'].mean() if 'combo_count_error' in subset.columns else np.nan

            ale_str = f"{ale:.2f}" if not np.isnan(ale) else "-"
            pd_str = f"{pd_val*100:.1f}%" if not np.isnan(pd_val) else "-"
            prec_str = f"{prec*100:.1f}%" if not np.isnan(prec) else "-"
            ce_str = f"{ce:.2f}" if not np.isnan(ce) else "-"

            report_lines.append(f"| {strategy} | {count} | {ale_str} | {pd_str} | {prec_str} | {ce_str} |")

    # Analysis by Whitening Config
    if 'whitening_config' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Analysis by Whitening Configuration")
        report_lines.append("")
        report_lines.append("| Whitening Config | Count | Mean ALE | Mean Pd | Mean Precision | Mean Count Error |")
        report_lines.append("|------------------|-------|----------|---------|----------------|------------------|")

        for wc in sorted(bic_df['whitening_config'].unique()):
            subset = bic_df[bic_df['whitening_config'] == wc]
            count = len(subset)
            ale = subset['combo_ale'].mean() if 'combo_ale' in subset.columns else np.nan
            pd_val = subset['combo_pd'].mean() if 'combo_pd' in subset.columns else np.nan
            prec = subset['combo_precision'].mean() if 'combo_precision' in subset.columns else np.nan
            ce = subset['combo_count_error'].mean() if 'combo_count_error' in subset.columns else np.nan

            ale_str = f"{ale:.2f}" if not np.isnan(ale) else "-"
            pd_str = f"{pd_val*100:.1f}%" if not np.isnan(pd_val) else "-"
            prec_str = f"{prec*100:.1f}%" if not np.isnan(prec) else "-"
            ce_str = f"{ce:.2f}" if not np.isnan(ce) else "-"

            report_lines.append(f"| {wc} | {count} | {ale_str} | {pd_str} | {prec_str} | {ce_str} |")

    # Analysis by TX Count
    if 'tx_count' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Analysis by True TX Count")
        report_lines.append("")
        report_lines.append("| TX Count | Experiments | Mean ALE | Mean Pd | Mean Precision | Mean Count Error | Mean Est. TXs |")
        report_lines.append("|----------|-------------|----------|---------|----------------|------------------|---------------|")

        for tx_count in sorted(bic_df['tx_count'].unique()):
            subset = bic_df[bic_df['tx_count'] == tx_count]
            count = len(subset)
            ale = subset['combo_ale'].mean() if 'combo_ale' in subset.columns else np.nan
            pd_val = subset['combo_pd'].mean() if 'combo_pd' in subset.columns else np.nan
            prec = subset['combo_precision'].mean() if 'combo_precision' in subset.columns else np.nan
            ce = subset['combo_count_error'].mean() if 'combo_count_error' in subset.columns else np.nan
            est_tx = subset['combo_n_tx'].mean() if 'combo_n_tx' in subset.columns else np.nan

            ale_str = f"{ale:.2f}" if not np.isnan(ale) else "-"
            pd_str = f"{pd_val*100:.1f}%" if not np.isnan(pd_val) else "-"
            prec_str = f"{prec*100:.1f}%" if not np.isnan(prec) else "-"
            ce_str = f"{ce:.2f}" if not np.isnan(ce) else "-"
            est_str = f"{est_tx:.2f}" if not np.isnan(est_tx) else "-"

            report_lines.append(f"| {int(tx_count)} | {count} | {ale_str} | {pd_str} | {prec_str} | {ce_str} | {est_str} |")

    # Analysis by Strategy + Whitening (Top configurations)
    if 'strategy' in bic_df.columns and 'whitening_config' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Best Configurations (by Mean ALE)")
        report_lines.append("")

        grouped = bic_df.groupby(['strategy', 'whitening_config']).agg({
            'combo_ale': ['mean', 'std', 'count'],
            'combo_pd': 'mean',
            'combo_precision': 'mean',
            'combo_count_error': 'mean'
        }).reset_index()

        # Flatten column names
        grouped.columns = ['strategy', 'whitening_config', 'ale_mean', 'ale_std', 'count', 'pd_mean', 'prec_mean', 'ce_mean']

        # Sort by ALE mean (lower is better)
        grouped = grouped.sort_values('ale_mean')

        report_lines.append("| Rank | Strategy | Whitening | Count | Mean ALE | Std ALE | Mean Pd | Mean Prec | Mean CE |")
        report_lines.append("|------|----------|-----------|-------|----------|---------|---------|-----------|---------|")

        for i, row in grouped.head(10).iterrows():
            rank = grouped.index.get_loc(i) + 1
            ale_str = f"{row['ale_mean']:.2f}" if not np.isnan(row['ale_mean']) else "-"
            ale_std_str = f"{row['ale_std']:.2f}" if not np.isnan(row['ale_std']) else "-"
            pd_str = f"{row['pd_mean']*100:.1f}%" if not np.isnan(row['pd_mean']) else "-"
            prec_str = f"{row['prec_mean']*100:.1f}%" if not np.isnan(row['prec_mean']) else "-"
            ce_str = f"{row['ce_mean']:.2f}" if not np.isnan(row['ce_mean']) else "-"

            report_lines.append(f"| {rank} | {row['strategy']} | {row['whitening_config']} | {int(row['count'])} | {ale_str} | {ale_std_str} | {pd_str} | {prec_str} | {ce_str} |")

    # TX Count Estimation Accuracy
    if 'tx_count' in bic_df.columns and 'combo_n_tx' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## TX Count Estimation Accuracy")
        report_lines.append("")

        # Perfect count rate
        perfect_count = (bic_df['combo_count_error'] == 0).sum()
        total = len(bic_df)
        report_lines.append(f"- **Perfect count rate**: {perfect_count}/{total} ({perfect_count/total*100:.1f}%)")

        # Under/over estimation
        under = (bic_df['combo_n_tx'] < bic_df['tx_count']).sum()
        over = (bic_df['combo_n_tx'] > bic_df['tx_count']).sum()
        exact = (bic_df['combo_n_tx'] == bic_df['tx_count']).sum()
        report_lines.append(f"- **Under-estimation**: {under} ({under/total*100:.1f}%)")
        report_lines.append(f"- **Exact**: {exact} ({exact/total*100:.1f}%)")
        report_lines.append(f"- **Over-estimation**: {over} ({over/total*100:.1f}%)")

        report_lines.append("")
        report_lines.append("### Count Error Distribution")
        report_lines.append("")
        report_lines.append("| Count Error | Occurrences | Percentage |")
        report_lines.append("|-------------|-------------|------------|")

        for ce in sorted(bic_df['combo_count_error'].unique()):
            ce_count = (bic_df['combo_count_error'] == ce).sum()
            report_lines.append(f"| {int(ce)} | {ce_count} | {ce_count/total*100:.1f}% |")

    # Per-Directory Analysis
    if 'dir_name' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Per-Directory Analysis")
        report_lines.append("")

        for dir_name in sorted(bic_df['dir_name'].unique()):
            dir_subset = bic_df[bic_df['dir_name'] == dir_name]
            report_lines.append(f"### {dir_name}")
            report_lines.append("")

            # Summary for this directory
            dir_count = len(dir_subset)
            dir_ale = dir_subset['combo_ale'].mean() if 'combo_ale' in dir_subset.columns else np.nan
            dir_pd = dir_subset['combo_pd'].mean() if 'combo_pd' in dir_subset.columns else np.nan
            dir_prec = dir_subset['combo_precision'].mean() if 'combo_precision' in dir_subset.columns else np.nan
            dir_ce = dir_subset['combo_count_error'].mean() if 'combo_count_error' in dir_subset.columns else np.nan

            report_lines.append(f"- **Experiments**: {dir_count}")
            if not np.isnan(dir_ale):
                report_lines.append(f"- **Mean ALE**: {dir_ale:.2f} m")
            if not np.isnan(dir_pd):
                report_lines.append(f"- **Mean Pd**: {dir_pd*100:.1f}%")
            if not np.isnan(dir_prec):
                report_lines.append(f"- **Mean Precision**: {dir_prec*100:.1f}%")
            if not np.isnan(dir_ce):
                report_lines.append(f"- **Mean Count Error**: {dir_ce:.2f}")

            # TX count estimation for this directory
            if 'tx_count' in dir_subset.columns and 'combo_n_tx' in dir_subset.columns:
                perfect = (dir_subset['combo_count_error'] == 0).sum()
                report_lines.append(f"- **Perfect Count Rate**: {perfect}/{dir_count} ({perfect/dir_count*100:.1f}%)")

            report_lines.append("")

            # Breakdown by TX count within this directory
            if 'tx_count' in dir_subset.columns and len(dir_subset['tx_count'].unique()) > 1:
                report_lines.append("| TX Count | Experiments | Mean ALE | Mean Pd | Mean Prec | Mean CE | Mean Est |")
                report_lines.append("|----------|-------------|----------|---------|-----------|---------|----------|")

                for tx_count in sorted(dir_subset['tx_count'].unique()):
                    tx_subset = dir_subset[dir_subset['tx_count'] == tx_count]
                    tx_count_n = len(tx_subset)
                    tx_ale = tx_subset['combo_ale'].mean() if 'combo_ale' in tx_subset.columns else np.nan
                    tx_pd = tx_subset['combo_pd'].mean() if 'combo_pd' in tx_subset.columns else np.nan
                    tx_prec = tx_subset['combo_precision'].mean() if 'combo_precision' in tx_subset.columns else np.nan
                    tx_ce = tx_subset['combo_count_error'].mean() if 'combo_count_error' in tx_subset.columns else np.nan
                    tx_est = tx_subset['combo_n_tx'].mean() if 'combo_n_tx' in tx_subset.columns else np.nan

                    ale_str = f"{tx_ale:.2f}" if not np.isnan(tx_ale) else "-"
                    pd_str = f"{tx_pd*100:.1f}%" if not np.isnan(tx_pd) else "-"
                    prec_str = f"{tx_prec*100:.1f}%" if not np.isnan(tx_prec) else "-"
                    ce_str = f"{tx_ce:.2f}" if not np.isnan(tx_ce) else "-"
                    est_str = f"{tx_est:.2f}" if not np.isnan(tx_est) else "-"

                    report_lines.append(f"| {int(tx_count)} | {tx_count_n} | {ale_str} | {pd_str} | {prec_str} | {ce_str} | {est_str} |")

                report_lines.append("")

    # Per-Transmitter Set Analysis (by unique transmitters combination)
    if 'transmitters' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Per-Transmitter Set Analysis")
        report_lines.append("")
        report_lines.append("| Transmitters | TX Count | Experiments | Mean ALE | Mean Pd | Mean Prec | Mean CE | Perfect Rate |")
        report_lines.append("|--------------|----------|-------------|----------|---------|-----------|---------|--------------|")

        for tx_set in sorted(bic_df['transmitters'].unique()):
            tx_subset = bic_df[bic_df['transmitters'] == tx_set]
            tx_count_val = tx_subset['tx_count'].iloc[0] if 'tx_count' in tx_subset.columns else 0
            n_exp = len(tx_subset)
            ale = tx_subset['combo_ale'].mean() if 'combo_ale' in tx_subset.columns else np.nan
            pd_val = tx_subset['combo_pd'].mean() if 'combo_pd' in tx_subset.columns else np.nan
            prec = tx_subset['combo_precision'].mean() if 'combo_precision' in tx_subset.columns else np.nan
            ce = tx_subset['combo_count_error'].mean() if 'combo_count_error' in tx_subset.columns else np.nan
            perfect = (tx_subset['combo_count_error'] == 0).sum() if 'combo_count_error' in tx_subset.columns else 0

            ale_str = f"{ale:.2f}" if not np.isnan(ale) else "-"
            pd_str = f"{pd_val*100:.1f}%" if not np.isnan(pd_val) else "-"
            prec_str = f"{prec*100:.1f}%" if not np.isnan(prec) else "-"
            ce_str = f"{ce:.2f}" if not np.isnan(ce) else "-"
            perfect_str = f"{perfect}/{n_exp} ({perfect/n_exp*100:.0f}%)"

            report_lines.append(f"| {tx_set} | {int(tx_count_val)} | {n_exp} | {ale_str} | {pd_str} | {prec_str} | {ce_str} | {perfect_str} |")

    # === Reconstruction Error Analysis ===
    if 'recon_rmse' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Reconstruction Error Analysis")
        report_lines.append("")

        valid_recon = bic_df[bic_df['recon_status'] == 'success']
        report_lines.append(f"- **Experiments with validation data**: {len(valid_recon)}/{len(bic_df)}")

        if len(valid_recon) > 0:
            report_lines.append(f"- **Mean Reconstruction RMSE**: {valid_recon['recon_rmse'].mean():.2f} dB")
            report_lines.append(f"- **Mean Reconstruction MAE**: {valid_recon['recon_mae'].mean():.2f} dB")
            report_lines.append(f"- **Mean Reconstruction Bias**: {valid_recon['recon_bias'].mean():.2f} dB")
            report_lines.append(f"- **Mean Max Error**: {valid_recon['recon_max_error'].mean():.2f} dB")
            report_lines.append(f"- **Mean Validation Points**: {valid_recon['recon_n_val_points'].mean():.0f}")

            # Breakdown by status
            report_lines.append("")
            report_lines.append("### Reconstruction Status Breakdown")
            report_lines.append("")
            report_lines.append("| Status | Count | Percentage |")
            report_lines.append("|--------|-------|------------|")
            for status in bic_df['recon_status'].unique():
                count = (bic_df['recon_status'] == status).sum()
                pct = count / len(bic_df) * 100
                report_lines.append(f"| {status} | {count} | {pct:.1f}% |")

            # Reconstruction error by TX count
            if 'tx_count' in valid_recon.columns and len(valid_recon) > 0:
                report_lines.append("")
                report_lines.append("### Reconstruction Error by TX Count")
                report_lines.append("")
                report_lines.append("| TX Count | Experiments | Mean RMSE | Mean MAE | Mean Bias | Mean Max Error |")
                report_lines.append("|----------|-------------|-----------|----------|-----------|----------------|")

                for tx_count in sorted(valid_recon['tx_count'].unique()):
                    subset = valid_recon[valid_recon['tx_count'] == tx_count]
                    n_exp = len(subset)
                    rmse = subset['recon_rmse'].mean()
                    mae = subset['recon_mae'].mean()
                    bias = subset['recon_bias'].mean()
                    max_err = subset['recon_max_error'].mean()

                    rmse_str = f"{rmse:.2f}" if not np.isnan(rmse) else "-"
                    mae_str = f"{mae:.2f}" if not np.isnan(mae) else "-"
                    bias_str = f"{bias:.2f}" if not np.isnan(bias) else "-"
                    max_str = f"{max_err:.2f}" if not np.isnan(max_err) else "-"

                    report_lines.append(f"| {int(tx_count)} | {n_exp} | {rmse_str} | {mae_str} | {bias_str} | {max_str} |")

    # Save report
    report_path = output_dir / 'analysis_report_bic.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"BIC analysis report saved to: {report_path}")


def create_final_results(results_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Create final_results.csv by selecting the best strategy per directory based on lowest BIC.

    For each unique (dir_name, transmitters, seed) combination, select the row with
    the lowest combo_bic score.

    Parameters
    ----------
    results_df : pd.DataFrame
        Full results dataframe with all strategies
    output_dir : Path
        Output directory

    Returns
    -------
    pd.DataFrame
        Final results with one row per directory (best strategy selected by BIC)
    """
    if 'combo_bic' not in results_df.columns:
        print("Warning: combo_bic column not found, cannot create final results")
        return pd.DataFrame()

    # Group by directory identifiers and select row with minimum BIC
    group_cols = ['dir_name', 'transmitters', 'seed']
    available_group_cols = [col for col in group_cols if col in results_df.columns]

    if not available_group_cols:
        print("Warning: No grouping columns found")
        return pd.DataFrame()

    # For each group, get the row with minimum combo_bic
    idx = results_df.groupby(available_group_cols)['combo_bic'].idxmin()
    final_df = results_df.loc[idx].copy()

    # Select relevant columns for final results
    final_columns = [
        'dir_name', 'transmitters', 'seed', 'tx_count',
        'strategy', 'whitening_config', 'sigma_noise',
        'combo_n_tx', 'combo_ale', 'combo_pd', 'combo_precision',
        'combo_count_error', 'combo_rmse', 'combo_bic',
        # Also include GLRT metrics for reference
        'ale', 'pd', 'precision', 'n_estimated',
    ]

    available_final_cols = [col for col in final_columns if col in final_df.columns]
    final_df = final_df[available_final_cols]

    # Sort by dir_name, transmitters, seed
    sort_cols = [col for col in ['dir_name', 'transmitters', 'seed'] if col in final_df.columns]
    if sort_cols:
        final_df = final_df.sort_values(sort_cols)

    # Save to CSV
    csv_path = output_dir / 'final_results.csv'
    final_df.to_csv(csv_path, index=False)

    print(f"Final results saved to: {csv_path}")
    print(f"  Total directories: {len(final_df)}")
    print(f"  (Selected best strategy per directory based on lowest BIC)")

    return final_df


def generate_final_analysis_report(final_df: pd.DataFrame, output_dir: Path):
    """
    Generate analysis_report_final.md for the best-strategy-per-directory results.

    Parameters
    ----------
    final_df : pd.DataFrame
        Final results dataframe (one row per directory)
    output_dir : Path
        Output directory
    """
    report_lines = []
    report_lines.append("# Final Results Analysis Report")
    report_lines.append("")
    report_lines.append("This report analyzes results after selecting the **best strategy per directory**")
    report_lines.append("based on lowest BIC score.")
    report_lines.append("")
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Overall Summary
    report_lines.append("## Overall Summary")
    report_lines.append("")
    report_lines.append(f"- **Total directories**: {len(final_df)}")

    if 'tx_count' in final_df.columns:
        report_lines.append(f"- **TX count range**: {int(final_df['tx_count'].min())} - {int(final_df['tx_count'].max())}")

    report_lines.append("")
    report_lines.append("### Aggregate Metrics (Best Strategy per Directory)")
    report_lines.append("")
    report_lines.append("| Metric | Mean | Std | Min | Max |")
    report_lines.append("|--------|------|-----|-----|-----|")

    for col, name in [('combo_ale', 'ALE (m)'), ('combo_pd', 'Pd'), ('combo_precision', 'Precision'), ('combo_count_error', 'Count Error'), ('combo_bic', 'BIC')]:
        if col in final_df.columns:
            valid = final_df[col].dropna()
            if len(valid) > 0:
                if col in ['combo_pd', 'combo_precision']:
                    report_lines.append(f"| {name} | {valid.mean()*100:.1f}% | {valid.std()*100:.1f}% | {valid.min()*100:.1f}% | {valid.max()*100:.1f}% |")
                else:
                    report_lines.append(f"| {name} | {valid.mean():.2f} | {valid.std():.2f} | {valid.min():.2f} | {valid.max():.2f} |")

    # Strategy selection distribution
    if 'strategy' in final_df.columns:
        report_lines.append("")
        report_lines.append("## Strategy Selection Distribution")
        report_lines.append("")
        report_lines.append("Which strategies were selected as best (by BIC) across directories:")
        report_lines.append("")
        report_lines.append("| Strategy | Times Selected | Percentage |")
        report_lines.append("|----------|----------------|------------|")

        strategy_counts = final_df['strategy'].value_counts()
        total = len(final_df)
        for strategy, count in strategy_counts.items():
            report_lines.append(f"| {strategy} | {count} | {count/total*100:.1f}% |")

    # Whitening config selection distribution
    if 'whitening_config' in final_df.columns:
        report_lines.append("")
        report_lines.append("## Whitening Configuration Selection Distribution")
        report_lines.append("")
        report_lines.append("| Whitening Config | Times Selected | Percentage |")
        report_lines.append("|------------------|----------------|------------|")

        wc_counts = final_df['whitening_config'].value_counts()
        for wc, count in wc_counts.items():
            report_lines.append(f"| {wc} | {count} | {count/total*100:.1f}% |")

    # Analysis by TX Count
    if 'tx_count' in final_df.columns:
        report_lines.append("")
        report_lines.append("## Results by True TX Count")
        report_lines.append("")
        report_lines.append("| TX Count | Directories | Mean ALE | Mean Pd | Mean Precision | Mean Count Error | Mean Est. TXs |")
        report_lines.append("|----------|-------------|----------|---------|----------------|------------------|---------------|")

        for tx_count in sorted(final_df['tx_count'].unique()):
            subset = final_df[final_df['tx_count'] == tx_count]
            n_dirs = len(subset)
            ale = subset['combo_ale'].mean() if 'combo_ale' in subset.columns else np.nan
            pd_val = subset['combo_pd'].mean() if 'combo_pd' in subset.columns else np.nan
            prec = subset['combo_precision'].mean() if 'combo_precision' in subset.columns else np.nan
            ce = subset['combo_count_error'].mean() if 'combo_count_error' in subset.columns else np.nan
            est_tx = subset['combo_n_tx'].mean() if 'combo_n_tx' in subset.columns else np.nan

            ale_str = f"{ale:.2f}" if not np.isnan(ale) else "-"
            pd_str = f"{pd_val*100:.1f}%" if not np.isnan(pd_val) else "-"
            prec_str = f"{prec*100:.1f}%" if not np.isnan(prec) else "-"
            ce_str = f"{ce:.2f}" if not np.isnan(ce) else "-"
            est_str = f"{est_tx:.2f}" if not np.isnan(est_tx) else "-"

            report_lines.append(f"| {int(tx_count)} | {n_dirs} | {ale_str} | {pd_str} | {prec_str} | {ce_str} | {est_str} |")

    # TX Count Estimation Accuracy
    if 'tx_count' in final_df.columns and 'combo_n_tx' in final_df.columns:
        report_lines.append("")
        report_lines.append("## TX Count Estimation Accuracy")
        report_lines.append("")

        total = len(final_df)
        if 'combo_count_error' in final_df.columns:
            perfect = (final_df['combo_count_error'] == 0).sum()
            report_lines.append(f"- **Perfect count rate**: {perfect}/{total} ({perfect/total*100:.1f}%)")

        under = (final_df['combo_n_tx'] < final_df['tx_count']).sum()
        over = (final_df['combo_n_tx'] > final_df['tx_count']).sum()
        exact = (final_df['combo_n_tx'] == final_df['tx_count']).sum()
        report_lines.append(f"- **Under-estimation**: {under} ({under/total*100:.1f}%)")
        report_lines.append(f"- **Exact**: {exact} ({exact/total*100:.1f}%)")
        report_lines.append(f"- **Over-estimation**: {over} ({over/total*100:.1f}%)")

    # Per-Directory Details
    if 'dir_name' in final_df.columns:
        report_lines.append("")
        report_lines.append("## Per-Directory Results")
        report_lines.append("")
        report_lines.append("| Directory | TXs | Best Strategy | Whitening | Est TXs | ALE | Pd | Prec | BIC |")
        report_lines.append("|-----------|-----|---------------|-----------|---------|-----|-----|------|-----|")

        for _, row in final_df.iterrows():
            dir_name = row.get('dir_name', '-')
            tx_count = int(row['tx_count']) if 'tx_count' in row and not pd.isna(row['tx_count']) else '-'
            strategy = row.get('strategy', '-')
            whitening = row.get('whitening_config', '-')
            est_tx = int(row['combo_n_tx']) if 'combo_n_tx' in row and not pd.isna(row['combo_n_tx']) else '-'
            ale = f"{row['combo_ale']:.1f}" if 'combo_ale' in row and not pd.isna(row['combo_ale']) else '-'
            pd_val = f"{row['combo_pd']*100:.0f}%" if 'combo_pd' in row and not pd.isna(row['combo_pd']) else '-'
            prec = f"{row['combo_precision']*100:.0f}%" if 'combo_precision' in row and not pd.isna(row['combo_precision']) else '-'
            bic = f"{row['combo_bic']:.1f}" if 'combo_bic' in row and not pd.isna(row['combo_bic']) else '-'

            report_lines.append(f"| {dir_name} | {tx_count} | {strategy} | {whitening} | {est_tx} | {ale} | {pd_val} | {prec} | {bic} |")

    # Save report
    report_path = output_dir / 'analysis_report_final.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Final analysis report saved to: {report_path}")


def cleanup_visualizations_for_best_only(final_df: pd.DataFrame, output_dir: Path):
    """
    Remove visualization directories for non-best strategies, keeping only the best.

    For each directory in final_df, we need to identify which experiment_name
    corresponds to the best strategy and remove others.

    Parameters
    ----------
    final_df : pd.DataFrame
        Final results with best strategy per directory
    output_dir : Path
        Output directory containing glrt_visualizations/
    """
    vis_dir = output_dir / 'glrt_visualizations'
    if not vis_dir.exists():
        print("No visualization directory found, skipping cleanup")
        return

    # Build set of experiment names to keep (from final_df)
    # experiment_name format: {tx_underscore}_{strategy}_{whitening}_{selection}_{pf}
    # We need to reconstruct this from the final_df rows

    keep_dirs = set()
    for _, row in final_df.iterrows():
        # Reconstruct experiment name components
        transmitters = row.get('transmitters', '')
        tx_underscore = transmitters.replace(',', '_') if transmitters else ''
        strategy = row.get('strategy', '')
        whitening = row.get('whitening_config', '')
        # selection_method and power_filtering are also part of the name
        # but they may vary - we need to check what's in the actual dir names

        # The experiment name pattern is complex, let's match by prefix
        if tx_underscore and strategy and whitening:
            # Look for directories starting with this pattern
            pattern_start = f"{tx_underscore}_{strategy}_{whitening}"
            keep_dirs.add(pattern_start)

    if not keep_dirs:
        print("Could not determine directories to keep, skipping cleanup")
        return

    # List all visualization subdirectories
    removed_count = 0
    kept_count = 0

    for subdir in vis_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Check if this directory matches any of the keep patterns
        should_keep = False
        for pattern in keep_dirs:
            if subdir.name.startswith(pattern):
                should_keep = True
                break

        if should_keep:
            kept_count += 1
        else:
            # Remove this directory
            import shutil
            try:
                shutil.rmtree(subdir)
                removed_count += 1
            except Exception as e:
                print(f"Warning: Could not remove {subdir}: {e}")

    print(f"Visualization cleanup: kept {kept_count} directories, removed {removed_count}")


def run_comprehensive_sweep(
    grouped_dirs: Dict[int, List[Dict]],
    config: Dict,
    map_data: Dict,
    all_tx_locations: Dict,
    output_dir: Path,
    test_mode: bool = False,
    tx_counts_filter: Optional[List[int]] = None,
    nloc_filter: Optional[int] = None,
    max_dirs_per_count: Optional[int] = None,
    model_type: str = 'tirem',
    recon_model_type: str = 'tirem',
    eta: float = 0.01,
    save_visualizations: bool = True,
    verbose: bool = True,
    n_workers: int = 1,
    power_thresholds: List[float] = None,
    whitening_configs: Dict = None,
    beam_width: int = 1,
    max_pool_size: int = 50,
    use_edf_penalty: bool = False,
    edf_threshold: float = 1.5,
    use_robust_scoring: bool = False,
    robust_threshold: float = 6.0,
    save_iterations: bool = False,

    pooling_lambda: float = 0.01,
    dedupe_distance_m: float = 60.0,

    # Combinatorial selection parameters
    combo_min_distance_m: float = 100.0,
    combo_max_size: int = 5,
    combo_max_candidates: int = 10,
    combo_bic_weight: float = 0.2,
    combo_max_power_diff_dB: float = 20.0,
    combo_sensor_proximity_threshold_m: float = 100.0,
    combo_sensor_proximity_penalty: float = 10.0,
) -> pd.DataFrame:
    """
    Run comprehensive parameter sweep across all directories.
    
    Parameters
    ----------
    n_workers : int, optional
        Number of parallel workers. Default: 1 (sequential).
        Set to -1 to use all CPUs minus 1.
    
    Returns
    -------
    pd.DataFrame
        Results dataframe with all experiments
    """
    results = []
    # Selection configurations: (method, use_power_filtering)
    selection_configs = [
        ('max', False),
        ('cluster', False),
        ('max', True),
        ('cluster', True),
    ]
    
    # If Beam Search is enabled (width > 1), 'max' and 'cluster' inputs yield identical Hybrid results.
    # To avoid 2x redundant computation, prune the list to only use 'max' as the representative input.
    if beam_width > 1:
        if verbose:
             print(f"  Beam Search (width={beam_width}) enabled: Pruning redundant 'cluster' selection methods.")
        selection_configs = [c for c in selection_configs if c[0] == 'max']
        
    # If no power thresholds are provided, disable power filtering experiments
    if not power_thresholds:
        if verbose:
            print("  No power density thresholds provided: Disabling power filtering experiments.")
        selection_configs = [c for c in selection_configs if not c[1]]
    
    # Whitening configurations
    if whitening_configs is None:
        whitening_configs = AVAILABLE_WHITENING_CONFIGS
        if test_mode:
             # Only use diagonal in test mode (faster) if not explicitly provided
             whitening_configs = {'hetero_diag': ('hetero_diag', None)}
    
    print(f"Whitening configs to run: {list(whitening_configs.keys())}")
    
    # Filter TX counts if specified
    if tx_counts_filter:
        grouped_dirs = {k: v for k, v in grouped_dirs.items() if k in tx_counts_filter}

    # Filter by nloc if specified
    if nloc_filter is not None:
        grouped_dirs = {
            k: [d for d in v if d.get('num_locations') == nloc_filter]
            for k, v in grouped_dirs.items()
        }
        # Remove empty groups
        grouped_dirs = {k: v for k, v in grouped_dirs.items() if v}

    # Flatten directories list with TX count info
    all_dirs = []
    for tx_count in sorted(grouped_dirs.keys()):
        dirs = grouped_dirs[tx_count]
        if max_dirs_per_count:
            dirs = dirs[:max_dirs_per_count]
        all_dirs.extend(dirs)
    
    total_dirs = len(all_dirs)
    
    # Determine actual worker count
    if n_workers == -1:
        n_workers = max(1, os.cpu_count() - 1)
    n_workers = min(n_workers, total_dirs)  # Don't use more workers than dirs
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE PARAMETER SWEEP")
    print(f"{'='*70}")
    print(f"TX counts to process: {sorted(grouped_dirs.keys())}")
    if nloc_filter is not None:
        print(f"nloc filter: {nloc_filter}")
    print(f"Total directories: {total_dirs}")
    print(f"Workers: {n_workers}")
    print(f"Test mode: {test_mode}")
    print(f"Localization model: {model_type}")
    print(f"Reconstruction model: {recon_model_type}")
    print(f"Whitening configs: {list(whitening_configs.keys())}")
    print(f"EDF Penalty: {use_edf_penalty} (Threshold: {edf_threshold})")
    print(f"Robust Scoring: {use_robust_scoring} (Threshold: {robust_threshold})")
    print(f"Save Iterations: {save_iterations}")
    print(f"Pooling Refinement Lambda: {pooling_lambda}")
    print(f"Dedupe Distance: {dedupe_distance_m}m")
    print(f"\nCombinatorial Selection:")
    print(f"  Min TX Distance: {combo_min_distance_m}m")
    print(f"  Max Combination Size: {combo_max_size}")
    print(f"  Max Candidates: {combo_max_candidates}")
    print(f"  BIC Penalty Weight: {combo_bic_weight}")
    print(f"  Max Power Diff: {combo_max_power_diff_dB}dB")
    print(f"  Sensor Proximity Threshold: {combo_sensor_proximity_threshold_m}m")
    print(f"  Sensor Proximity Penalty: {combo_sensor_proximity_penalty}")

    start_time = time.time()
    
    # Prepare serializable arguments for each directory
    # Convert Path objects to strings to avoid pickling issues on Windows
    output_dir_str = str(output_dir)
    
    task_args = []
    for data_info in all_dirs:
        # Create serializable copy of data_info
        data_info_serializable = {
            'name': data_info['name'],
            'transmitters': data_info['transmitters'],
            'num_locations': data_info.get('num_locations'),
            'seed': data_info['seed'],
            'path_str': str(data_info['path']),  # Convert Path to string
        }
        task_args.append((
            data_info_serializable,
            config,
            map_data,
            all_tx_locations,
            output_dir_str,
            test_mode,
            model_type,
            recon_model_type,
            eta,
            save_visualizations,
            whitening_configs,
            selection_configs,
            power_thresholds,
            beam_width,
            max_pool_size,
            use_edf_penalty,
            edf_threshold,
            use_robust_scoring,
            robust_threshold,
            save_iterations,
            pooling_lambda,
            dedupe_distance_m,
            # Combinatorial selection parameters
            combo_min_distance_m,
            combo_max_size,
            combo_max_candidates,
            combo_bic_weight,
            combo_max_power_diff_dB,
            combo_sensor_proximity_threshold_m,
            combo_sensor_proximity_penalty,
        ))
    
    if n_workers == 1:
        # Sequential execution
        print("\nRunning in sequential mode...")
        for i, args in enumerate(task_args):
            dir_name = args[0]['name']
            print(f"  [{i+1}/{total_dirs}] Processing {dir_name}...", end=" ", flush=True)
            
            dir_results, skip_reason = process_single_directory(args)
            
            if skip_reason:
                print(f"SKIPPED ({skip_reason})")
            else:
                print(f"{len(dir_results)} experiments")
                # Incremental save
                results.extend(dir_results)
                append_results_to_csv(dir_results, output_dir)
            
            # Progress estimate
            elapsed = time.time() - start_time
            if i > 0:
                avg_time = elapsed / (i + 1)
                remaining = (total_dirs - i - 1) * avg_time
                print(f"      Progress: {i+1}/{total_dirs} | "
                      f"Elapsed: {elapsed/60:.1f}min | "
                      f"Remaining: ~{remaining/60:.1f}min")
    else:
        # Parallel execution using concurrent.futures for incremental processing
        print(f"\nRunning in parallel mode with {n_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(process_single_directory, args): args for args in task_args}
            
            n_skipped = 0
            n_success = 0
            
            # Process as they complete
            for i, future in enumerate(as_completed(future_to_args)):
                try:
                    dir_results, skip_reason = future.result()
                    
                    if skip_reason:
                        n_skipped += 1
                    else:
                        n_success += 1
                        results.extend(dir_results)
                        # Incremental save
                        append_results_to_csv(dir_results, output_dir)
                        
                except Exception as exc:
                    print(f"Generated an exception: {exc}")
                    n_skipped += 1

        print(f"  Directories processed: {n_success} success, {n_skipped} skipped")
    
    elapsed_total = time.time() - start_time
    print(f"\nSweep completed in {elapsed_total/60:.1f} minutes")
    print(f"Total experiments collected: {len(results)}")
    
    return pd.DataFrame(results)


def analyze_by_tx_count(results_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Generate analysis summary for each TX count.
    
    Returns
    -------
    dict
        Dictionary mapping TX count -> summary DataFrame
    """
    summaries = {}
    
    for tx_count in sorted(results_df['tx_count'].unique()):
        df = results_df[results_df['tx_count'] == tx_count]
        
        # Group by strategy, selection method, power filtering, threshold, and whitening_config
        grouped = df.groupby(['strategy', 'selection_method', 'power_filtering', 'power_threshold', 'whitening_config'], dropna=False).agg({
            'ale': ['mean', 'std', 'min', 'max', 'count'],
            'pd': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'n_estimated': ['mean', 'std'],
            'n_est_raw': ['mean'],
            'n_est_diff': ['mean'],
            'count_error': ['mean', 'std', 'min', 'max'],
            'tp': 'sum',
            'fp': 'sum',
            'fn': 'sum',
            'runtime_s': 'mean',
            'glrt_final_score': ['mean', 'std', 'min', 'max'],
            'glrt_n_iterations': ['mean', 'std'],
            'glrt_score_reduction': ['mean', 'std'],
        }).reset_index()
        
        # Flatten column names
        grouped.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col 
            for col in grouped.columns
        ]
        
        # Sort by mean ALE
        grouped = grouped.sort_values('ale_mean')
        
        summaries[tx_count] = grouped
    
    return summaries


def analyze_universal(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate universal analysis across all TX counts.
    
    Returns
    -------
    pd.DataFrame
        Summary dataframe
    """
    # Group by strategy, selection method, power filtering, threshold, and whitening_config
    grouped = results_df.groupby(['strategy', 'selection_method', 'power_filtering', 'power_threshold', 'whitening_config'], dropna=False).agg({
        'ale': ['mean', 'std', 'min', 'max', 'count'],
        'pd': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'n_estimated': ['mean', 'std'],
        'n_est_raw': ['mean'],
        'n_est_diff': ['mean'],
        'count_error': ['mean', 'std', 'min', 'max'],
        'tp': 'sum',
        'fp': 'sum',
        'fn': 'sum',
        'runtime_s': 'mean',
        'glrt_final_score': ['mean', 'std', 'min', 'max'],
        'glrt_n_iterations': ['mean', 'std'],
        'glrt_score_reduction': ['mean', 'std'],
    }).reset_index()
    
    # Flatten column names
    grouped.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col 
        for col in grouped.columns
    ]
    
    # Sort by mean ALE
    grouped = grouped.sort_values('ale_mean')
    
    return grouped



def analyze_by_tx_set(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance grouped by specific transmitter sets."""
    # Group by key columns
    group_cols = ['transmitters', 'strategy', 'selection_method', 'power_filtering', 'power_threshold', 'whitening_config']
    
    # Calculate aggregation
    summary = results_df.groupby(group_cols).agg(
        ale_mean=('ale', 'mean'),
        ale_std=('ale', 'std'),
        ale_min=('ale', 'min'),
        ale_max=('ale', 'max'),
        ale_count=('ale', 'count'),
        pd_mean=('pd', 'mean'),
        pd_std=('pd', 'std'),
        precision_mean=('precision', 'mean'),
        f1_mean=('f1_score', 'mean'),
        n_est_mean=('n_estimated', 'mean'),
        n_est_raw_mean=('n_est_raw', 'mean'),
        n_est_diff_mean=('n_est_diff', 'mean'),
        count_error_mean=('count_error', 'mean'),
    ).reset_index()
    
    # Sort by transmitters and then by ALE
    summary = summary.sort_values(['transmitters', 'ale_mean'])
    
    return summary


def analyze_glrt_score_correlation(results_df: pd.DataFrame) -> Dict:
    """
    Analyze correlation between GLRT scores and localization performance.
    
    Determines if GLRT scores can predict which configuration will perform best.
    
    Returns
    -------
    dict
        Correlation analysis results including:
        - Overall correlations (Pearson/Spearman)
        - Per-directory comparison of best ALE config vs. lowest GLRT score config
        - Summary statistics
    """
    from scipy import stats
    
    analysis = {
        'overall_correlations': {},
        'per_directory_analysis': [],
        'matching_rate': 0.0,
        'summary': '',
    }
    
    # Check if GLRT score columns exist
    if 'glrt_final_score' not in results_df.columns:
        analysis['summary'] = 'GLRT score columns not found in results.'
        return analysis
    
    # Filter out rows with missing/invalid GLRT scores OR missing ALE
    valid_df = results_df[
        (results_df['glrt_final_score'].notna()) & 
        (results_df['glrt_final_score'] > 0) &
        (results_df['ale'].notna())
    ].copy()
    
    if len(valid_df) < 10:
        analysis['summary'] = f'Insufficient data for correlation analysis ({len(valid_df)} valid experiments).'
        return analysis
    
    # Overall correlations
    try:
        # Pearson correlation between final GLRT score and ALE
        pearson_r, pearson_p = stats.pearsonr(valid_df['glrt_final_score'], valid_df['ale'])
        analysis['overall_correlations']['final_score_vs_ale_pearson'] = {
            'r': float(pearson_r),
            'p_value': float(pearson_p),
        }
        
        # Spearman correlation (rank-based, more robust to outliers)
        spearman_r, spearman_p = stats.spearmanr(valid_df['glrt_final_score'], valid_df['ale'])
        analysis['overall_correlations']['final_score_vs_ale_spearman'] = {
            'r': float(spearman_r),
            'p_value': float(spearman_p),
        }
        
        # Correlation between score reduction and ALE
        if valid_df['glrt_score_reduction'].notna().any():
            reduction_data = valid_df[valid_df['glrt_score_reduction'].notna()]
            if len(reduction_data) > 10:
                sr_pearson_r, sr_pearson_p = stats.pearsonr(reduction_data['glrt_score_reduction'], reduction_data['ale'])
                analysis['overall_correlations']['score_reduction_vs_ale'] = {
                    'r': float(sr_pearson_r),
                    'p_value': float(sr_pearson_p),
                }
        
        # Correlation between n_iterations and ALE
        if valid_df['glrt_n_iterations'].notna().any():
            iter_data = valid_df[valid_df['glrt_n_iterations'].notna()]
            if len(iter_data) > 10:
                iter_pearson_r, iter_pearson_p = stats.pearsonr(iter_data['glrt_n_iterations'], iter_data['ale'])
                analysis['overall_correlations']['n_iterations_vs_ale'] = {
                    'r': float(iter_pearson_r),
                    'p_value': float(iter_pearson_p),
                }
                
    except Exception as e:
        analysis['overall_correlations']['error'] = str(e)
    
    # Per-directory analysis: does lowest GLRT score predict best ALE?
    matches = 0
    total_dirs = 0
    
    for dir_name in valid_df['dir_name'].unique():
        dir_df = valid_df[valid_df['dir_name'] == dir_name]
        
        if len(dir_df) < 2:
            continue
        
        total_dirs += 1
        
        # Find config with lowest ALE
        best_ale_idx = dir_df['ale'].idxmin()
        best_ale_config = dir_df.loc[best_ale_idx]
        
        # Find config with lowest final GLRT score
        lowest_glrt_idx = dir_df['glrt_final_score'].idxmin()
        lowest_glrt_config = dir_df.loc[lowest_glrt_idx]
        
        # Check if they match (same strategy, selection method, etc.)
        config_match = (
            best_ale_config['strategy'] == lowest_glrt_config['strategy'] and
            best_ale_config['selection_method'] == lowest_glrt_config['selection_method'] and
            best_ale_config['power_filtering'] == lowest_glrt_config['power_filtering']
        )
        
        if config_match:
            matches += 1
        
        analysis['per_directory_analysis'].append({
            'dir_name': dir_name,
            'best_ale': float(best_ale_config['ale']),
            'best_ale_config': f"{best_ale_config['strategy']}_{best_ale_config['selection_method']}",
            'best_ale_glrt_score': float(best_ale_config['glrt_final_score']),
            'lowest_glrt_score': float(lowest_glrt_config['glrt_final_score']),
            'lowest_glrt_config': f"{lowest_glrt_config['strategy']}_{lowest_glrt_config['selection_method']}",
            'lowest_glrt_ale': float(lowest_glrt_config['ale']),
            'configs_match': config_match,
        })
    
    # Compute matching rate
    if total_dirs > 0:
        analysis['matching_rate'] = matches / total_dirs
    
    # Generate summary
    pearson_r = analysis['overall_correlations'].get('final_score_vs_ale_pearson', {}).get('r', None)
    spearman_r = analysis['overall_correlations'].get('final_score_vs_ale_spearman', {}).get('r', None)
    
    if pearson_r is not None:
        correlation_strength = 'weak'
        if abs(pearson_r) > 0.5:
            correlation_strength = 'moderate'
        if abs(pearson_r) > 0.7:
            correlation_strength = 'strong'
        
        direction = 'positive' if pearson_r > 0 else 'negative'
        
        analysis['summary'] = (
            f"GLRT final score shows {correlation_strength} {direction} correlation with ALE "
            f"(Pearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}). "
            f"The config with lowest GLRT score matched the best ALE config in "
            f"{analysis['matching_rate']*100:.1f}% of directories ({matches}/{total_dirs})."
        )
    else:
        analysis['summary'] = f"Could not compute correlations. Matching rate: {analysis['matching_rate']*100:.1f}%"
    
    return analysis

def generate_analysis_report(
    results_df: pd.DataFrame,
    tx_count_summaries: Dict[int, pd.DataFrame],
    universal_summary: pd.DataFrame,
    output_dir: Path,
    tx_set_summary: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate a comprehensive markdown analysis report.
    
    Returns
    -------
    str
        Path to the generated report
    """
    report_lines = []
    
    report_lines.append("# Comprehensive Reconstruction Parameter Sweep Analysis")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Overview
    report_lines.append("## Overview")
    report_lines.append("")
    report_lines.append(f"- **Total experiments**: {len(results_df)}")
    report_lines.append(f"- **Unique directories**: {results_df['dir_name'].nunique()}")
    report_lines.append(f"- **TX counts analyzed**: {sorted(results_df['tx_count'].unique())}")
    report_lines.append(f"- **Strategies tested**: {results_df['strategy'].nunique()}")
    report_lines.append(f"- **Selection methods**: {results_df['selection_method'].unique().tolist()}")
    report_lines.append(f"- **Whitening Configs**: {results_df['whitening_config'].unique().tolist()}")
    report_lines.append("")
    
    # Universal Analysis
    report_lines.append("## Universal Analysis (Across All TX Counts)")
    report_lines.append("")
    
    # Best overall strategy
    best_row = universal_summary.iloc[0]
    report_lines.append(f"### Best Overall Strategy")
    report_lines.append("")
    report_lines.append(f"- **Strategy**: `{best_row['strategy']}`")
    report_lines.append(f"- **Selection Method**: `{best_row['selection_method']}`")
    report_lines.append(f"- **Whitening Config**: `{best_row['whitening_config']}`")
    report_lines.append(f"- **Mean ALE**: {best_row['ale_mean']:.2f} m (±{best_row['ale_std']:.2f})")
    report_lines.append(f"- **Mean Pd**: {best_row['pd_mean']*100:.1f}% (±{best_row['pd_std']*100:.1f})")
    report_lines.append(f"- **Mean Precision**: {best_row['precision_mean']*100:.1f}%")
    if 'n_est_mean' in best_row:
        report_lines.append(f"- **Mean N Est (Filtered)**: {best_row['n_est_mean']:.2f} (raw: {best_row.get('n_est_raw_mean', 0):.2f})")
    if 'count_error_mean' in best_row:
        report_lines.append(f"- **Mean Count Error**: {best_row['count_error_mean']:.2f}")
    report_lines.append(f"- **Experiments**: {int(best_row['ale_count'])}")
    report_lines.append("")
    
    # Top 10 strategies table
    report_lines.append("### Top 10 Strategies (by Mean ALE)")
    report_lines.append("")
    report_lines.append("| Rank | Strategy | Selection | Power Filter | Threshold | Whitening Config | Mean ALE (m) | Mean Pd (%) | N |")
    report_lines.append("|------|----------|-----------|--------------|-----------|------------------|--------------|-------------|---|")
    for i, row in universal_summary.head(10).iterrows():
        rank = universal_summary.index.get_loc(i) + 1
        pf_str = "Yes" if row['power_filtering'] else "No"
        thresh_str = f"{row['power_threshold']}" if row['power_filtering'] else "-"
        report_lines.append(
            f"| {rank} | {row['strategy']} | {row['selection_method']} | {pf_str} | {thresh_str} | "
            f"{row['whitening_config']} | {row['ale_mean']:.2f} | "
            f"{row['pd_mean']*100:.1f} | {int(row['ale_count'])} |"
        )
    report_lines.append("")
    
    # Selection Method Comparison
    report_lines.append("### Selection Method Comparison")
    report_lines.append("")
    
    # Selection configurations: (method, use_power_filtering)
    selection_configs = [
        ('max', False),
        ('cluster', False),
        ('max', True),
        ('cluster', True),
    ]
    
    for method, use_pf in selection_configs:
        pf_suffix = " + PF" if use_pf else ""
        method_name = f"{method}{pf_suffix}"
        
        method_df = results_df[
            (results_df['selection_method'] == method) & 
            (results_df['power_filtering'] == use_pf)
        ]
        
        if len(method_df) > 0:
            avg_ale = method_df['ale'].mean()
            avg_pd = method_df['pd'].mean()
            report_lines.append(f"- **{method_name}**: Mean ALE = {avg_ale:.2f} m, Mean Pd = {avg_pd*100:.1f}%")
    report_lines.append("")
    
    # Whitening Config Comparison
    report_lines.append("### Whitening Config Comparison")
    report_lines.append("")
    
    for config_name in results_df['whitening_config'].unique():
        config_df = results_df[results_df['whitening_config'] == config_name]
        if len(config_df) > 0:
            avg_ale = config_df['ale'].mean()
            avg_pd = config_df['pd'].mean()
            report_lines.append(f"- **{config_name}**: Mean ALE = {avg_ale:.2f} m, Mean Pd = {avg_pd*100:.1f}%")
    report_lines.append("")
    
    # Fixed vs Dynamic Comparison
    report_lines.append("### Fixed vs Dynamic Sigma Noise Comparison")
    report_lines.append("")
    
    fixed_df = results_df[results_df['strategy'].str.startswith('fixed')]
    dynamic_df = results_df[~results_df['strategy'].str.startswith('fixed')]
    
    if len(fixed_df) > 0 and len(dynamic_df) > 0:
        fixed_ale = fixed_df['ale'].mean()
        dynamic_ale = dynamic_df['ale'].mean()
        fixed_pd = fixed_df['pd'].mean()
        dynamic_pd = dynamic_df['pd'].mean()
        
        report_lines.append(f"- **Fixed strategies**: Mean ALE = {fixed_ale:.2f} m, Mean Pd = {fixed_pd*100:.1f}%")
        report_lines.append(f"- **Dynamic strategies**: Mean ALE = {dynamic_ale:.2f} m, Mean Pd = {dynamic_pd*100:.1f}%")
        
        if dynamic_ale < fixed_ale:
            improvement = (fixed_ale - dynamic_ale) / fixed_ale * 100
            report_lines.append(f"- **Conclusion**: Dynamic strategies outperform fixed by {improvement:.1f}% in ALE")
        else:
            degradation = (dynamic_ale - fixed_ale) / fixed_ale * 100
            report_lines.append(f"- **Conclusion**: Fixed strategies outperform dynamic by {degradation:.1f}% in ALE")
    report_lines.append("")
    
    # Per TX Count Analysis
    report_lines.append("## Analysis by TX Count")
    report_lines.append("")
    
    for tx_count in sorted(tx_count_summaries.keys()):
        summary = tx_count_summaries[tx_count]
        
        report_lines.append(f"### TX Count = {tx_count}")
        report_lines.append("")
        
        if len(summary) > 0:
            best = summary.iloc[0]
            report_lines.append(f"**Best Strategy**: `{best['strategy']}` with `{best['selection_method']}` and `{best['whitening_config']}`")
            report_lines.append(f"- Mean ALE: {best['ale_mean']:.2f} m (±{best['ale_std']:.2f})")
            report_lines.append(f"- Mean Pd: {best['pd_mean']*100:.1f}%")
            report_lines.append(f"- Mean Precision: {best['precision_mean']*100:.1f}%")
            if 'n_est_mean' in best:
                report_lines.append(f"- Mean N (Est/Raw): {best['n_est_mean']:.1f} / {best.get('n_est_raw_mean', 0):.1f}")
            if 'count_error_mean' in best:
                report_lines.append(f"- Mean Count Error: {best['count_error_mean']:.2f}")
            report_lines.append("")
            
            # Top 5 for this TX count
            report_lines.append("| Strategy | Selection | Whitening Config | Mean ALE | Mean Pd |")
            report_lines.append("|----------|-----------|------------------|----------|---------|")
            for _, row in summary.head(5).iterrows():
                report_lines.append(
                    f"| {row['strategy']} | {row['selection_method']} | {row['whitening_config']} | "
                    f"{row['ale_mean']:.2f} | {row['pd_mean']*100:.1f}% |"
                )
            report_lines.append("")
    
    
    # Per TX Set Analysis
    if tx_set_summary is not None:
        report_lines.append("## Analysis by Transmitter Set")
        report_lines.append("")
        
        for tx_set in sorted(tx_set_summary['transmitters'].unique()):
            set_summary = tx_set_summary[tx_set_summary['transmitters'] == tx_set]
            
            report_lines.append(f"### Transmitters: {tx_set}")
            report_lines.append("")
            
            if len(set_summary) > 0:
                best = set_summary.iloc[0]
                report_lines.append(f"**Best Strategy**: `{best['strategy']}` with `{best['selection_method']}`")
                report_lines.append(f"- Mean ALE: {best['ale_mean']:.2f} m")
                report_lines.append(f"- Mean Pd: {best['pd_mean']*100:.1f}%")
                if 'n_est_mean' in best:
                     report_lines.append(f"- Mean N Est: {best['n_est_mean']:.1f}")
                report_lines.append("")
                
                # Top 3 for this set
                report_lines.append("| Strategy | Selection | Mean ALE | Mean Pd | Precision | Count Err |")
                report_lines.append("|----------|-----------|----------|---------|-----------|-----------|")
                for _, row in set_summary.head(3).iterrows():
                    prec_str = f"{row['precision_mean']*100:.1f}%" if 'precision_mean' in row else "-"
                    err_str = f"{row['count_error_mean']:.2f}" if 'count_error_mean' in row else "-"
                    report_lines.append(
                        f"| {row['strategy']} | {row['selection_method']} | "
                        f"{row['ale_mean']:.2f} | {row['pd_mean']*100:.1f}% | {prec_str} | {err_str} |"
                    )
                report_lines.append("")

    # GLRT Score Analysis
    report_lines.append("## GLRT Score Analysis")
    report_lines.append("")
    
    glrt_analysis = analyze_glrt_score_correlation(results_df)
    report_lines.append(f"**Summary**: {glrt_analysis['summary']}")
    report_lines.append("")
    
    if glrt_analysis['overall_correlations']:
        report_lines.append("### Correlations (Overall)")
        report_lines.append("| Metric Pair | Type | Correlation (r) | p-value |")
        report_lines.append("|-------------|------|----------------|---------|")
        
        corrs = glrt_analysis['overall_correlations']
        if 'final_score_vs_ale_pearson' in corrs:
            c = corrs['final_score_vs_ale_pearson']
            report_lines.append(f"| Final Score vs ALE | Pearson | {c['r']:.3f} | {c['p_value']:.4e} |")
        if 'final_score_vs_ale_spearman' in corrs:
            c = corrs['final_score_vs_ale_spearman']
            report_lines.append(f"| Final Score vs ALE | Spearman | {c['r']:.3f} | {c['p_value']:.4e} |")
        if 'score_reduction_vs_ale' in corrs:
            c = corrs['score_reduction_vs_ale']
            report_lines.append(f"| Score Reduction vs ALE | Pearson | {c['r']:.3f} | {c['p_value']:.4e} |")
        if 'n_iterations_vs_ale' in corrs:
            c = corrs['n_iterations_vs_ale']
            report_lines.append(f"| N Iterations vs ALE | Pearson | {c['r']:.3f} | {c['p_value']:.4e} |")
        report_lines.append("")
        
        report_lines.append("> **Note**: Strong positive correlation means higher score -> higher ALE (bad). Strong negative means higher score -> lower ALE (good). Ideally we want the score to be a good quality indicator (negative correlation).")
        report_lines.append("")
    
    if len(glrt_analysis['per_directory_analysis']) > 0:
        report_lines.append("### Best ALE vs Lowest GLRT Score (Per Directory)")
        report_lines.append("")
        report_lines.append(f"**Matching Rate**: {glrt_analysis['matching_rate']*100:.1f}% of directories had their best ALE result from the configuration that produced the lowest GLRT score.")
        report_lines.append("")
        
        # Show top 5 mismatches (largest ALE difference)
        mismatches = [item for item in glrt_analysis['per_directory_analysis'] if not item['configs_match']]
        mismatches.sort(key=lambda x: x['lowest_glrt_ale'] - x['best_ale'], reverse=True)
        
        if mismatches:
            report_lines.append("#### Top Mismatches (Where lowest GLRT score misled the most)")
            report_lines.append("| Directory | Best ALE Config | Best ALE (m) | Lowest GLRT Config | Lowest GLRT ALE (m) | Diff (m) |")
            report_lines.append("|-----------|-----------------|--------------|--------------------|---------------------|----------|")
            
            for item in mismatches[:5]:
                diff = item['lowest_glrt_ale'] - item['best_ale']
                report_lines.append(
                    f"| {item['dir_name']} | {item['best_ale_config']} | {item['best_ale']:.2f} | "
                    f"{item['lowest_glrt_config']} | {item['lowest_glrt_ale']:.2f} | +{diff:.2f} |"
                )
            report_lines.append("")

    # Conclusions
    report_lines.append("## Summary and Recommendations")
    report_lines.append("")
    
    # Find best strategy per TX count
    best_per_count = {}
    for tx_count, summary in tx_count_summaries.items():
        if len(summary) > 0:
            best_per_count[tx_count] = {
                'strategy': summary.iloc[0]['strategy'],
                'selection': summary.iloc[0]['selection_method'],
                'whitening_config': summary.iloc[0]['whitening_config'],
                'ale': summary.iloc[0]['ale_mean'],
            }
    
    # Check if same strategy is best across counts
    best_strategies = [v['strategy'] for v in best_per_count.values()]
    best_selections = [v['selection'] for v in best_per_count.values()]
    best_whitening_configs = [v['whitening_config'] for v in best_per_count.values()]
    
    if len(set(best_strategies)) == 1:
        report_lines.append(f"✓ **Consistent winner**: `{best_strategies[0]}` performs best across all TX counts")
    else:
        report_lines.append("ℹ **Mixed results**: Different strategies perform best at different TX counts:")
        for tx_count, info in best_per_count.items():
            report_lines.append(f"  - TX={tx_count}: `{info['strategy']}` (ALE={info['ale']:.2f}m)")
    
    report_lines.append("")
    
    if len(set(best_selections)) == 1:
        report_lines.append(f"✓ **Selection method**: `{best_selections[0]}` consistently outperforms")
    else:
        report_lines.append("ℹ **Selection method**: Results vary by TX count")
    
    report_lines.append("")
    
    if len(set(best_whitening_configs)) == 1:
        report_lines.append(f"✓ **Whitening Config**: `{best_whitening_configs[0]}` consistently outperforms")
    else:
        report_lines.append("ℹ **Whitening Config**: Results vary by TX count")
    
    report_lines.append("")
    
    # Write report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return str(report_path)


def generate_plots(
    results_df: pd.DataFrame,
    tx_count_summaries: Dict[int, pd.DataFrame],
    universal_summary: pd.DataFrame,
    output_dir: Path,
):
    """Generate visualization plots."""
    
    # Plot 1: Summary plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1a: Best ALE per TX count
    ax = axes[0, 0]
    tx_counts = sorted(tx_count_summaries.keys())
    best_ales = [tx_count_summaries[tc].iloc[0]['ale_mean'] for tc in tx_counts]
    ax.bar(tx_counts, best_ales, color='steelblue')
    ax.set_xlabel('TX Count')
    ax.set_ylabel('Best Mean ALE (m)')
    ax.set_title('Best ALE by TX Count')
    ax.set_xticks(tx_counts)
    
    # Plot 1b: Selection method comparison
    ax = axes[0, 1]
    # Create label combining method and power filtering
    results_df['method_label'] = results_df.apply(
        lambda r: f"{r['selection_method']}{' + PF' if r['power_filtering'] else ''}", axis=1
    )
    method_comparison = results_df.groupby('method_label')['ale'].agg(['mean', 'std']).reset_index()
    # Colors: Light/Dark Green for Cluster, Light/Dark Red for Max
    method_colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b'][:len(method_comparison)]
    bars = ax.bar(method_comparison['method_label'], method_comparison['mean'], 
                  yerr=method_comparison['std'], color=method_colors, capsize=5)
    ax.set_xlabel('Selection Configuration')
    ax.set_ylabel('Mean ALE (m)')
    ax.set_title('Selection Method Comparison')
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    
    # Plot 1c: Whitening Config comparison
    ax = axes[1, 0]
    config_comparison = results_df.groupby('whitening_config')['ale'].agg(['mean', 'std']).reset_index()
    # Use dynamic colors
    cmap = plt.get_cmap('tab10')
    whitening_colors = [cmap(i) for i in np.linspace(0, 1, len(config_comparison))]
    bars = ax.bar(config_comparison['whitening_config'], config_comparison['mean'], 
                  yerr=config_comparison['std'], color=whitening_colors, capsize=5)
    ax.set_xlabel('Whitening Config')
    ax.set_ylabel('Mean ALE (m)')
    ax.set_title('Whitening Config Comparison')
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    
    # Plot 1d: Fixed vs Dynamic boxplot
    ax = axes[1, 1]
    fixed_ales = results_df[results_df['strategy'].str.startswith('fixed')]['ale']
    dynamic_ales = results_df[~results_df['strategy'].str.startswith('fixed')]['ale']
    ax.boxplot([fixed_ales, dynamic_ales], labels=['Fixed σ', 'Dynamic σ'])
    ax.set_ylabel('ALE (m)')
    ax.set_title('Fixed vs Dynamic Sigma Noise')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_plots.png', dpi=150)
    plt.close()
    
    # Plot 2: Strategy comparison heatmap by TX count
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get top strategies
    top_strategies = universal_summary.head(10)[['strategy', 'selection_method', 'whitening_config']].values.tolist()
    
    # Build heatmap data
    heatmap_data = []
    for tx_count in sorted(tx_count_summaries.keys()):
        row = []
        for strat, sel, rho in top_strategies:
            df = tx_count_summaries[tx_count]
            match = df[(df['strategy'] == strat) & (df['selection_method'] == sel) & (df['whitening_config'] == rho)]
            if len(match) > 0:
                row.append(match.iloc[0]['ale_mean'])
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(range(len(top_strategies)))
    ax.set_xticklabels([f"{s[0]}\n({s[1]}/{s[2][:3]})" for s in top_strategies], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(tx_count_summaries)))
    ax.set_yticklabels([f"TX={tc}" for tc in sorted(tx_count_summaries.keys())])
    
    plt.colorbar(im, label='Mean ALE (m)')
    ax.set_title('Strategy Performance by TX Count (Mean ALE)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap.png', dpi=150)
    plt.close()
    
    # Plot 3: Power Threshold Sensitivity
    if 'power_threshold' in results_df.columns:
        pf_results = results_df[results_df['power_filtering'] == True].copy()
        if len(pf_results) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Group by threshold and selection method
            sensitivity = pf_results.groupby(['power_threshold', 'selection_method'])['ale'].mean().reset_index()
            
            for method in sensitivity['selection_method'].unique():
                 subset = sensitivity[sensitivity['selection_method'] == method]
                 subset = subset.sort_values('power_threshold')
                 ax.plot(subset['power_threshold'], subset['ale'], marker='o', label=f"{method} + PF")
            
            ax.set_xlabel('Power Density Threshold')
            ax.set_ylabel('Mean ALE (m)')
            ax.set_title('Sensitivity to Power Density Threshold (Mean across all strategies)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / 'threshold_sensitivity.png', dpi=150)
            plt.close()
    
    print(f"✓ Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive parameter sweep across all data directories.'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run in test mode with reduced parameter set and limited directories'
    )
    parser.add_argument(
        '--tx-counts', type=str, default=None,
        help='Comma-separated list of TX counts to process (e.g., 1,2,3). Default: all'
    )
    parser.add_argument(
        '--nloc', type=int, default=None,
        help='Only process directories with this specific num_locations value (e.g., 30 for nloc30)'
    )
    parser.add_argument(
        '--max-dirs', type=int, default=None,
        help='Maximum directories to process per TX count (for testing)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save results (default: results/comprehensive_sweep_<timestamp>)'
    )
    parser.add_argument(
        '--model-type', type=str, default='tirem',
        choices=['tirem', 'log_distance', 'raytracing'],
        help='Default propagation model for both localization and reconstruction (default: tirem)'
    )
    parser.add_argument(
        '--localization-model', type=str, default=None,
        choices=['tirem', 'log_distance', 'raytracing'],
        help='Propagation model for localization/GLRT (overrides --model-type)'
    )
    parser.add_argument(
        '--reconstruction-model', type=str, default=None,
        choices=['tirem', 'log_distance', 'raytracing'],
        help='Propagation model for power recomputation & reconstruction validation (overrides --model-type)'
    )
    parser.add_argument(
        '--eta', type=float, default=0.1,
        help='Eta parameter for heteroscedastic whitening (default: 0.1)'
    )
    parser.add_argument(
        '--no-visualizations', action='store_true',
        help='Disable GLRT visualization generation'
    )
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Number of parallel workers (default: 1 for sequential, -1 for all CPUs minus 1)'
    )
    parser.add_argument(
        '--power-thresholds', type=str, default=None,
        help='Comma-separated list of power density thresholds to sweep (e.g., "0.01,0.1,0.3")'
    )
    parser.add_argument(
        '--whitening-methods', type=str, default=None,
        help='Comma-separated list of whitening methods to sweep (e.g., "hetero_diag,hetero_spatial")'
    )
    parser.add_argument(
        '--beam-width', type=int, default=1,
        help='Beam width for GLRT search (default: 1)'
    )
    parser.add_argument(
        '--max-pool-size', type=int, default=50,
        help='Max candidates for pool refinement (default: 50)'
    )
    parser.add_argument(
        '--use-edf', action='store_true',
        help='Enable Consensus-Based Scoring (EDF) penalty'
    )
    parser.add_argument(
        '--edf-threshold', type=float, default=1.5,
        help='Threshold for EDF penalty (default: 1.5)'
    )
    parser.add_argument(
        '--use-robust-scoring', action='store_true',
        help='Enable Robust GLRT scoring (Huber-like loss)'
    )
    parser.add_argument(
        '--robust-threshold', type=float, default=6.0,
        help='Threshold for robust clipping (default: 6.0)'
    )
    parser.add_argument(
        '--pooling-lambda', type=float, default=0.01,
        help='Regularization constant for pooling refinement (active only if refinement enabled)'
    )
    parser.add_argument(
        '--save-iterations', action='store_true',
        help='Save visualization for each GLRT iteration (default: False)'
    )
    parser.add_argument(
        '--dedupe-distance', type=float, default=60.0,
        help='Distance threshold for post-search transmitter deduplication (default: 60.0 m)'
    )

    # Combinatorial selection arguments
    parser.add_argument(
        '--combo-min-distance', type=float, default=100.0,
        help='Minimum distance between paired TXs in combinatorial selection (default: 100.0 m)'
    )
    parser.add_argument(
        '--combo-max-size', type=int, default=5,
        help='Maximum number of TXs in a combination (default: 5)'
    )
    parser.add_argument(
        '--combo-max-candidates', type=int, default=10,
        help='Maximum number of top candidates to consider for combinations (default: 10)'
    )
    parser.add_argument(
        '--combo-bic-weight', type=float, default=0.05,
        help='BIC penalty weight for model complexity (default: 0.05)'
    )
    parser.add_argument(
        '--combo-max-power-diff', type=float, default=20.0,
        help='Maximum TX power difference in dB for combinations (default: 20.0)'
    )
    parser.add_argument(
        '--combo-sensor-proximity-threshold', type=float, default=100.0,
        help='Distance threshold (m) for sensor proximity penalty (default: 100.0)'
    )
    parser.add_argument(
        '--combo-sensor-proximity-penalty', type=float, default=10.0,
        help='Constant BIC penalty for each TX within proximity threshold of a sensor (default: 10.0)'
    )

    args = parser.parse_args()

    # Resolve model types: specific flags override --model-type
    localization_model = args.localization_model or args.model_type
    reconstruction_model = args.reconstruction_model or args.model_type

    # Parse thresholds list
    if args.power_thresholds:
        power_thresholds = [float(x) for x in args.power_thresholds.split(',')]
    else:
        # Default changed: If flag omitted, disable power filtering entirely
        power_thresholds = None
    
    # Parse TX counts filter
    tx_counts_filter = None
    if args.tx_counts:
        tx_counts_filter = [int(x.strip()) for x in args.tx_counts.split(',')]
        
    # Parse whitening methods
    whitening_configs = None
    if args.whitening_methods:
        methods = [x.strip() for x in args.whitening_methods.split(',')]
        whitening_configs = {}
        for m in methods:
            if m in AVAILABLE_WHITENING_CONFIGS:
                whitening_configs[m] = AVAILABLE_WHITENING_CONFIGS[m]
            else:
                print(f"Warning: Unknown whitening method '{m}', skipping. Available: {list(AVAILABLE_WHITENING_CONFIGS.keys())}")
        
        if not whitening_configs:
            print("Error: No valid whitening methods selected. Exiting.")
            return
    
    # Set max dirs for test mode
    max_dirs = args.max_dirs
    if args.test and max_dirs is None:
        max_dirs = 2  # Only 2 dirs per TX count in test mode
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f'results/comprehensive_sweep_{timestamp}')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("COMPREHENSIVE RECONSTRUCTION PARAMETER SWEEP")
    print("=" * 70)
    
    # Load base configuration
    print("\nLoading configuration...")
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load SLC map
    print("Loading SLC map...")
    map_data = load_slc_map(
        map_folder_dir="./",
        downsample_factor=config['spatial']['downsample_factor']
    )
    print(f"  Map shape: {map_data['shape']}")
    
    # Load transmitter locations
    print("Loading transmitter locations...")
    all_tx_locations = load_transmitter_locations(
        config_path='config/transmitter_locations.yaml',
        map_data=map_data
    )
    print(f"  Transmitters: {list(all_tx_locations.keys())}")
    
    # Discover data directories
    print("\nDiscovering data directories...")
    base_dir = Path('data/processed')
    grouped_dirs = discover_data_directories(base_dir)
    
    for tx_count in sorted(grouped_dirs.keys()):
        print(f"  TX count {tx_count}: {len(grouped_dirs[tx_count])} directories")
        
    # If using custom whitening configs, print them
    if whitening_configs:
        print(f"Custom whitening configs: {list(whitening_configs.keys())}")
    
    # Run comprehensive sweep
    results_df = run_comprehensive_sweep(
        grouped_dirs=grouped_dirs,
        config=config,
        map_data=map_data,
        all_tx_locations=all_tx_locations,
        output_dir=output_dir,
        test_mode=args.test,
        tx_counts_filter=tx_counts_filter,
        nloc_filter=args.nloc,
        max_dirs_per_count=max_dirs,
        model_type=localization_model,
        recon_model_type=reconstruction_model,
        eta=args.eta,
        save_visualizations=not args.no_visualizations,
        verbose=True,
        n_workers=args.workers,
        power_thresholds=power_thresholds,
        whitening_configs=whitening_configs,
        beam_width=args.beam_width,
        max_pool_size=args.max_pool_size,
        use_edf_penalty=args.use_edf,
        edf_threshold=args.edf_threshold,
        use_robust_scoring=args.use_robust_scoring,
        robust_threshold=args.robust_threshold,

        save_iterations=args.save_iterations,

        pooling_lambda=args.pooling_lambda,
        dedupe_distance_m=args.dedupe_distance,

        # Combinatorial selection parameters
        combo_min_distance_m=args.combo_min_distance,
        combo_max_size=args.combo_max_size,
        combo_max_candidates=args.combo_max_candidates,
        combo_bic_weight=args.combo_bic_weight,
        combo_max_power_diff_dB=args.combo_max_power_diff,
        combo_sensor_proximity_threshold_m=args.combo_sensor_proximity_threshold,
        combo_sensor_proximity_penalty=args.combo_sensor_proximity_penalty,
    )
    
    if len(results_df) == 0:
        print("\nNo results collected. Exiting.")
        return
    
    # Sort results for consistent grouping
    results_df = results_df.sort_values(
        by=['dir_name', 'transmitters', 'seed', 'strategy', 'selection_method', 'power_filtering'],
        na_position='last'
    )
    
    # Reorder columns as requested
    existing_cols = list(results_df.columns)
    ordered_cols = []
    
    # Add desired columns if they are present in the dataframe
    for col in DESIRED_COLUMN_ORDER:
        if col in existing_cols:
            ordered_cols.append(col)
            
    # Add remaining columns
    for col in existing_cols:
        if col not in ordered_cols:
            ordered_cols.append(col)
            
    # Apply reordering
    results_df = results_df[ordered_cols]
    

    # Save raw results
    results_path = output_dir / 'all_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Raw results saved to: {results_path}")

    # Save BIC-only results CSV and generate BIC analysis report
    print("\nSaving BIC results...")
    bic_df = save_bic_results_csv(results_df, output_dir)
    if bic_df is not None and len(bic_df) > 0:
        generate_bic_analysis_report(bic_df, output_dir)

    # Create final results (best strategy per directory based on lowest BIC)
    print("\nCreating final results (best strategy per directory)...")
    final_df = create_final_results(results_df, output_dir)
    if final_df is not None and len(final_df) > 0:
        generate_final_analysis_report(final_df, output_dir)
        # Note: Visualization cleanup no longer needed - visualizations are only created
        # for the best BIC strategy per directory during processing

    # Generate analysis
    print("\nGenerating analysis...")
    tx_count_summaries = analyze_by_tx_count(results_df)
    universal_summary = analyze_universal(results_df)
    
    # Save summaries
    universal_summary.to_csv(output_dir / 'universal_summary.csv', index=False)
    for tx_count, summary in tx_count_summaries.items():
        summary.to_csv(output_dir / f'summary_tx{tx_count}.csv', index=False)
    
    # Generate report
    report_path = generate_analysis_report(
        results_df=results_df,
        tx_count_summaries=tx_count_summaries,
        universal_summary=universal_summary,
        output_dir=output_dir,
    )
    print(f"✓ Analysis report saved to: {report_path}")

    # Generate per-tx set analysis
    print("\nGenerating per-transmitter set analysis...")
    tx_set_summary = analyze_by_tx_set(results_df)
    tx_set_summary.to_csv(output_dir / 'summary_by_tx_set.csv', index=False)
    
    # Re-generate report with new data
    report_path = generate_analysis_report(
        results_df=results_df,
        tx_count_summaries=tx_count_summaries,
        universal_summary=universal_summary,
        tx_set_summary=tx_set_summary,
        output_dir=output_dir,
    )
    print(f"✓ Extended analysis report saved to: {report_path}")
    
    # Generate plots
    generate_plots(
        results_df=results_df,
        tx_count_summaries=tx_count_summaries,
        universal_summary=universal_summary,
        output_dir=output_dir,
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"\nTotal experiments: {len(results_df)}")
    print(f"Results directory: {output_dir}")
    
    # Print best overall
    best = universal_summary.iloc[0]
    print(f"\nBest overall strategy: {best['strategy']} ({best['selection_method']}, {best['whitening_config']})")
    print(f"  Mean ALE: {best['ale_mean']:.2f} m")
    print(f"  Mean Pd: {best['pd_mean']*100:.1f}%")
    
    return results_df


if __name__ == "__main__":
    # Required for Windows multiprocessing support
    multiprocessing.freeze_support()
    main()
