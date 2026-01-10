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
"""

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


# Known transmitter names (alphabetically sorted for canonical ordering)
KNOWN_TRANSMITTERS = ['guesthouse', 'mario', 'moran', 'ustar', 'wasatch']

# Feature rho configurations to sweep
FEATURE_RHO_CONFIGS = {
    'los_normalized': [0.2, 1e10, 1e10, 1e10],  # LOS normalized
    'no_normalization': [1e10, 1e10, 1e10, 1e10],  # No normalization
}

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


def parse_directory_name(dir_name: str) -> Tuple[List[str], Optional[int]]:
    """
    Parse a data directory name to extract transmitter names and seed.
    
    Examples:
        'mario_moran_seed_32' -> (['mario', 'moran'], 32)
        'guesthouse_wasatch_ustar_seed_5' -> (['guesthouse', 'wasatch', 'ustar'], 5)
        'mario' -> (['mario'], None)
        'validation_mario' -> ([], None)  # Skip validation directories
    
    Parameters
    ----------
    dir_name : str
        Directory name to parse
        
    Returns
    -------
    tuple
        (list of transmitter names, seed value or None)
    """
    # Skip validation directories
    if dir_name.startswith('validation_'):
        return [], None
    
    # Check for seed pattern
    seed_match = re.search(r'_seed_(\d+)$', dir_name)
    seed = int(seed_match.group(1)) if seed_match else None
    
    # Remove seed suffix if present
    name_part = re.sub(r'_seed_\d+$', '', dir_name)
    
    # Split by underscore and filter for known transmitters
    parts = name_part.split('_')
    transmitters = [p for p in parts if p in KNOWN_TRANSMITTERS]
    
    return transmitters, seed


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
        Each info dict contains: 'path', 'transmitters', 'seed', 'name'
    """
    grouped = defaultdict(list)
    
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
            
        transmitters, seed = parse_directory_name(item.name)
        
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
    strategies = {
        # Fixed values (common choices)
        'fixed_1e-10': 1e-10,
        'fixed_1e-9': 1e-9,
        'fixed_5e-9': 5e-9,
        'fixed_1e-8': 1e-8,
        'fixed_1e-7': 1e-7,
        
        # Min-power based strategies (multiples of min observed power)
        '0.5x_min': 0.5 * min_power,
        'min_power': min_power,
        '2x_min': 2.0 * min_power,
        '5x_min': 5.0 * min_power,
        '10x_min': 10.0 * min_power,
        '20x_min': 20.0 * min_power,
        
        # Sqrt-based (often used for Poisson-like noise)
        'sqrt_min': np.sqrt(min_power),
        'sqrt_mean': np.sqrt(mean_power),
        
        # Mean-power based
        '0.01_mean': 0.01 * mean_power,
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
        
        # Highlight selected
        sel_row, sel_col = np.unravel_index(item['selected_index'], map_data['shape'])
        ax.scatter([sel_col], [sel_row], c='magenta', marker='*', s=400, label='Selected Candidate', zorder=11)
        
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


def run_single_experiment(
    data_info: Dict,
    config: Dict,
    map_data: Dict,
    all_tx_locations: Dict,
    sigma_noise: float,
    selection_method: str,
    feature_rho: List[float],
    feature_rho_name: str,
    model_type: str = 'tirem',
    eta: float = 0.01,
    output_dir: Optional[Path] = None,
    save_visualization: bool = False,
    verbose: bool = False,
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
        
        # Build config path
        if seed is not None:
            config_path = f'config/monitoring_locations_{tx_underscore}_seed_{seed}.yaml'
        else:
            config_path = f'config/monitoring_locations_{tx_underscore}.yaml'
        
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
        
        # Get true transmitter locations
        tx_locations = {name: all_tx_locations[name] for name in transmitters if name in all_tx_locations}
        true_locs_pixels = np.array([tx['coordinates'] for tx in tx_locations.values()])
        
        start_time = time.time()
        
        # Determine model config path
        if model_type == 'tirem':
            model_config_path = 'config/tirem_parameters.yaml'
        elif model_type == 'raytracing':
            model_config_path = 'config/sionna_parameters.yaml'
        else:
            model_config_path = None
        
        # Run reconstruction with hetero_geo_aware whitening
        tx_map, info = joint_sparse_reconstruction(
            sensor_locations=sensor_locations,
            observed_powers_dBm=observed_powers_dB,
            input_is_linear=False,
            solve_in_linear_domain=True,
            map_shape=map_data['shape'],
            scale=config['spatial']['proxel_size'],
            np_exponent=config['localization']['path_loss_exponent'],
            lambda_reg=0,
            norm_exponent=0,
            whitening_method='hetero_geo_aware',
            sigma_noise=sigma_noise,
            eta=eta,
            feature_rho=feature_rho,
            solver='glrt',
            selection_method=selection_method,
            cluster_max_candidates=30,
            glrt_max_iter=len(transmitters) + 1,
            glrt_threshold=4.0,
            dedupe_distance_m=25.0,
            return_linear_scale=False,
            verbose=False,
            model_type=model_type,
            model_config_path=model_config_path,
            n_jobs=-1
        )
        
        elapsed = time.time() - start_time
        
        # Save GLRT visualization if requested
        if save_visualization and output_dir is not None:
            experiment_name = f"{data_info['name']}_{feature_rho_name}_{selection_method}"
            save_glrt_visualization(
                info=info,
                map_data=map_data,
                sensor_locations=sensor_locations,
                observed_powers_dB=observed_powers_dB,
                tx_locations=tx_locations,
                output_dir=output_dir,
                experiment_name=experiment_name,
            )
        
        # Extract estimated locations
        if 'solver_info' in info and 'support' in info['solver_info']:
            support_indices = info['solver_info']['support']
            height, width = map_data['shape']
            
            valid_indices = []
            for idx in support_indices:
                r, c = idx // width, idx % width
                power_dbm = tx_map[r, c]
                if power_dbm > -190:
                    valid_indices.append(idx)
            
            est_rows = [idx // width for idx in valid_indices]
            est_cols = [idx % width for idx in valid_indices]
            est_locs_pixels = np.column_stack((est_cols, est_rows)) if valid_indices else np.empty((0, 2))
        else:
            from src.evaluation.metrics import extract_locations_from_map
            est_locs_pixels = extract_locations_from_map(tx_map, threshold=1e-10)
        
        # Compute metrics
        metrics = compute_localization_metrics(
            true_locations=true_locs_pixels,
            estimated_locations=est_locs_pixels,
            scale=config['spatial']['proxel_size'],
            tolerance=200.0
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
            'runtime_s': elapsed,
            'obs_min_dbm': np.min(observed_powers_dB),
            'obs_mean_dbm': np.mean(observed_powers_dB),
            'obs_max_dbm': np.max(observed_powers_dB),
        }
        
    except Exception as e:
        if verbose:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
        return None


def run_comprehensive_sweep(
    grouped_dirs: Dict[int, List[Dict]],
    config: Dict,
    map_data: Dict,
    all_tx_locations: Dict,
    output_dir: Path,
    test_mode: bool = False,
    tx_counts_filter: Optional[List[int]] = None,
    max_dirs_per_count: Optional[int] = None,
    model_type: str = 'tirem',
    eta: float = 0.01,
    save_visualizations: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run comprehensive parameter sweep across all directories.
    
    Returns
    -------
    pd.DataFrame
        Results dataframe with all experiments
    """
    results = []
    selection_methods = ['max', 'cluster']
    
    # Feature rho configurations
    feature_rho_configs = FEATURE_RHO_CONFIGS
    if test_mode:
        # Only use one config in test mode
        feature_rho_configs = {'los_normalized': [0.2, 1e10, 1e10, 1e10]}
    
    # Filter TX counts if specified
    if tx_counts_filter:
        grouped_dirs = {k: v for k, v in grouped_dirs.items() if k in tx_counts_filter}
    
    # Count total experiments for progress
    total_dirs = sum(
        min(len(dirs), max_dirs_per_count) if max_dirs_per_count else len(dirs)
        for dirs in grouped_dirs.values()
    )
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE PARAMETER SWEEP")
    print(f"{'='*70}")
    print(f"TX counts to process: {sorted(grouped_dirs.keys())}")
    print(f"Total directories: {total_dirs}")
    print(f"Test mode: {test_mode}")
    print(f"Model type: {model_type}")
    print(f"Whitening method: hetero_geo_aware")
    print(f"Feature rho configs: {list(feature_rho_configs.keys())}")
    
    dir_counter = 0
    start_time = time.time()
    
    # Track which directories have been visualized (only visualize first strategy combo per dir)
    visualized_dirs = set()
    
    for tx_count in sorted(grouped_dirs.keys()):
        dirs = grouped_dirs[tx_count]
        if max_dirs_per_count:
            dirs = dirs[:max_dirs_per_count]
        
        print(f"\n{'-'*40}")
        print(f"TX COUNT: {tx_count} ({len(dirs)} directories)")
        print(f"{'-'*40}")
        
        for data_info in dirs:
            dir_counter += 1
            dir_name = data_info['name']
            transmitters = data_info['transmitters']
            tx_underscore = "_".join(transmitters)
            
            # Load power data for strategy definition
            powers_file = data_info['path'] / f"{tx_underscore}_avg_powers.npy"
            if not powers_file.exists():
                print(f"  [{dir_counter}/{total_dirs}] Skipping {dir_name} - no powers file")
                continue
            
            observed_powers_dB = np.load(powers_file)
            observed_powers_linear = dbm_to_linear(observed_powers_dB)
            
            # Check if TIREM cache exists (for tirem model)
            if model_type == 'tirem':
                # Load sensor locations for cache check
                seed = data_info['seed']
                if seed is not None:
                    config_path = f'config/monitoring_locations_{tx_underscore}_seed_{seed}.yaml'
                else:
                    config_path = f'config/monitoring_locations_{tx_underscore}.yaml'
                
                if not Path(config_path).exists():
                    print(f"  [{dir_counter}/{total_dirs}] Skipping {dir_name} - no config file")
                    continue
                
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
                
                if not features_cached or not prop_cached:
                    print(f"  [{dir_counter}/{total_dirs}] Skipping {dir_name} - no TIREM cache "
                          f"(features={features_cached}, prop={prop_cached})")
                    continue
            
            # Define strategies based on this dataset's observations
            strategies = define_sigma_noise_strategies(observed_powers_linear, test_mode=test_mode)
            
            print(f"\n  [{dir_counter}/{total_dirs}] {dir_name}")
            print(f"      TX: {transmitters}")
            print(f"      Strategies: {len(strategies)}, Selection methods: {len(selection_methods)}, Feature rhos: {len(feature_rho_configs)}")
            
            for strategy_name, sigma_noise in strategies.items():
                for sel_method in selection_methods:
                    for rho_name, feature_rho in feature_rho_configs.items():
                        if verbose:
                            print(f"      Running: {strategy_name} / {sel_method} / {rho_name}...", end=" ", flush=True)
                        
                        # Only save visualization for first strategy combo per directory
                        should_visualize = save_visualizations and (dir_name not in visualized_dirs)
                        
                        result = run_single_experiment(
                            data_info=data_info,
                            config=config,
                            map_data=map_data,
                            all_tx_locations=all_tx_locations,
                            sigma_noise=sigma_noise,
                            selection_method=sel_method,
                            feature_rho=feature_rho,
                            feature_rho_name=rho_name,
                            model_type=model_type,
                            eta=eta,
                            output_dir=output_dir,
                            save_visualization=should_visualize,
                            verbose=False,
                        )
                        
                        if result is not None:
                            if should_visualize:
                                visualized_dirs.add(dir_name)
                            
                            result.update({
                                'dir_name': dir_name,
                                'tx_count': tx_count,
                                'transmitters': ','.join(transmitters),
                                'seed': data_info['seed'],
                                'strategy': strategy_name,
                                'selection_method': sel_method,
                                'feature_rho': rho_name,
                                'sigma_noise': sigma_noise,
                                'sigma_noise_dB': 10 * np.log10(sigma_noise) if sigma_noise > 0 else -np.inf,
                            })
                            results.append(result)
                            
                            if verbose:
                                print(f"ALE={result['ale']:.1f}m, Pd={result['pd']*100:.0f}%")
                        else:
                            if verbose:
                                print("FAILED")
            
            # Progress estimate
            elapsed = time.time() - start_time
            if dir_counter > 0:
                avg_time = elapsed / dir_counter
                remaining = (total_dirs - dir_counter) * avg_time
                print(f"      Progress: {dir_counter}/{total_dirs} | "
                      f"Elapsed: {elapsed/60:.1f}min | "
                      f"Remaining: ~{remaining/60:.1f}min")
    
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
        
        # Group by strategy, selection method, and feature_rho
        grouped = df.groupby(['strategy', 'selection_method', 'feature_rho']).agg({
            'ale': ['mean', 'std', 'min', 'max', 'count'],
            'pd': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'tp': 'sum',
            'fp': 'sum',
            'fn': 'sum',
            'runtime_s': 'mean',
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
    # Group by strategy, selection method, and feature_rho
    grouped = results_df.groupby(['strategy', 'selection_method', 'feature_rho']).agg({
        'ale': ['mean', 'std', 'min', 'max', 'count'],
        'pd': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'tp': 'sum',
        'fp': 'sum',
        'fn': 'sum',
        'runtime_s': 'mean',
    }).reset_index()
    
    # Flatten column names
    grouped.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col 
        for col in grouped.columns
    ]
    
    # Sort by mean ALE
    grouped = grouped.sort_values('ale_mean')
    
    return grouped


def generate_analysis_report(
    results_df: pd.DataFrame,
    tx_count_summaries: Dict[int, pd.DataFrame],
    universal_summary: pd.DataFrame,
    output_dir: Path,
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
    report_lines.append(f"- **Feature rho configs**: {results_df['feature_rho'].unique().tolist()}")
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
    report_lines.append(f"- **Feature Rho**: `{best_row['feature_rho']}`")
    report_lines.append(f"- **Mean ALE**: {best_row['ale_mean']:.2f} m (±{best_row['ale_std']:.2f})")
    report_lines.append(f"- **Mean Pd**: {best_row['pd_mean']*100:.1f}% (±{best_row['pd_std']*100:.1f})")
    report_lines.append(f"- **Mean Precision**: {best_row['precision_mean']*100:.1f}%")
    report_lines.append(f"- **Experiments**: {int(best_row['ale_count'])}")
    report_lines.append("")
    
    # Top 10 strategies table
    report_lines.append("### Top 10 Strategies (by Mean ALE)")
    report_lines.append("")
    report_lines.append("| Rank | Strategy | Selection | Feature Rho | Mean ALE (m) | Mean Pd (%) | N |")
    report_lines.append("|------|----------|-----------|-------------|--------------|-------------|---|")
    for i, row in universal_summary.head(10).iterrows():
        rank = universal_summary.index.get_loc(i) + 1
        report_lines.append(
            f"| {rank} | {row['strategy']} | {row['selection_method']} | "
            f"{row['feature_rho']} | {row['ale_mean']:.2f} | "
            f"{row['pd_mean']*100:.1f} | {int(row['ale_count'])} |"
        )
    report_lines.append("")
    
    # Selection Method Comparison
    report_lines.append("### Selection Method Comparison")
    report_lines.append("")
    
    for method in ['max', 'cluster']:
        method_df = results_df[results_df['selection_method'] == method]
        if len(method_df) > 0:
            avg_ale = method_df['ale'].mean()
            avg_pd = method_df['pd'].mean()
            report_lines.append(f"- **{method}**: Mean ALE = {avg_ale:.2f} m, Mean Pd = {avg_pd*100:.1f}%")
    report_lines.append("")
    
    # Feature Rho Comparison
    report_lines.append("### Feature Rho Comparison")
    report_lines.append("")
    
    for rho_name in results_df['feature_rho'].unique():
        rho_df = results_df[results_df['feature_rho'] == rho_name]
        if len(rho_df) > 0:
            avg_ale = rho_df['ale'].mean()
            avg_pd = rho_df['pd'].mean()
            report_lines.append(f"- **{rho_name}**: Mean ALE = {avg_ale:.2f} m, Mean Pd = {avg_pd*100:.1f}%")
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
            report_lines.append(f"**Best Strategy**: `{best['strategy']}` with `{best['selection_method']}` and `{best['feature_rho']}`")
            report_lines.append(f"- Mean ALE: {best['ale_mean']:.2f} m (±{best['ale_std']:.2f})")
            report_lines.append(f"- Mean Pd: {best['pd_mean']*100:.1f}%")
            report_lines.append(f"- Mean Precision: {best['precision_mean']*100:.1f}%")
            report_lines.append("")
            
            # Top 5 for this TX count
            report_lines.append("| Strategy | Selection | Feature Rho | Mean ALE | Mean Pd |")
            report_lines.append("|----------|-----------|-------------|----------|---------|")
            for _, row in summary.head(5).iterrows():
                report_lines.append(
                    f"| {row['strategy']} | {row['selection_method']} | {row['feature_rho']} | "
                    f"{row['ale_mean']:.2f} | {row['pd_mean']*100:.1f}% |"
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
                'feature_rho': summary.iloc[0]['feature_rho'],
                'ale': summary.iloc[0]['ale_mean'],
            }
    
    # Check if same strategy is best across counts
    best_strategies = [v['strategy'] for v in best_per_count.values()]
    best_selections = [v['selection'] for v in best_per_count.values()]
    best_rhos = [v['feature_rho'] for v in best_per_count.values()]
    
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
    
    if len(set(best_rhos)) == 1:
        report_lines.append(f"✓ **Feature rho**: `{best_rhos[0]}` consistently outperforms")
    else:
        report_lines.append("ℹ **Feature rho**: Results vary by TX count")
    
    report_lines.append("")
    
    # Write report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, 'w') as f:
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
    method_comparison = results_df.groupby('selection_method')['ale'].agg(['mean', 'std']).reset_index()
    bars = ax.bar(method_comparison['selection_method'], method_comparison['mean'], 
                  yerr=method_comparison['std'], color=['#2ecc71', '#e74c3c'], capsize=5)
    ax.set_xlabel('Selection Method')
    ax.set_ylabel('Mean ALE (m)')
    ax.set_title('Selection Method Comparison')
    
    # Plot 1c: Feature rho comparison
    ax = axes[1, 0]
    rho_comparison = results_df.groupby('feature_rho')['ale'].agg(['mean', 'std']).reset_index()
    bars = ax.bar(rho_comparison['feature_rho'], rho_comparison['mean'], 
                  yerr=rho_comparison['std'], color=['#3498db', '#9b59b6'], capsize=5)
    ax.set_xlabel('Feature Rho Config')
    ax.set_ylabel('Mean ALE (m)')
    ax.set_title('Feature Rho Comparison')
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
    top_strategies = universal_summary.head(10)[['strategy', 'selection_method', 'feature_rho']].values.tolist()
    
    # Build heatmap data
    heatmap_data = []
    for tx_count in sorted(tx_count_summaries.keys()):
        row = []
        for strat, sel, rho in top_strategies:
            df = tx_count_summaries[tx_count]
            match = df[(df['strategy'] == strat) & (df['selection_method'] == sel) & (df['feature_rho'] == rho)]
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
        help='Propagation model (default: tirem)'
    )
    parser.add_argument(
        '--eta', type=float, default=0.01,
        help='Eta parameter for heteroscedastic whitening (default: 0.01)'
    )
    parser.add_argument(
        '--no-visualizations', action='store_true',
        help='Disable GLRT visualization generation'
    )
    
    args = parser.parse_args()
    
    # Parse TX counts filter
    tx_counts_filter = None
    if args.tx_counts:
        tx_counts_filter = [int(x.strip()) for x in args.tx_counts.split(',')]
    
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
    
    # Run comprehensive sweep
    results_df = run_comprehensive_sweep(
        grouped_dirs=grouped_dirs,
        config=config,
        map_data=map_data,
        all_tx_locations=all_tx_locations,
        output_dir=output_dir,
        test_mode=args.test,
        tx_counts_filter=tx_counts_filter,
        max_dirs_per_count=max_dirs,
        model_type=args.model_type,
        eta=args.eta,
        save_visualizations=not args.no_visualizations,
        verbose=True,
    )
    
    if len(results_df) == 0:
        print("\nNo results collected. Exiting.")
        return
    
    # Save raw results
    results_path = output_dir / 'all_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Raw results saved to: {results_path}")
    
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
    print(f"\nBest overall strategy: {best['strategy']} ({best['selection_method']}, {best['feature_rho']})")
    print(f"  Mean ALE: {best['ale_mean']:.2f} m")
    print(f"  Mean Pd: {best['pd_mean']*100:.1f}%")
    
    return results_df


if __name__ == "__main__":
    main()
