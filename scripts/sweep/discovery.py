"""Data directory discovery, parsing, TIREM cache checks, and sigma strategies."""

import re
import numpy as np
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .constants import KNOWN_TRANSMITTERS, TIREM_CACHE_DIR


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
