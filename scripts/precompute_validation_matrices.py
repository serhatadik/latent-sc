"""
Pre-compute TIREM validation propagation matrices for all TX combos.

This script creates the cached prop_matrix_val_*.npy files that the
comprehensive_parameter_sweep needs when --reconstruction-model tirem.
Run this ONCE before the sweep to avoid silent hangs during matrix computation.

Usage:
    python scripts/precompute_validation_matrices.py [--n-jobs N]
"""

import sys
import os
import hashlib
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import yaml

from src.utils import load_slc_map, get_sensor_locations_array, load_monitoring_locations
from src.evaluation.validation import ReconstructionValidator
from src.evaluation.reconstruction_validation import normalize_tx_id, get_validation_paths

KNOWN_TRANSMITTERS = ['mario', 'moran', 'guesthouse', 'ustar', 'wasatch']


def get_all_tx_combos(project_root: Path):
    """Discover all TX combos that have both config and data for validation."""
    combos = []
    config_dir = project_root / "config"
    for f in sorted(config_dir.glob("validation_*.yaml")):
        tx_id = f.stem.replace("validation_", "")
        data_dir = project_root / "data" / "processed" / f"validation_{tx_id}"
        avg_file = data_dir / f"validation_{tx_id}_avg_powers.npy"
        if data_dir.exists() and avg_file.exists():
            combos.append(tx_id)
    return combos


def check_cache_exists(val_points, map_shape, scale, model_type, model_config_path, cache_dir):
    """Replicate the validator's cache key logic to check if a matrix is cached."""
    hasher = hashlib.md5()
    hasher.update(val_points.tobytes())
    hasher.update(str(map_shape).encode('utf-8'))
    hasher.update(str(scale).encode('utf-8'))
    hasher.update(model_type.encode('utf-8'))

    if model_config_path:
        with open(model_config_path, 'rb') as f:
            hasher.update(f.read())

    cache_key = hasher.hexdigest()
    cache_file = Path(cache_dir) / f"prop_matrix_val_{cache_key}.npy"
    return cache_file.exists(), cache_file


def main():
    parser = argparse.ArgumentParser(description="Pre-compute TIREM validation matrices")
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs for TIREM computation (-1 = all cores)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config' / 'parameters.yaml'
    tirem_config_path = str(project_root / 'config' / 'tirem_parameters.yaml')
    cache_dir = str(project_root / 'data' / 'cache')

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    scale = config['spatial']['proxel_size']
    downsample_factor = config['spatial']['downsample_factor']

    # Load map
    print("Loading SLC map...")
    os.chdir(str(project_root))
    map_data = load_slc_map(map_folder_dir="./", downsample_factor=downsample_factor)
    print(f"  Map shape: {map_data['shape']}")
    print(f"  Scale: {scale} m/pixel")

    # Discover all TX combos with validation data
    all_combos = get_all_tx_combos(project_root)
    print(f"\nFound {len(all_combos)} TX combos with validation data.")

    # Check which are already cached
    cached = []
    uncached = []

    for tx_id in all_combos:
        transmitters = tx_id.split('_')
        val_config = project_root / "config" / f"validation_{tx_id}.yaml"

        # Load validation locations to get val_points (needed for cache key)
        locations_config = load_monitoring_locations(
            config_path=str(val_config),
            map_data=map_data
        )
        val_points = get_sensor_locations_array(locations_config)

        # Filter out-of-bounds
        height, width = map_data['shape']
        cols = val_points[:, 0]
        rows = val_points[:, 1]
        valid_mask = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        val_points = val_points[valid_mask]

        exists, cache_file = check_cache_exists(
            val_points, map_data['shape'], scale, 'tirem', tirem_config_path, cache_dir
        )

        if exists:
            cached.append((tx_id, len(val_points)))
        else:
            uncached.append((tx_id, len(val_points), str(val_config)))

    print(f"\n  Already cached: {len(cached)}")
    for tx_id, n_pts in cached:
        print(f"    {tx_id} ({n_pts} points)")

    print(f"\n  Need computing: {len(uncached)}")
    for tx_id, n_pts, _ in uncached:
        n_grid = np.prod(map_data['shape'])
        n_calls = n_pts * n_grid
        print(f"    {tx_id} ({n_pts} points, ~{n_calls:,.0f} TIREM calls)")

    if not uncached:
        print("\nAll validation matrices are cached. Nothing to do.")
        return

    total = len(uncached)
    print(f"\n{'='*60}")
    print(f"Starting computation of {total} TIREM validation matrices...")
    print(f"{'='*60}\n")

    for i, (tx_id, n_pts, val_config_path) in enumerate(uncached):
        print(f"[{i+1}/{total}] Computing: {tx_id} ({n_pts} validation points)")
        val_data_dir = str(project_root / "data" / "processed" / f"validation_{tx_id}")

        validator = ReconstructionValidator(
            map_data=map_data,
            validation_config_path=val_config_path,
            validation_data_dir=val_data_dir
        )

        # Load and filter
        file_prefix = f"validation_{tx_id}"
        validator.load_observed_data(file_prefix)
        validator.filter_out_of_bounds(verbose=True)

        # Compute (this is the expensive step)
        start = time.time()
        validator.get_propagation_matrix(
            model_type='tirem',
            model_config_path=tirem_config_path,
            scale=scale,
            cache_dir=cache_dir,
            n_jobs=args.n_jobs,
            verbose=True
        )
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s\n")

    print(f"\nAll {total} matrices computed and cached.")


if __name__ == '__main__':
    main()
