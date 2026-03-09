"""
Pre-compute and cache Sionna propagation matrices for all data directories.

This avoids repeated computation during ablation studies or parameter sweeps.
Each unique (sensor_locations, map_shape, scale, sionna_config) combination
gets cached to data/cache/sionna/.

Covers both:
  1. Localization matrices (30 sensors per seed/nloc)
  2. Validation matrices (unique validation point sets per TX combo)

Usage:
    python -m scripts.precompute_sionna_matrices --nloc 30
    python -m scripts.precompute_sionna_matrices --nloc 30 --max-dirs 5  # test with few dirs
    python -m scripts.precompute_sionna_matrices --nloc 30 --skip-localization  # validation only
    python -m scripts.precompute_sionna_matrices --nloc 30 --skip-validation    # localization only
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import yaml
import numpy as np

from src.utils import load_slc_map, load_monitoring_locations, get_sensor_locations_array
from scripts.sweep.discovery import discover_data_directories


def _check_sionna_cache(sensor_locations, map_shape, scale, model):
    """Check if a Sionna propagation matrix is already cached."""
    cache_params = {
        'sensor_locations': sensor_locations.tolist(),
        'map_shape': list(map_shape),
        'scale': float(scale),
        'sionna_config': model.config,
        'map_path': str(model.map_path)
    }
    cache_string = json.dumps(cache_params, sort_keys=True)
    cache_hash = hashlib.md5(cache_string.encode('utf-8')).hexdigest()
    cache_file = Path('data/cache/sionna') / f"sionna_prop_matrix_{cache_hash}.npy"
    return cache_file.exists()


def _compute_and_cache(label, sensor_locations, map_shape, scale, model):
    """Compute a propagation matrix (will be cached internally by SionnaModel)."""
    print(f"  Computing: {label} ({len(sensor_locations)} points)")
    try:
        model.compute_propagation_matrix(
            sensor_locations=sensor_locations,
            map_shape=map_shape,
            scale=scale,
            verbose=True,
        )
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Pre-compute Sionna propagation matrices')
    parser.add_argument('--nloc', type=int, default=None, help='Filter to specific num_locations')
    parser.add_argument('--tx-counts', type=str, default=None, help='Comma-separated TX counts (e.g., 1,2,3)')
    parser.add_argument('--max-dirs', type=int, default=None, help='Max directories to process')
    parser.add_argument('--config-path', type=str, default='config/sionna_parameters.yaml',
                        help='Path to sionna_parameters.yaml')
    parser.add_argument('--skip-localization', action='store_true',
                        help='Skip localization matrices (sensor layouts)')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip validation matrices')
    args = parser.parse_args()

    # Load project config
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Loading SLC map...")
    map_data = load_slc_map("./", downsample_factor=config['spatial']['downsample_factor'])
    map_shape = map_data['shape']
    scale = config['spatial']['proxel_size']
    print(f"  Map shape: {map_shape}, scale: {scale}")

    # Initialize Sionna model once
    from src.propagation import SionnaModel
    if SionnaModel is None:
        print("ERROR: Sionna is not installed. Cannot precompute matrices.")
        return

    print(f"Initializing SionnaModel from {args.config_path}...")
    model = SionnaModel(args.config_path)

    start_time = time.time()
    computed = 0
    skipped_cached = 0

    # ---- Part 1: Localization matrices (per sensor layout) ----
    if not args.skip_localization:
        print("\n" + "=" * 60)
        print("LOCALIZATION MATRICES")
        print("=" * 60)

        base_dir = Path('data/processed')
        grouped_dirs = discover_data_directories(base_dir)

        if args.tx_counts:
            tx_filter = [int(x.strip()) for x in args.tx_counts.split(',')]
            grouped_dirs = {k: v for k, v in grouped_dirs.items() if k in tx_filter}
        if args.nloc is not None:
            grouped_dirs = {
                k: [d for d in v if d.get('num_locations') == args.nloc]
                for k, v in grouped_dirs.items()
            }
            grouped_dirs = {k: v for k, v in grouped_dirs.items() if v}

        all_dirs = []
        for tc in sorted(grouped_dirs.keys()):
            all_dirs.extend(grouped_dirs[tc])
        if args.max_dirs:
            all_dirs = all_dirs[:args.max_dirs]

        # Deduplicate sensor layouts
        unique_layouts = {}
        for d in all_dirs:
            tx_underscore = "_".join(d['transmitters'])
            config_id = tx_underscore
            if d.get('num_locations') is not None:
                config_id = f"{config_id}_nloc{d['num_locations']}"
            if d.get('seed') is not None:
                config_id = f"{config_id}_seed_{d['seed']}"
            loc_config_path = f'config/monitoring_locations_{config_id}.yaml'

            if not Path(loc_config_path).exists():
                continue

            locations_config = load_monitoring_locations(config_path=loc_config_path, map_data=map_data)
            sensor_locations = get_sensor_locations_array(locations_config)

            sensor_key = sensor_locations.tobytes()
            if sensor_key not in unique_layouts:
                unique_layouts[sensor_key] = (sensor_locations, d['name'])

        print(f"  {len(all_dirs)} directories -> {len(unique_layouts)} unique sensor layouts")

        for i, (sensor_key, (sensor_locations, dir_name)) in enumerate(unique_layouts.items()):
            if _check_sionna_cache(sensor_locations, map_shape, scale, model):
                skipped_cached += 1
                continue

            elapsed = time.time() - start_time
            print(f"\n[{i+1}/{len(unique_layouts)}] {dir_name} "
                  f"({computed} computed, {skipped_cached} cached, {elapsed/60:.1f}min)")
            if _compute_and_cache(dir_name, sensor_locations, map_shape, scale, model):
                computed += 1

        print(f"\nLocalization: {computed} computed, {skipped_cached} cached")

    # ---- Part 2: Validation matrices (per unique val_points set) ----
    if not args.skip_validation:
        print("\n" + "=" * 60)
        print("VALIDATION MATRICES")
        print("=" * 60)

        val_computed = 0
        val_cached = 0

        # Find all unique validation point sets
        val_configs = sorted(Path('config').glob('validation_*.yaml'))
        unique_val_layouts = {}

        for vc in val_configs:
            # Check that corresponding data dir exists
            val_id = vc.stem.replace('validation_', '')
            val_data_dir = Path('data/processed') / f"validation_{val_id}"
            if not val_data_dir.exists():
                continue

            locs = load_monitoring_locations(str(vc), map_data)
            val_points = get_sensor_locations_array(locs)

            val_key = val_points.tobytes()
            if val_key not in unique_val_layouts:
                unique_val_layouts[val_key] = (val_points, vc.name)

        print(f"  {len(val_configs)} validation configs -> {len(unique_val_layouts)} unique point sets")

        for i, (val_key, (val_points, config_name)) in enumerate(unique_val_layouts.items()):
            if _check_sionna_cache(val_points, map_shape, scale, model):
                val_cached += 1
                print(f"  [{i+1}/{len(unique_val_layouts)}] {config_name}: already cached")
                continue

            elapsed = time.time() - start_time
            print(f"\n[{i+1}/{len(unique_val_layouts)}] {config_name} "
                  f"({len(val_points)} points, {elapsed/60:.1f}min)")
            if _compute_and_cache(config_name, val_points, map_shape, scale, model):
                val_computed += 1

        computed += val_computed
        skipped_cached += val_cached
        print(f"\nValidation: {val_computed} computed, {val_cached} cached")

    elapsed_total = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed_total/60:.1f} minutes")
    print(f"  Total computed: {computed}")
    print(f"  Total cached: {skipped_cached}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
