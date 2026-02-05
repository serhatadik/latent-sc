"""
Process Raw IQ Data to Validation Set Configuration.

This script extracts geographically widespread received signal strength values
for evaluation of reconstruction performance. Unlike the monitoring script which
selects a small subset of locations, this script extracts ALL available aggregated
locations to serve as a dense validation set.

Usage:
    # Process a specific transmitter combination:
    python scripts/process_raw_data_to_validation.py \
        --input-dir "C:/Users/serha/raw_data/stat_rot/stat/" \
        --transmitter mario \
        --output-yaml "config/validation_mario.yaml"

    # Process ALL remaining transmitter combinations (recommended):
    python scripts/process_raw_data_to_validation.py \
        --input-dir "C:/Users/serha/raw_data/stat_rot/stat/" \
                    "C:/Users/serha/raw_data/walking" \
                    "C:/Users/serha/raw_data/driving"

    This will automatically:
    - Generate all 31 unique combinations of 5 transmitters
    - Skip combinations that already have validation data
    - Process all remaining combinations in one run (amortizing data loading cost)

Features:
    - Processes raw IQ samples from any directory under raw_data/
    - Extracts power for a specific transmitter's frequency band
    - Automatically matches with GPS coordinates
    - Aggregates measurements by receiver location (deduplication)
    - Exports ALL valid locations as a validation dataset
    - Batch processing of multiple transmitter combinations
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
import argparse
from typing import List, Dict, Optional, Tuple
from itertools import combinations, permutations
import pandas as pd

# All available transmitters for batch processing
ALL_TRANSMITTERS = ['mario', 'moran', 'ustar', 'guesthouse', 'wasatch']


def get_all_transmitter_combinations() -> List[List[str]]:
    """Generate all unique combinations of transmitters (1 to 5 at a time)."""
    all_combos = []
    for r in range(1, len(ALL_TRANSMITTERS) + 1):
        for combo in combinations(ALL_TRANSMITTERS, r):
            all_combos.append(list(combo))
    return all_combos


def get_combo_id(transmitter_names: List[str]) -> str:
    """Get a consistent ID for a transmitter combination (sorted, underscore-separated)."""
    return "_".join(sorted(transmitter_names))


def check_existing_validation(combo_id: str, config_dir: Path, data_dir: Path, transmitter_names: List[str]) -> bool:
    """
    Check if validation data already exists for this combination.

    Handles both sorted naming (new convention) and original ordering (legacy files).
    For example, detects both 'mario_guesthouse' and 'guesthouse_mario'.
    """
    # Check sorted naming (new convention)
    yaml_path = config_dir / f"validation_{combo_id}.yaml"
    data_path = data_dir / f"validation_{combo_id}"
    if yaml_path.exists() or data_path.exists():
        return True

    # Check original ordering permutations for legacy compatibility
    for perm in permutations(transmitter_names):
        legacy_id = "_".join(perm)
        if legacy_id == combo_id:
            continue  # Already checked above
        yaml_path = config_dir / f"validation_{legacy_id}.yaml"
        data_path = data_dir / f"validation_{legacy_id}"
        if yaml_path.exists() or data_path.exists():
            return True

    return False


def get_pending_combinations(config_dir: Path, data_dir: Path) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Get lists of pending and completed transmitter combinations.

    Returns:
        (pending_combos, completed_combos)
    """
    all_combos = get_all_transmitter_combinations()
    pending = []
    completed = []

    for combo in all_combos:
        combo_id = get_combo_id(combo)
        if check_existing_validation(combo_id, config_dir, data_dir, combo):
            completed.append(combo)
        else:
            pending.append(combo)

    return pending, completed

from src.data_processing.iq_processor import (
    load_iq_samples_from_directories,
    load_gps_from_csv,
    match_power_with_gps,
    aggregate_measurements_by_location,
    precompute_psds_with_gps_matching,
    extract_power_for_combination,
    TRANSMITTER_TO_CHANNEL,
    RF_CHANNELS
)



def find_iq_sample_directories(input_dirs: List[Path], pattern: str = "samples_*") -> List[Path]:
    """Find all IQ sample directories recursively in all input directories."""
    sample_dirs = []
    for input_dir in input_dirs:
        if input_dir.is_dir():
            for item in input_dir.rglob(pattern):
                if item.is_dir():
                    sample_dirs.append(item)
    return sorted(sample_dirs)


def auto_detect_gps_directory(input_dirs: List[Path], raw_data_root: Path) -> Optional[Path]:
    """Auto-detect GPS data directory."""
    # If any input suggests stat_rot, try that first? 
    # Actually, if we have multiple diverse inputs (walking, driving), 
    # we should probably prefer 'all_gps_data' if available.
    
    # Check for general gps_data first if we have multiple inputs or non-stat_rot inputs
    gps_dir_all = raw_data_root / "gps_data" / "all_gps_data"
    
    # Check if we are strictly stat_rot
    is_only_stat_rot = all('stat_rot' in str(d) for d in input_dirs)
    
    if is_only_stat_rot:
        gps_dir_stat = raw_data_root / "gps_data" / "stat_rot"
        if gps_dir_stat.exists():
            return gps_dir_stat

    if gps_dir_all.exists():
        return gps_dir_all
        
    return None


def find_gps_files(gps_dir: Path, file_pattern: str = "*.txt") -> List[Path]:
    """Find GPS CSV files."""
    if not gps_dir.exists():
        return []
    return sorted(list(gps_dir.glob(file_pattern)))


def generate_validation_yaml(
    locations: List[Dict],
    transmitter_names: List[str],
    output_path: Path,
    utm_zone: int = 12,
    northern_hemisphere: bool = True
) -> None:
    """Generate validation locations YAML file."""
    yaml_data = {
        'utm_zone': utm_zone,
        'northern_hemisphere': northern_hemisphere,
        'data_points': []
    }

    # Header info
    if len(transmitter_names) == 1:
        tx_info = f"# Transmitter: {transmitter_names[0]}\n"
    else:
        tx_info = f"# Transmitters: {', '.join(transmitter_names)}\n"

    header_comment = f"""# Validation Set Locations
# Auto-generated dense validation set
# Contains {len(locations)} widespread locations
{tx_info}# Coordinates in latitude/longitude (WGS84)
"""

    for i, loc in enumerate(locations):
        # Use simple numeric indexing for validation if names are generic
        name = loc['name'] if loc['name'] else f"val_{i:04d}"
        
        data_point = {
            'name': name,
            'longitude': loc['longitude'],
            'latitude': loc['latitude']
        }
        yaml_data['data_points'].append(data_point)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(header_comment)
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"\n[OK] Generated YAML config: {output_path}")
    print(f"  Contains {len(locations)} validation locations")


def save_processed_data(
    locations: List[Dict],
    transmitter_names: List[str],
    output_dir: Path
) -> None:
    """Save processed measurement data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(transmitter_names) == 1:
        file_prefix = transmitter_names[0]
    else:
        file_prefix = "_".join(transmitter_names)

    # Naming convention: add 'validation_' prefix to distinguish from monitoring data
    file_prefix = f"validation_{file_prefix}"

    names = [loc['name'] for loc in locations]
    lats = np.array([loc['latitude'] for loc in locations])
    lons = np.array([loc['longitude'] for loc in locations])
    powers = np.array([loc['avg_power'] for loc in locations])
    stds = np.array([loc['std_power'] for loc in locations])
    counts = np.array([loc['num_samples'] for loc in locations])

    # Save arrays
    np.save(output_dir / f"{file_prefix}_names.npy", names)
    np.save(output_dir / f"{file_prefix}_latitudes.npy", lats)
    np.save(output_dir / f"{file_prefix}_longitudes.npy", lons)
    np.save(output_dir / f"{file_prefix}_avg_powers.npy", powers)
    np.save(output_dir / f"{file_prefix}_std_powers.npy", stds)
    np.save(output_dir / f"{file_prefix}_sample_counts.npy", counts)

    # Save summary CSV
    df = pd.DataFrame({
        'Name': names,
        'Latitude': lats,
        'Longitude': lons,
        'Avg_Power_dB': powers,
        'Std_Power_dB': stds,
        'Num_Samples': counts
    })
    df.to_csv(output_dir / f"{file_prefix}_summary.csv", index=False)

    print(f"\n[OK] Saved validation data to: {output_dir}")
    print(f"  Files: {file_prefix}_*.npy and {file_prefix}_summary.csv")



def process_single_combination(
    transmitter_names: List[str],
    iq_data: Dict,
    gps_coords: Dict,
    output_yaml: Path,
    output_data: Path,
    dedup_threshold: float,
    min_samples: int
) -> bool:
    """
    Process a single transmitter combination (non-optimized path).

    Returns True on success, False on failure.
    """
    combo_id = get_combo_id(transmitter_names)
    print(f"\n{'='*60}")
    print(f"Processing: {combo_id}")
    print(f"{'='*60}")

    try:
        # Match Power & GPS
        print("  Matching GPS and computing power...")
        measurements = match_power_with_gps(
            iq_data,
            gps_coords,
            transmitter_names,
            compute_power=True
        )

        if not measurements:
            print(f"  [WARN] No measurements found for {combo_id}")
            return False

        # Aggregate
        print("  Aggregating spatially...")
        final_locations = aggregate_measurements_by_location(
            measurements,
            dedup_threshold,
            min_samples,
            include_timestamps=False
        )

        if not final_locations:
            print(f"  [WARN] No valid locations for {combo_id} after aggregation.")
            return False

        print(f"  Generated {len(final_locations)} unique validation points.")

        # Save
        generate_validation_yaml(final_locations, transmitter_names, output_yaml)
        save_processed_data(final_locations, transmitter_names, output_data)

        print(f"  [OK] Completed {combo_id}")
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to process {combo_id}: {e}")
        return False


def process_combination_from_cache(
    transmitter_names: List[str],
    frequencies: np.ndarray,
    cached_psd_data: List[Dict],
    output_yaml: Path,
    output_data: Path,
    dedup_threshold: float,
    min_samples: int
) -> bool:
    """
    Process a single transmitter combination using precomputed PSDs (optimized path).

    This is much faster than process_single_combination() because it skips FFT
    computation and only does frequency band extraction.

    Returns True on success, False on failure.
    """
    combo_id = get_combo_id(transmitter_names)
    print(f"\n{'='*60}")
    print(f"Processing (from cache): {combo_id}")
    print(f"{'='*60}")

    try:
        # Extract power from cached PSDs (fast - no FFT)
        print("  Extracting power from cached PSDs...")
        measurements = extract_power_for_combination(
            frequencies,
            cached_psd_data,
            transmitter_names
        )

        if not measurements:
            print(f"  [WARN] No measurements found for {combo_id}")
            return False

        print(f"  Extracted {len(measurements)} measurements")

        # Aggregate
        print("  Aggregating spatially...")
        final_locations = aggregate_measurements_by_location(
            measurements,
            dedup_threshold,
            min_samples,
            include_timestamps=False
        )

        if not final_locations:
            print(f"  [WARN] No valid locations for {combo_id} after aggregation.")
            return False

        print(f"  Generated {len(final_locations)} unique validation points.")

        # Save
        generate_validation_yaml(final_locations, transmitter_names, output_yaml)
        save_processed_data(final_locations, transmitter_names, output_data)

        print(f"  [OK] Completed {combo_id}")
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to process {combo_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Process raw IQ data to generate validation set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a specific combination:
  python scripts/process_raw_data_to_validation.py \\
      --input-dir "C:/Users/serha/raw_data/stat_rot/stat/" \\
      --transmitter mario,moran

  # Process ALL remaining combinations (batch mode):
  python scripts/process_raw_data_to_validation.py \\
      --input-dir "C:/Users/serha/raw_data/stat_rot/stat/" \\
                  "C:/Users/serha/raw_data/walking" \\
                  "C:/Users/serha/raw_data/driving"
"""
    )

    parser.add_argument("--input-dir", type=Path, nargs='+', required=True, help="Input raw data directories (one or more)")
    parser.add_argument("--transmitter", type=str, default=None, help="Transmitter name(s), comma-separated. If omitted, processes all remaining combinations.")

    # Optional args
    parser.add_argument("--gps-dir", type=Path, default=None)
    parser.add_argument("--output-yaml", type=Path, default=None, help="Output YAML path (only for single combination mode)")
    parser.add_argument("--output-data", type=Path, default=None, help="Output data dir (only for single combination mode)")
    parser.add_argument("--gps-pattern", type=str, default=None)

    # Validation specific defaults
    parser.add_argument("--dedup-threshold", type=float, default=20.0, help="Spatial aggregation radius (meters)")
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum samples required to include a location")
    parser.add_argument("--samples-to-skip", type=int, default=2)
    parser.add_argument("--time-offset", type=int, default=-6)

    # Batch mode options
    parser.add_argument("--force", action="store_true", help="Re-process even if validation data already exists")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without actually processing")

    args = parser.parse_args()

    # Base directories for checking existing data
    config_dir = Path("config")
    data_base_dir = Path("data/processed")

    # Determine which combinations to process
    if args.transmitter is not None:
        # Single combination mode (original behavior)
        transmitter_names = [tx.strip() for tx in args.transmitter.split(',')]
        combos_to_process = [transmitter_names]
        batch_mode = False
    else:
        # Batch mode: process all pending combinations
        pending, completed = get_pending_combinations(config_dir, data_base_dir)

        print("="*80)
        print("BATCH MODE: Processing all transmitter combinations")
        print("="*80)
        print(f"Total possible combinations: {len(pending) + len(completed)}")
        print(f"Already completed: {len(completed)}")
        print(f"Pending: {len(pending)}")

        if completed:
            print("\nCompleted combinations:")
            for combo in completed:
                print(f"  - {get_combo_id(combo)}")

        if pending:
            print("\nPending combinations:")
            for combo in pending:
                print(f"  - {get_combo_id(combo)}")

        if args.force:
            combos_to_process = get_all_transmitter_combinations()
            print(f"\n[FORCE] Will re-process all {len(combos_to_process)} combinations")
        else:
            combos_to_process = pending

        if not combos_to_process:
            print("\nAll combinations already processed. Use --force to re-process.")
            return 0

        batch_mode = True

    if args.dry_run:
        print("\n[DRY RUN] Would process the following combinations:")
        for combo in combos_to_process:
            print(f"  - {get_combo_id(combo)}")
        return 0

    # Auto-detect GPS
    if args.gps_dir is None:
        raw_data_root = Path("C:/Users/serha/raw_data")
        args.gps_dir = auto_detect_gps_directory(args.input_dir, raw_data_root)
        if args.gps_dir is None:
            print("Error: Could not auto-detect GPS dir")
            return 1

    if args.gps_pattern is None:
        args.gps_pattern = "*.txt"

    print("\n" + "="*80)
    print("VALIDATION SET GENERATOR")
    print("="*80)
    print(f"Combinations to process: {len(combos_to_process)}")
    print(f"Inputs: {[str(d) for d in args.input_dir]}")
    print(f"GPS: {args.gps_dir} ({args.gps_pattern})")

    # 1. Find Samples
    sample_dirs = find_iq_sample_directories(args.input_dir)
    if not sample_dirs:
        print("No sample directories found.")
        return 1
    print(f"Found {len(sample_dirs)} sample directories")

    # 2. Load IQ data ONCE (the expensive step)
    print("\nLoading IQ samples (one-time cost)...")
    iq_data = load_iq_samples_from_directories(sample_dirs, args.samples_to_skip)
    print(f"Loaded {len(iq_data)} IQ sample timestamps")

    # 3. Load GPS ONCE
    print("Loading GPS data (one-time cost)...")
    gps_files = find_gps_files(args.gps_dir, args.gps_pattern)
    gps_coords = load_gps_from_csv(gps_files, args.time_offset)
    print(f"Loaded {len(gps_coords)} GPS timestamps")

    # 4. Decide processing strategy
    use_optimized_path = len(combos_to_process) > 1

    if use_optimized_path:
        # OPTIMIZED PATH: Precompute PSDs once, then extract for each combination
        print("\n" + "="*80)
        print("OPTIMIZED MODE: Precomputing PSDs (FFT computed once per timestamp)")
        print("="*80)

        frequencies, cached_psd_data = precompute_psds_with_gps_matching(
            iq_data,
            gps_coords,
            progress_interval=100,
            time_tolerance_seconds=10
        )

        if frequencies is None or not cached_psd_data:
            print("Error: Failed to precompute PSDs")
            return 1

        print(f"\nCached {len(cached_psd_data)} PSDs in memory")
        # Estimate memory usage
        if cached_psd_data:
            psd_size = cached_psd_data[0]['psd_linear'].nbytes
            total_mb = (psd_size * len(cached_psd_data)) / (1024 * 1024)
            print(f"Estimated PSD cache size: {total_mb:.1f} MB")
        # Note: IQ samples are freed incrementally during precomputation
        # to avoid peak memory of holding both IQ data and cached PSDs

    # 5. Process each combination
    success_count = 0
    fail_count = 0

    for combo in combos_to_process:
        combo_id = get_combo_id(combo)

        # Determine output paths
        if batch_mode or args.output_yaml is None:
            output_yaml = config_dir / f"validation_{combo_id}.yaml"
        else:
            output_yaml = args.output_yaml

        if batch_mode or args.output_data is None:
            output_data = data_base_dir / f"validation_{combo_id}"
        else:
            output_data = args.output_data

        if use_optimized_path:
            # Use cached PSDs (fast path - no FFT)
            success = process_combination_from_cache(
                combo,
                frequencies,
                cached_psd_data,
                output_yaml,
                output_data,
                args.dedup_threshold,
                args.min_samples
            )
        else:
            # Single combination - use original path
            success = process_single_combination(
                combo,
                iq_data,
                gps_coords,
                output_yaml,
                output_data,
                args.dedup_threshold,
                args.min_samples
            )

        if success:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total: {success_count + fail_count}")

    if fail_count > 0:
        return 1

    print("\nDone.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
