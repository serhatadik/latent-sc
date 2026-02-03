"""
Process Raw IQ Data to Validation Set Configuration.

This script extracts geographically widespread received signal strength values
for evaluation of reconstruction performance. Unlike the monitoring script which
selects a small subset of locations, this script extracts ALL available aggregated
locations to serve as a dense validation set.

Usage:
    python scripts/process_raw_data_to_validation.py \
        --input-dir "C:/Users/serha/raw_data/stat_rot/stat/" \
        --transmitter mario \
        --output-yaml "config/validation_mario.yaml"

Features:
    - Processes raw IQ samples from any directory under raw_data/
    - Extracts power for a specific transmitter's frequency band
    - Automatically matches with GPS coordinates
    - Aggregates measurements by receiver location (deduplication)
    - Exports ALL valid locations as a validation dataset
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
import argparse
from typing import List, Dict, Optional
import pandas as pd

from src.data_processing.iq_processor import (
    load_iq_samples_from_directories,
    load_gps_from_csv,
    match_power_with_gps,
    aggregate_measurements_by_location,
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



def main():
    parser = argparse.ArgumentParser(
        description="Process raw IQ data to generate validation set",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--input-dir", type=Path, nargs='+', required=True, help="Input raw data directories (one or more)")
    parser.add_argument("--transmitter", type=str, required=True, help="Transmitter name(s)")
    
    # Optional args
    parser.add_argument("--gps-dir", type=Path, default=None)
    parser.add_argument("--output-yaml", type=Path, default=None)
    parser.add_argument("--output-data", type=Path, default=None)
    parser.add_argument("--gps-pattern", type=str, default=None)
    
    # Validation specific defaults
    parser.add_argument("--dedup-threshold", type=float, default=20.0, help="Spatial aggregation radius (meters)")
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum samples required to include a location")
    parser.add_argument("--samples-to-skip", type=int, default=2)
    parser.add_argument("--time-offset", type=int, default=-6)

    args = parser.parse_args()

    transmitter_names = [tx.strip() for tx in args.transmitter.split(',')]
    
    # Transmitter Set ID
    if len(transmitter_names) == 1:
        transmitter_id = transmitter_names[0]
    else:
        transmitter_id = "_".join(transmitter_names)

    # Defaults
    if args.output_yaml is None:
        args.output_yaml = Path(f"config/validation_{transmitter_id}.yaml")
    if args.output_data is None:
        args.output_data = Path(f"data/processed/validation_{transmitter_id}/")
    
    # Auto-detect GPS
    if args.gps_dir is None:
        raw_data_root = Path("C:/Users/serha/raw_data")
        args.gps_dir = auto_detect_gps_directory(args.input_dir, raw_data_root)
        if args.gps_dir is None:
            print("Error: Could not auto-detect GPS dir")
            return 1
            
    if args.gps_pattern is None:
        # Heuristic: if strictly stat_rot usage, could use specific patterns, 
        # but for general validation with walking/driving, *.txt is safer.
        is_stat_rot = any('stat_rot' in str(d) for d in args.input_dir)
        is_walking = any('walking' in str(d) for d in args.input_dir)
        
        if is_stat_rot and not is_walking:
             # Logic from before (check for stat vs rot folders if separable, but here we likely mix)
             args.gps_pattern = "*.txt" 
        else:
             args.gps_pattern = "*.txt"

    print("="*80)
    print("VALIDATION SET GENERATOR")
    print("="*80)
    print(f"Transmitters: {transmitter_names}")
    print(f"Inputs: {[str(d) for d in args.input_dir]}")
    print(f"GPS: {args.gps_dir} ({args.gps_pattern})")
    
    # 1. Find Samples
    sample_dirs = find_iq_sample_directories(args.input_dir)
    if not sample_dirs:
        print("No sample directories found.")
        return 1
    print(f"Found {len(sample_dirs)} sample directories")


    # 2. Load IQ (Lazy/Metadata only mostly? No, iq_processor loads full samples. Optimization: load all?)
    # Since we need power for ALL locations, we cannot skip the FFT/Power calc for subsets.
    # We must calculate power for everything that matches GPS.
    # To avoid OOM for huge datasets, we might want to do this iteratively, but let's stick to the existing flow for now.
    print("Loading IQ samples...")
    iq_data = load_iq_samples_from_directories(sample_dirs, args.samples_to_skip)
    
    # 3. Load GPS
    gps_files = find_gps_files(args.gps_dir, args.gps_pattern)
    gps_coords = load_gps_from_csv(gps_files, args.time_offset)
    
    # 4. Match Power & GPS
    # This computes power for ALL timestamps that satisfy GPS matching
    print("Matching GPS and computing power (this may take a while)...")
    measurements = match_power_with_gps(
        iq_data,
        gps_coords,
        transmitter_names,
        compute_power=True 
    )
    
    # 5. Aggregate
    print("Aggregating spatially...")
    final_locations = aggregate_measurements_by_location(
        measurements,
        args.dedup_threshold,
        args.min_samples,
        include_timestamps=False
    )
    
    if not final_locations:
        print("No valid locations found after aggregation.")
        return 1

    print(f"Generated {len(final_locations)} unique validation points.")

    # 6. Save
    generate_validation_yaml(final_locations, transmitter_names, args.output_yaml)
    save_processed_data(final_locations, transmitter_names, args.output_data)
    
    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
