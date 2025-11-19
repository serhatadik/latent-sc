"""
Process Raw IQ Data to Monitoring Locations Configuration.

This script provides a general-purpose pipeline to convert raw IQ samples
into monitoring location configurations compatible with the paper_reproduction notebook.

Usage:
    python scripts/process_raw_data_to_monitoring.py \
        --input-dir "C:/Users/serha/raw_data/stat_rot/stat/" \
        --transmitter mario \
        --num-locations 10 \
        --output-yaml "config/monitoring_locations_custom.yaml"

Features:
    - Processes raw IQ samples from any directory under raw_data/
    - Extracts power for a specific transmitter's frequency band
    - Automatically matches with GPS coordinates
    - Aggregates measurements by receiver location
    - Generates monitoring_locations.yaml compatible with existing workflow
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
import argparse
from typing import List, Dict, Optional

from src.data_processing.iq_processor import (
    load_iq_samples_from_directories,
    load_gps_from_csv,
    match_power_with_gps,
    aggregate_measurements_by_location,
    TRANSMITTER_TO_CHANNEL,
    RF_CHANNELS
)


def find_iq_sample_directories(input_dir: Path, pattern: str = "samples_*") -> List[Path]:
    """
    Recursively find all IQ sample directories under the input directory.

    This function searches through all subdirectories to find sample folders,
    allowing for nested directory structures (e.g., walking/serhat_walking_27jun/samples_*/).

    Parameters
    ----------
    input_dir : Path
        Root directory to search recursively
    pattern : str, optional
        Pattern to match sample directories (default: "samples_*")

    Returns
    -------
    List[Path]
        List of paths to sample directories, sorted alphabetically
    """
    sample_dirs = []

    if input_dir.is_dir():
        # Use rglob for recursive search through all subdirectories
        for item in input_dir.rglob(pattern):
            if item.is_dir():
                sample_dirs.append(item)

    return sorted(sample_dirs)


def auto_detect_gps_directory(input_dir: Path, raw_data_root: Path) -> Optional[Path]:
    """
    Auto-detect GPS data directory based on input directory structure.

    Parameters
    ----------
    input_dir : Path
        Input IQ sample directory
    raw_data_root : Path
        Root raw_data directory

    Returns
    -------
    Optional[Path]
        Path to GPS data directory, or None if not found
    """
    # Check if input_dir is under stat_rot
    if 'stat_rot' in str(input_dir):
        gps_dir = raw_data_root / "gps_data" / "stat_rot"
        if gps_dir.exists():
            return gps_dir

    # Check for general gps_data
    gps_dir = raw_data_root / "gps_data" / "all_gps_data"
    if gps_dir.exists():
        return gps_dir

    return None


def find_gps_files(gps_dir: Path, file_pattern: str = "*.txt") -> List[Path]:
    """
    Find GPS CSV files in directory.

    Parameters
    ----------
    gps_dir : Path
        GPS data directory
    file_pattern : str, optional
        File pattern to match (default: "*.txt")

    Returns
    -------
    List[Path]
        List of GPS file paths
    """
    if not gps_dir.exists():
        return []

    if file_pattern == "*.txt":
        # Get all .txt files
        gps_files = list(gps_dir.glob("*.txt"))
    else:
        gps_files = list(gps_dir.glob(file_pattern))

    return sorted(gps_files)


def generate_monitoring_yaml(
    locations: List[Dict],
    transmitter_name: str,
    output_path: Path,
    utm_zone: int = 12,
    northern_hemisphere: bool = True
) -> None:
    """
    Generate monitoring_locations.yaml file.

    Parameters
    ----------
    locations : List[Dict]
        List of location dictionaries from aggregate_measurements_by_location()
    transmitter_name : str
        Name of the transmitter used
    output_path : Path
        Output YAML file path
    utm_zone : int, optional
        UTM zone number (default: 12 for Salt Lake City)
    northern_hemisphere : bool, optional
        True if Northern hemisphere (default: True)
    """
    # Create YAML structure
    yaml_data = {
        'utm_zone': utm_zone,
        'northern_hemisphere': northern_hemisphere,
        'data_points': []
    }

    # Add comment header
    header_comment = f"""# Monitoring Station Locations
# Auto-generated from raw IQ data processing
# Transmitter: {transmitter_name}
# RF Channel: {TRANSMITTER_TO_CHANNEL[transmitter_name]} {RF_CHANNELS[TRANSMITTER_TO_CHANNEL[transmitter_name]]}
# Coordinates in latitude/longitude (WGS84)
# Format: [longitude, latitude] (negative longitude for West)
# These will be automatically converted to pixel coordinates using the SLC map

"""

    # Add data points
    for loc in locations:
        data_point = {
            'name': loc['name'],
            'longitude': loc['longitude'],
            'latitude': loc['latitude']
        }
        yaml_data['data_points'].append(data_point)

    # Write YAML file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        # Write header comment
        f.write(header_comment)
        # Write YAML data
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"\n[OK] Generated YAML config: {output_path}")
    print(f"  Contains {len(locations)} monitoring locations")


def save_processed_data(
    locations: List[Dict],
    transmitter_name: str,
    output_dir: Path
) -> None:
    """
    Save processed measurement data.

    Parameters
    ----------
    locations : List[Dict]
        List of location dictionaries
    transmitter_name : str
        Name of the transmitter
    output_dir : Path
        Output directory for processed data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as numpy arrays
    names = [loc['name'] for loc in locations]
    lats = np.array([loc['latitude'] for loc in locations])
    lons = np.array([loc['longitude'] for loc in locations])
    powers = np.array([loc['avg_power'] for loc in locations])
    stds = np.array([loc['std_power'] for loc in locations])
    counts = np.array([loc['num_samples'] for loc in locations])

    np.save(output_dir / f"{transmitter_name}_names.npy", names)
    np.save(output_dir / f"{transmitter_name}_latitudes.npy", lats)
    np.save(output_dir / f"{transmitter_name}_longitudes.npy", lons)
    np.save(output_dir / f"{transmitter_name}_avg_powers.npy", powers)
    np.save(output_dir / f"{transmitter_name}_std_powers.npy", stds)
    np.save(output_dir / f"{transmitter_name}_sample_counts.npy", counts)

    # Save as CSV for easy viewing
    import pandas as pd
    df = pd.DataFrame({
        'Name': names,
        'Latitude': lats,
        'Longitude': lons,
        'Avg_Power_dB': powers,
        'Std_Power_dB': stds,
        'Num_Samples': counts
    })
    df.to_csv(output_dir / f"{transmitter_name}_summary.csv", index=False)

    print(f"\n[OK] Saved processed data to: {output_dir}")
    print(f"  Files: {transmitter_name}_*.npy and {transmitter_name}_summary.csv")


def print_summary(locations: List[Dict], transmitter_name: str) -> None:
    """
    Print summary of processed locations.

    Parameters
    ----------
    locations : List[Dict]
        List of location dictionaries
    transmitter_name : str
        Name of the transmitter
    """
    channel = TRANSMITTER_TO_CHANNEL[transmitter_name]
    freq_range = RF_CHANNELS[channel]

    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Transmitter: {transmitter_name}")
    print(f"RF Channel: {channel}")
    print(f"Frequency Range: {freq_range[0]/1e6:.3f} - {freq_range[1]/1e6:.3f} MHz")
    print(f"Number of Locations: {len(locations)}")
    print("\n" + "-"*80)
    print(f"{'Name':<15} {'Latitude':>10} {'Longitude':>11} {'Avg Power':>10} {'Std':>8} {'Samples':>8}")
    print("-"*80)

    for loc in locations:
        print(f"{loc['name']:<15} {loc['latitude']:>10.5f} {loc['longitude']:>11.5f} "
              f"{loc['avg_power']:>10.2f} {loc['std_power']:>8.2f} {loc['num_samples']:>8d}")

    print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process raw IQ data to generate monitoring locations configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process stationary data for Mario transmitter
  python scripts/process_raw_data_to_monitoring.py \\
      --input-dir "C:/Users/serha/raw_data/stat_rot/stat/" \\
      --transmitter mario \\
      --num-locations 10 \\
      --output-yaml "config/monitoring_locations_mario.yaml"

  # Process with custom GPS directory
  python scripts/process_raw_data_to_monitoring.py \\
      --input-dir "C:/Users/serha/raw_data/stat_rot/stat/" \\
      --gps-dir "C:/Users/serha/raw_data/gps_data/stat_rot/" \\
      --transmitter moran \\
      --num-locations 5

Available transmitters: ebc, ustar, guesthouse, mario, moran, wasatch
  Note: ebc and ustar share TX1 frequency band but are date-separated:
    - ebc: Data from June 27, 2023 and earlier
    - ustar: Data from June 28, 2023 and later
        """
    )

    # Required arguments
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing raw IQ samples (e.g., 'C:/Users/serha/raw_data/stat_rot/stat/')"
    )
    parser.add_argument(
        "--transmitter",
        type=str,
        required=True,
        choices=list(TRANSMITTER_TO_CHANNEL.keys()),
        help="Transmitter name to extract power for (ebc: <=June 27, ustar: >=June 28)"
    )
    parser.add_argument(
        "--num-locations",
        type=int,
        default=10,
        help="Number of monitoring locations to generate (default: 10)"
    )

    # Optional arguments
    parser.add_argument(
        "--gps-dir",
        type=Path,
        default=None,
        help="GPS data directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=None,
        help="Output YAML file path (default: config/monitoring_locations_<transmitter>.yaml)"
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=None,
        help="Output directory for processed data (default: data/processed/<transmitter>/)"
    )
    parser.add_argument(
        "--gps-pattern",
        type=str,
        default=None,
        help="GPS file pattern (default: auto-detect based on input-dir)"
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=20.0,
        help="GPS deduplication threshold in meters (default: 20.0)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum measurements per location (default: 10)"
    )
    parser.add_argument(
        "--samples-to-skip",
        type=int,
        default=2,
        help="Number of IQ sample files to skip at end of each directory (default: 2)"
    )
    parser.add_argument(
        "--time-offset",
        type=int,
        default=-6,
        help="GPS time offset in hours for timezone conversion (default: -6 for UTC-6)"
    )

    args = parser.parse_args()

    # Set default output paths if not specified
    if args.output_yaml is None:
        args.output_yaml = Path(f"config/monitoring_locations_{args.transmitter}.yaml")

    if args.output_data is None:
        args.output_data = Path(f"data/processed/{args.transmitter}/")

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1

    # Auto-detect GPS directory if not specified
    if args.gps_dir is None:
        raw_data_root = Path("C:/Users/serha/raw_data")
        args.gps_dir = auto_detect_gps_directory(args.input_dir, raw_data_root)

        if args.gps_dir is None:
            print("Error: Could not auto-detect GPS directory. Please specify with --gps-dir")
            return 1

        print(f"Auto-detected GPS directory: {args.gps_dir}")

    if not args.gps_dir.exists():
        print(f"Error: GPS directory does not exist: {args.gps_dir}")
        return 1

    # Auto-detect GPS pattern if not specified
    if args.gps_pattern is None:
        if 'stat_rot' in str(args.input_dir):
            if '/stat' in str(args.input_dir) or '\\stat' in str(args.input_dir):
                args.gps_pattern = "*Stat.txt"
                print(f"Auto-detected GPS pattern: {args.gps_pattern} (stationary data)")
            elif '/rot' in str(args.input_dir) or '\\rot' in str(args.input_dir):
                args.gps_pattern = "*Rot.txt"
                print(f"Auto-detected GPS pattern: {args.gps_pattern} (rotation data)")
            else:
                args.gps_pattern = "*.txt"
        else:
            args.gps_pattern = "*.txt"

    print("\n" + "="*80)
    print("RAW IQ DATA TO MONITORING LOCATIONS PROCESSOR")
    print("="*80)
    print(f"Input Directory: {args.input_dir}")
    print(f"GPS Directory: {args.gps_dir}")
    print(f"Transmitter: {args.transmitter}")
    print(f"Target Locations: {args.num_locations}")
    print("="*80 + "\n")

    # Step 1: Find IQ sample directories
    print("Step 1: Finding IQ sample directories...")
    sample_dirs = find_iq_sample_directories(args.input_dir)

    if not sample_dirs:
        print(f"Error: No IQ sample directories found in {args.input_dir}")
        return 1

    print(f"Found {len(sample_dirs)} sample directories")

    # Step 2: Load IQ samples
    print("\nStep 2: Loading IQ samples...")
    iq_data = load_iq_samples_from_directories(sample_dirs, args.samples_to_skip)

    if not iq_data:
        print("Error: No IQ samples loaded")
        return 1

    print(f"Loaded {len(iq_data)} IQ samples")

    # Step 3: Load GPS coordinates
    print("\nStep 3: Loading GPS coordinates...")
    gps_files = find_gps_files(args.gps_dir, args.gps_pattern)

    if not gps_files:
        print(f"Error: No GPS files found in {args.gps_dir}")
        return 1

    print(f"Found {len(gps_files)} GPS files")
    gps_coords = load_gps_from_csv(gps_files, args.time_offset)

    if not gps_coords:
        print("Error: No GPS coordinates loaded")
        return 1

    print(f"Loaded {len(gps_coords)} GPS coordinates")

    # Step 4: Process IQ samples and match with GPS
    print(f"\nStep 4: Processing IQ samples for transmitter '{args.transmitter}'...")
    measurements = match_power_with_gps(
        iq_data,
        gps_coords,
        args.transmitter
    )

    if not measurements:
        print("Error: No measurements matched with GPS coordinates")
        return 1

    # Step 5: Aggregate by location
    print(f"\nStep 5: Aggregating measurements by location...")
    aggregated = aggregate_measurements_by_location(
        measurements,
        args.dedup_threshold,
        args.min_samples
    )

    if not aggregated:
        print("Error: No locations met the minimum sample requirement")
        return 1

    # Select top N locations
    selected_locations = aggregated[:args.num_locations]

    # Step 6: Generate outputs
    print(f"\nStep 6: Generating outputs...")

    # Print summary
    print_summary(selected_locations, args.transmitter)

    # Generate YAML
    generate_monitoring_yaml(
        selected_locations,
        args.transmitter,
        args.output_yaml
    )

    # Save processed data
    save_processed_data(
        selected_locations,
        args.transmitter,
        args.output_data
    )

    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Review generated YAML: {args.output_yaml}")
    print(f"  2. Check processed data: {args.output_data}")
    print(f"  3. Use in notebook:")
    print(f"     locations = load_monitoring_locations('{args.output_yaml}', map_data)")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
