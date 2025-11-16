"""
Unified RF Measurement Processing Script.

This script consolidates the functionality of:
- read_samples.py (mobile measurements)
- Fading_Analysis.py (stationary measurements)
- fading_analysis_rot.py (rotation measurements)

Usage:
    python process_measurements.py --dataset-type mobile
    python process_measurements.py --dataset-type stationary
    python process_measurements.py --dataset-type rotation
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import argparse
from config import settings
from core.signal_processing import process_sample_to_powers
from core.gps_utils import load_mobile_gps_data, load_stationary_gps_data
from core.data_loading import load_mobile_iq_samples, load_stationary_iq_samples


def process_measurements(dataset_type: str, output_dir: Path = None):
    """
    Process RF measurements for a given dataset type.

    Args:
        dataset_type: One of 'mobile', 'stationary', 'rotation'
        output_dir: Directory to save output files (default: LEGACY_OUTPUT_DIR)
    """
    if output_dir is None:
        output_dir = settings.LEGACY_OUTPUT_DIR

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data based on dataset type
    print(f"Processing {dataset_type} measurements...")

    if dataset_type == "mobile":
        # Load mobile (walking + driving) data
        print("Loading mobile IQ samples...")
        data = load_mobile_iq_samples(
            settings.DATA_DIRS["walking"],
            settings.DATA_DIRS["driving"],
            samples_to_skip=settings.SAMPLES_TO_SKIP
        )

        print("Loading mobile GPS data...")
        coords = load_mobile_gps_data(
            settings.GPS_DATA_DIRS["mobile"],
            settings.GPS_DATA_DIRS["serhat"],
            time_offset_hours=settings.TIME_OFFSET_HOURS,
            min_latitude=settings.MIN_LATITUDE
        )

        output_suffix = "test"

    elif dataset_type == "stationary":
        # Load stationary data
        print("Loading stationary IQ samples...")
        data = load_stationary_iq_samples(
            settings.DATA_DIRS["stat_rot"],
            folder_pattern="stat",
            samples_to_skip=settings.SAMPLES_TO_SKIP
        )

        print("Loading stationary GPS data...")
        coords = load_stationary_gps_data(
            settings.GPS_DATA_DIRS["stat_rot"],
            file_pattern="*Stat.txt",
            time_offset_hours=settings.TIME_OFFSET_HOURS
        )

        output_suffix = "stat"

    elif dataset_type == "rotation":
        # Load rotation data
        print("Loading rotation IQ samples...")
        data = load_stationary_iq_samples(
            settings.DATA_DIRS["stat_rot"],
            folder_pattern="rot",
            samples_to_skip=settings.SAMPLES_TO_SKIP
        )

        print("Loading rotation GPS data...")
        # Note: rotation data uses files ending with 'Rot.txt' (capital R)
        import os
        gps_files = [
            str(settings.GPS_DATA_DIRS["stat_rot"] / name)
            for name in os.listdir(settings.GPS_DATA_DIRS["stat_rot"])
            if name.endswith('Rot.txt')
        ]
        import pandas as pd
        from core.gps_utils import load_gps_from_csv
        coords = load_gps_from_csv(
            gps_files,
            time_offset_hours=settings.TIME_OFFSET_HOURS
        )

        output_suffix = "rot"

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Sort times
    times = sorted(list(data.keys()))
    print(f"Processing {len(times)} samples...")

    # Initialize power and coordinate lists
    TX1EBC_Pow = []
    TX1Ustar_Pow = []
    TX2_Pow = []
    TX3_Pow = []
    TX4_Pow = []
    TX5_Pow = []
    coordinates = []
    coordinates_ebc = []
    coordinates_ustar = []

    # Process each sample
    for i, time in enumerate(times):
        if i % settings.PROGRESS_PRINT_INTERVAL == 0:
            print(f"Progress: {i/len(times)*100:.1f}%")

        # Process IQ samples to extract power for each channel
        powers = process_sample_to_powers(
            data[time],
            settings.SAMPLE_RATE,
            settings.CENTER_FREQ,
            settings.RF_CHANNELS
        )

        # Only save if we have GPS coordinates for this timestamp
        if time in coords:
            # For rotation data, TX1 is always Ustar (no day 27 split, no EBC)
            if dataset_type == "rotation":
                TX1Ustar_Pow.append(powers["TX1"])
                coordinates_ustar.append(coords[time])
            else:
                # For mobile and stationary, split TX1 based on day
                if time.astype('datetime64[D]').item().day == settings.TX1_SPLIT_DAY:
                    TX1EBC_Pow.append(powers["TX1"])
                    coordinates_ebc.append(coords[time])
                else:
                    TX1Ustar_Pow.append(powers["TX1"])
                    coordinates_ustar.append(coords[time])

            # TX2-TX5 are saved for all dataset types
            TX2_Pow.append(powers["TX2"])
            TX3_Pow.append(powers["TX3"])
            TX4_Pow.append(powers["TX4"])
            TX5_Pow.append(powers["TX5"])
            coordinates.append(coords[time])

    print("Saving results...")

    # Save results
    # For rotation, we don't save EBC data
    if dataset_type != "rotation":
        np.save(output_dir / f"TX1EBC_pow_{output_suffix}.npy", TX1EBC_Pow)
        np.save(output_dir / f"coordinates_ebc_{output_suffix}.npy", coordinates_ebc)

    np.save(output_dir / f"TX1Ustar_pow_{output_suffix}.npy", TX1Ustar_Pow)
    np.save(output_dir / f"TX2_pow_{output_suffix}.npy", TX2_Pow)
    np.save(output_dir / f"TX3_pow_{output_suffix}.npy", TX3_Pow)
    np.save(output_dir / f"TX4_pow_{output_suffix}.npy", TX4_Pow)
    np.save(output_dir / f"TX5_pow_{output_suffix}.npy", TX5_Pow)
    np.save(output_dir / f"coordinates_{output_suffix}.npy", coordinates)
    np.save(output_dir / f"coordinates_ustar_{output_suffix}.npy", coordinates_ustar)

    print(f"Done! Processed {len(coordinates)} samples with GPS coordinates.")
    print(f"Output saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process RF measurements and extract power per channel"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["mobile", "stationary", "rotation"],
        required=True,
        help="Type of dataset to process"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: files_generated_by_process_data_scripts/)"
    )

    args = parser.parse_args()

    process_measurements(args.dataset_type, args.output_dir)


if __name__ == "__main__":
    main()
