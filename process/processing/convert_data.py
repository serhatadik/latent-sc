"""
Unified Data Conversion Script.

This script consolidates the functionality of:
- convert_to_json.py (convert to JSON format)
- tabular_loc_power.py (convert to MATLAB .mat format)

Usage:
    python convert_data.py --format json --output data.json
    python convert_data.py --format matlab --output dd_meas_data.mat
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import argparse
from scipy.io import savemat
from geopy import distance
from config import settings
from core.distance_utils import load_transmitter_locations


def convert_to_json(input_dir: Path, output_file: Path, apply_outlier_filter: bool = True):
    """
    Convert numpy arrays to JSON format.

    Args:
        input_dir: Directory containing the .npy files
        output_file: Output JSON file path
        apply_outlier_filter: Apply outlier filtering for Ustar data
    """
    print("Loading data files...")

    # Load test data files
    tx1_ebc = np.load(input_dir / "TX1EBC_pow_test.npy")
    tx1_ustar = np.load(input_dir / "TX1Ustar_pow_test.npy")
    tx2 = np.load(input_dir / "TX2_pow_test.npy")
    tx3 = np.load(input_dir / "TX3_pow_test.npy")
    tx4 = np.load(input_dir / "TX4_pow_test.npy")
    tx5 = np.load(input_dir / "TX5_pow_test.npy")

    coord = np.load(input_dir / "coordinates_test.npy")
    coord_ebc = np.load(input_dir / "coordinates_ebc_test.npy")
    coord_ustar = np.load(input_dir / "coordinates_ustar_test.npy")

    # Load transmitter locations
    transmitter_config_path = Path(__file__).parent.parent / "config" / "transmitters.json"
    transmitters_latlon = load_transmitter_locations(str(transmitter_config_path))

    # Create transmitters dict with old naming convention for compatibility
    transmitters = {
        "ebcdd": transmitters_latlon["ebc"],
        "guesthousedd": transmitters_latlon["guesthouse"],
        "mariodd": transmitters_latlon["mario"],
        "morandd": transmitters_latlon["moran"],
        "wasatchdd": transmitters_latlon["wasatch"],
        "ustar": transmitters_latlon["ustar"]
    }

    # Calculate distances for outlier filtering
    if apply_outlier_filter:
        print("Calculating distances for outlier filtering...")
        dist_ustar = []
        for c in coord_ustar:
            dist_ustar.append(distance.distance(tuple(c[::-1]), transmitters["ustar"]).m)

        # Find outlier indices
        ind_rem = []
        for i in range(len(tx1_ustar)):
            if (dist_ustar[i] < settings.OUTLIER_DISTANCE_THRESHOLD_M and
                tx1_ustar[i] < settings.OUTLIER_POWER_THRESHOLD_DB):
                ind_rem.append(i)

        ind_rem = np.array(ind_rem)
        mask = ~np.isin(np.arange(tx1_ustar.size), ind_rem)

        # Apply mask to filter arrays
        tx1_ustar_filtered = np.array(tx1_ustar)[mask]
        print(f"Filtered {len(ind_rem)} outlier measurements from Ustar data")
    else:
        mask = np.ones(len(tx1_ustar), dtype=bool)
        tx1_ustar_filtered = tx1_ustar

    # Reconstruct times from IQ samples (need to reload to get timestamps)
    print("Loading IQ sample timestamps...")
    from core.data_loading import load_mobile_iq_samples
    data = load_mobile_iq_samples(
        settings.DATA_DIRS["walking"],
        settings.DATA_DIRS["driving"],
        samples_to_skip=settings.SAMPLES_TO_SKIP
    )
    times = sorted(list(data.keys()))

    # Load GPS data to get metadata (walking vs driving)
    # This requires checking which folder each sample came from
    import os
    walking_folders = [
        settings.DATA_DIRS["walking"] / name
        for name in os.listdir(settings.DATA_DIRS["walking"])
        if name.startswith('samples_20')
    ]
    driving_folders = [
        settings.DATA_DIRS["driving"] / name
        for name in os.listdir(settings.DATA_DIRS["driving"])
        if name.startswith('samples_20')
    ]

    # Create metadata lookup
    metadata_lookup = {}
    for folder in walking_folders:
        for file in os.listdir(folder):
            import pandas as pd
            time = pd.to_datetime(file.split('-IQ')[0].split('.')[0])
            time = np.datetime64(time).astype('datetime64[s]')
            metadata_lookup[time] = "walking"

    for folder in driving_folders:
        for file in os.listdir(folder):
            import pandas as pd
            time = pd.to_datetime(file.split('-IQ')[0].split('.')[0])
            time = np.datetime64(time).astype('datetime64[s]')
            metadata_lookup[time] = "driving"

    # Initialize JSON structure
    print("Building JSON structure...")
    structured_data = {}

    cnt = 0
    for i, time in enumerate(times):
        time_str = str(time)

        # Get metadata
        metadata = metadata_lookup.get(time, "unknown")

        structured_data[time_str] = {
            "pow_rx_tx": [],
            "metadata": [metadata]
        }

        # Add TX1 (EBC or Ustar based on day 27 split)
        if i < len(tx1_ebc) + len(tx1_ustar):
            if times[i].astype('datetime64[D]').item().day == settings.TX1_SPLIT_DAY and i < len(tx1_ebc):
                # EBC data
                structured_data[time_str]["pow_rx_tx"].append([
                    float(tx1_ebc[i]),
                    float(coord_ebc[i][1]),  # latitude
                    float(coord_ebc[i][0]),  # longitude
                    transmitters["ebcdd"][0],
                    transmitters["ebcdd"][1]
                ])
            elif times[i].astype('datetime64[D]').item().day != settings.TX1_SPLIT_DAY:
                # Ustar data
                if cnt == 0:
                    cnt += np.where(coord_ustar[:, 0] == coord[i][0])[0][0]
                if mask[cnt] == 1:
                    structured_data[time_str]["pow_rx_tx"].append([
                        float(tx1_ustar_filtered[cnt]),
                        float(coord_ustar[cnt][1]),
                        float(coord_ustar[cnt][0]),
                        transmitters["ustar"][0],
                        transmitters["ustar"][1]
                    ])
                cnt += 1

        # Add TX2-TX5 (Guesthouse, Mario, Moran, Wasatch)
        if i < len(tx2):
            structured_data[time_str]["pow_rx_tx"].append([
                float(tx2[i]), float(coord[i][1]), float(coord[i][0]),
                transmitters["guesthousedd"][0], transmitters["guesthousedd"][1]
            ])
        if i < len(tx3):
            structured_data[time_str]["pow_rx_tx"].append([
                float(tx3[i]), float(coord[i][1]), float(coord[i][0]),
                transmitters["mariodd"][0], transmitters["mariodd"][1]
            ])
        if i < len(tx4):
            structured_data[time_str]["pow_rx_tx"].append([
                float(tx4[i]), float(coord[i][1]), float(coord[i][0]),
                transmitters["morandd"][0], transmitters["morandd"][1]
            ])
        if i < len(tx5):
            structured_data[time_str]["pow_rx_tx"].append([
                float(tx5[i]), float(coord[i][1]), float(coord[i][0]),
                transmitters["wasatchdd"][0], transmitters["wasatchdd"][1]
            ])

    # Export to JSON
    print(f"Writing JSON to {output_file}...")
    with open(output_file, 'w') as json_file:
        json.dump(structured_data, json_file, indent=4)

    print("Done!")


def convert_to_matlab(input_dir: Path, output_file: Path):
    """
    Convert numpy arrays to MATLAB .mat format.

    Args:
        input_dir: Directory containing the .npy files
        output_file: Output .mat file path
    """
    print("Loading data files...")

    # Load test data files
    tx1_ebc = np.load(input_dir / "TX1EBC_pow_test.npy")
    tx2 = np.load(input_dir / "TX2_pow_test.npy")
    tx3 = np.load(input_dir / "TX3_pow_test.npy")
    tx4 = np.load(input_dir / "TX4_pow_test.npy")
    tx5 = np.load(input_dir / "TX5_pow_test.npy")

    coord = np.load(input_dir / "coordinates_test.npy")
    coord_ebc = np.load(input_dir / "coordinates_ebc_test.npy")

    # Reshape power arrays to column vectors if needed
    if tx2.ndim == 1:
        tx2 = tx2[:, np.newaxis]
    if tx3.ndim == 1:
        tx3 = tx3[:, np.newaxis]
    if tx4.ndim == 1:
        tx4 = tx4[:, np.newaxis]
    if tx5.ndim == 1:
        tx5 = tx5[:, np.newaxis]

    print(f"Coordinate array shape: {coord.shape}")
    print(f"EBC coordinate array shape: {coord_ebc.shape}")

    # Pad EBC data to match coordinate array length
    x = coord.shape[0]
    y = coord_ebc.shape[0]
    pad_length = x - y

    if tx1_ebc.ndim == 1:
        tx1_ebc_padded = np.pad(tx1_ebc, (0, pad_length), 'constant', constant_values=0)
    elif tx1_ebc.ndim == 2 and tx1_ebc.shape[1] == 1:
        tx1_ebc_padded = np.pad(tx1_ebc, ((0, pad_length), (0, 0)), 'constant', constant_values=0)

    tx1_ebc_padded = tx1_ebc_padded[:, np.newaxis]
    coord_ebc_padded = np.pad(coord_ebc, ((0, pad_length), (0, 0)), 'constant', constant_values=0)

    print(f"Padded EBC power shape: {tx1_ebc_padded.shape}")
    print(f"Padded EBC coordinates shape: {coord_ebc_padded.shape}")

    # Combine arrays: [tx_power, rx_lat, rx_lon] for each transmitter
    # Format: TX1EBC_pow, EBC_lat, EBC_lon, TX2_pow, lat, lon, TX3_pow, lat, lon, ...
    combined_array = np.hstack([
        tx1_ebc_padded, coord_ebc_padded[:, 1:2], coord_ebc_padded[:, 0:1],  # EBC
        tx2, coord[:, 1:2], coord[:, 0:1],  # Guesthouse
        tx3, coord[:, 1:2], coord[:, 0:1],  # Mario
        tx4, coord[:, 1:2], coord[:, 0:1],  # Moran
        tx5, coord[:, 1:2], coord[:, 0:1]   # Wasatch
    ])

    print(f"Combined array shape: {combined_array.shape}")

    # Create .mat file
    mat_data = {'measurements': combined_array}

    print(f"Saving to {output_file}...")
    savemat(str(output_file), mat_data)

    print("Done!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert processed measurement data to JSON or MATLAB format"
    )
    parser.add_argument(
        "--format",
        choices=["json", "matlab"],
        required=True,
        help="Output format"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory containing .npy files (default: files_generated_by_process_data_scripts/)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "--no-outlier-filter",
        action="store_true",
        help="Disable outlier filtering for JSON export (default: enabled)"
    )

    args = parser.parse_args()

    # Default input directory
    if args.input_dir is None:
        args.input_dir = settings.LEGACY_OUTPUT_DIR

    if args.format == "json":
        convert_to_json(
            args.input_dir,
            args.output,
            apply_outlier_filter=not args.no_outlier_filter
        )
    elif args.format == "matlab":
        convert_to_matlab(args.input_dir, args.output)


if __name__ == "__main__":
    main()
