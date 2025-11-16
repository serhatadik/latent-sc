"""
Unified Measurement Visualization Script.

This script consolidates the functionality of:
- visualize_RSSI_dist.py (mobile measurements)
- visualize_fading.py (stationary measurements)
- visualize_fading_rot.py (rotation measurements)
- fading_statistics.py (stationary statistics - duplicate of visualize_fading.py)

Usage:
    python visualize_measurements.py --dataset-type mobile
    python visualize_measurements.py --dataset-type stationary
    python visualize_measurements.py --dataset-type rotation
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import argparse
from geopy import distance
from config import settings
from core.distance_utils import load_transmitter_locations
from core.gps_utils import deduplicate_coordinates, find_unique_coordinates


def visualize_mobile(input_dir: Path):
    """Visualize mobile (walking/driving) measurements."""
    print("Loading mobile measurement data...")

    # Load data
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
    transmitters = load_transmitter_locations(str(transmitter_config_path))

    # Calculate distances
    print("Calculating distances...")
    dist1 = [distance.distance(tuple(c[::-1]), transmitters["ebc"]).m for c in coord_ebc]
    dist2 = [distance.distance(tuple(c[::-1]), transmitters["ustar"]).m for c in coord_ustar]
    dist3 = [distance.distance(tuple(c[::-1]), transmitters["guesthouse"]).m for c in coord]
    dist4 = [distance.distance(tuple(c[::-1]), transmitters["mario"]).m for c in coord]
    dist5 = [distance.distance(tuple(c[::-1]), transmitters["moran"]).m for c in coord]
    dist6 = [distance.distance(tuple(c[::-1]), transmitters["wasatch"]).m for c in coord]

    # Apply outlier filter for Ustar
    print("Applying outlier filter for Ustar...")
    ind_rem = []
    for i in range(len(tx1_ustar)):
        if (dist2[i] < settings.OUTLIER_DISTANCE_THRESHOLD_M and
            tx1_ustar[i] < settings.OUTLIER_POWER_THRESHOLD_DB):
            ind_rem.append(i)

    mask = ~np.isin(np.arange(tx1_ustar.size), np.array(ind_rem))
    tx1_ustar = np.array(tx1_ustar)[mask]
    dist2 = np.array(dist2)[mask]

    print(f"Filtered {len(ind_rem)} outliers from Ustar data")

    # Create visualization
    print("Creating plots...")
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(20, 4))

    # EBC
    ax1.scatter(np.log10(dist1), tx1_ebc, s=20, c='blue', alpha=0.7, edgecolors="k")
    ax1.grid(visible=True, color='green', linestyle='--', which='major')
    ax1.set_xlabel("Log. Distance (log10(d[m]))")
    ax1.set_ylabel("10log10(W/Hz)")
    m, b = np.polyfit(np.log10(dist1), tx1_ebc, 1)
    ax1.plot(np.log10(dist1), m*np.log10(dist1)+b, linewidth=4, color='red', linestyle='-.')
    ax1.set_title(f"EBC \n max. d = {max(dist1):.0f} m \n n = {abs(m/10):.2f}")

    # Ustar
    ax2.scatter(np.log10(dist2), tx1_ustar, s=20, c='blue', alpha=0.7, edgecolors="k")
    ax2.grid(visible=True, color='green', linestyle='--', which='major')
    ax2.set_xlabel("Log. Distance (log10(d[m]))")
    m, b = np.polyfit(np.log10(dist2), tx1_ustar, 1)
    ax2.plot(np.log10(dist2), m*np.log10(dist2)+b, linewidth=4, color='red', linestyle='-.')
    ax2.set_title(f"Ustar \n max. d = {max(dist2):.0f} m \n n = {abs(m/10):.2f}")

    # Guesthouse
    ax3.scatter(np.log10(dist3), tx2, s=20, c='blue', alpha=0.7, edgecolors="k")
    ax3.grid(visible=True, color='green', linestyle='--', which='major')
    ax3.set_xlabel("Log. Distance (log10(d[m]))")
    m, b = np.polyfit(np.log10(dist3), tx2, 1)
    ax3.plot(np.log10(dist3), m*np.log10(dist3)+b, linewidth=4, color='red', linestyle='-.')
    ax3.set_title(f"Guesthouse \n max. d = {max(dist3):.0f} m \n n = {abs(m/10):.2f}")

    # Mario
    ax4.scatter(np.log10(dist4), tx3, s=20, c='blue', alpha=0.7, edgecolors="k")
    ax4.grid(visible=True, color='green', linestyle='--', which='major')
    ax4.set_xlabel("Log. Distance (log10(d[m]))")
    m, b = np.polyfit(np.log10(dist4), tx3, 1)
    ax4.plot(np.log10(dist4), m*np.log10(dist4)+b, linewidth=4, color='red', linestyle='-.')
    ax4.set_title(f"Mario \n max. d = {max(dist4):.0f} m \n n = {abs(m/10):.2f}")

    # Moran
    ax5.scatter(np.log10(dist5), tx4, s=20, c='blue', alpha=0.7, edgecolors="k")
    ax5.grid(visible=True, color='green', linestyle='--', which='major')
    ax5.set_xlabel("Log. Distance (log10(d[m]))")
    m, b = np.polyfit(np.log10(dist5), tx4, 1)
    ax5.plot(np.log10(dist5), m*np.log10(dist5)+b, linewidth=4, color='red', linestyle='-.')
    ax5.set_title(f"Moran \n max. d = {max(dist5):.0f} m \n n = {abs(m/10):.2f}")

    # Wasatch
    ax6.scatter(np.log10(dist6), tx5, s=20, c='blue', alpha=0.7, edgecolors="k")
    ax6.grid(visible=True, color='green', linestyle='--', which='major')
    ax6.set_xlabel("Log. Distance (log10(d[m]))")
    m, b = np.polyfit(np.log10(dist6), tx5, 1)
    ax6.plot(np.log10(dist6), m*np.log10(dist6)+b, linewidth=4, color='red', linestyle='-.')
    ax6.set_title(f"Wasatch \n max. d = {max(dist6):.0f} m \n n = {abs(m/10):.2f}")

    plt.suptitle("Power Distribution of Mobile Measurements")
    plt.tight_layout()
    plt.show()


def visualize_stationary(input_dir: Path, dataset_suffix: str, dataset_name: str):
    """
    Visualize stationary or rotation measurements.

    Args:
        input_dir: Directory containing .npy files
        dataset_suffix: 'stat' or 'rot'
        dataset_name: 'Stationary' or 'Rotationary' for plot titles
    """
    print(f"Loading {dataset_name.lower()} measurement data...")

    # Load data
    tx2_ustar = np.load(input_dir / f"TX1Ustar_pow_{dataset_suffix}.npy")
    tx3 = np.load(input_dir / f"TX2_pow_{dataset_suffix}.npy")
    tx4 = np.load(input_dir / f"TX3_pow_{dataset_suffix}.npy")
    tx5 = np.load(input_dir / f"TX4_pow_{dataset_suffix}.npy")
    tx6 = np.load(input_dir / f"TX5_pow_{dataset_suffix}.npy")

    coord = np.load(input_dir / f"coordinates_{dataset_suffix}.npy")
    coord_ustar = np.load(input_dir / f"coordinates_ustar_{dataset_suffix}.npy")

    # Load transmitter locations
    transmitter_config_path = Path(__file__).parent.parent / "config" / "transmitters.json"
    transmitters = load_transmitter_locations(str(transmitter_config_path))

    # Calculate distances (before deduplication)
    print("Calculating distances...")
    dist2 = [distance.distance(tuple(c[::-1]), transmitters["ustar"]).m for c in coord_ustar]
    dist3 = [distance.distance(tuple(c[::-1]), transmitters["guesthouse"]).m for c in coord]
    dist4 = [distance.distance(tuple(c[::-1]), transmitters["mario"]).m for c in coord]
    dist5 = [distance.distance(tuple(c[::-1]), transmitters["moran"]).m for c in coord]
    dist6 = [distance.distance(tuple(c[::-1]), transmitters["wasatch"]).m for c in coord]

    # Deduplicate coordinates
    print("Deduplicating coordinates...")
    coord_unified = deduplicate_coordinates(coord, settings.COORD_DEDUP_THRESHOLD_M)
    coord_ustar_unified = deduplicate_coordinates(coord, settings.COORD_DEDUP_THRESHOLD_M)

    # Calculate distances after deduplication
    dist2_ded = [distance.distance(tuple(c[::-1]), transmitters["ustar"]).m for c in coord_ustar_unified]
    dist3_ded = [distance.distance(tuple(c[::-1]), transmitters["guesthouse"]).m for c in coord_unified]
    dist4_ded = [distance.distance(tuple(c[::-1]), transmitters["mario"]).m for c in coord_unified]
    dist5_ded = [distance.distance(tuple(c[::-1]), transmitters["moran"]).m for c in coord_unified]
    dist6_ded = [distance.distance(tuple(c[::-1]), transmitters["wasatch"]).m for c in coord_unified]

    # Figure 1: Power distribution
    print("Creating power distribution plots...")
    fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(18, 4))

    ax1.scatter(np.log10(dist2_ded), tx2_ustar, s=20, c='blue', alpha=0.7, edgecolors="k")
    ax1.grid(visible=True)
    ax1.set_xlabel("Log. Distance (log10(d[m]))")
    ax1.set_ylabel("10log10(W/Hz)")
    ax1.set_title(f"Ustar \n max. d = {max(dist2_ded):.0f} m")

    ax2.scatter(np.log10(dist3_ded), tx3, s=20, c='blue', alpha=0.7, edgecolors="k")
    ax2.grid(visible=True)
    ax2.set_xlabel("Log. Distance (log10(d[m]))")
    ax2.set_title(f"Guesthouse \n max. d = {max(dist3_ded):.0f} m")

    ax3.scatter(np.log10(dist4_ded), tx4, s=20, c='blue', alpha=0.7, edgecolors="k")
    ax3.grid(visible=True)
    ax3.set_xlabel("Log. Distance (log10(d[m]))")
    ax3.set_title(f"Mario \n max. d = {max(dist4_ded):.0f} m")

    ax4.scatter(np.log10(dist5_ded), tx5, s=20, c='blue', alpha=0.7, edgecolors="k")
    ax4.grid(visible=True)
    ax4.set_xlabel("Log. Distance (log10(d[m]))")
    ax4.set_title(f"Moran \n max. d = {max(dist5_ded):.0f} m")

    ax5.scatter(np.log10(dist6_ded), tx6, s=20, c='blue', alpha=0.7, edgecolors="k")
    ax5.grid(visible=True)
    ax5.set_xlabel("Log. Distance (log10(d[m]))")
    ax5.set_title(f"Wasatch \n max. d = {max(dist6_ded):.0f} m")

    plt.suptitle(f"Power Distribution of {dataset_name} Measurements")
    plt.tight_layout()

    # Figure 2: Standard deviation
    print("Calculating variance statistics...")
    locs = find_unique_coordinates(coord_unified)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []

    for uniq_loc in locs:
        idxs = np.where(coord_unified == uniq_loc)
        var_3.append(np.var(tx3[np.unique(idxs[0])]))
        var_4.append(np.var(tx4[np.unique(idxs[0])]))
        var_5.append(np.var(tx5[np.unique(idxs[0])]))
        var_6.append(np.var(tx6[np.unique(idxs[0])]))

    var_2 = []
    for uniq_loc in find_unique_coordinates(coord_ustar_unified):
        idxs = np.where(coord_ustar_unified == uniq_loc)
        var_2.append(np.var(tx2_ustar[np.unique(idxs[0])]))

    # Calculate distances for variance plots
    dist2_var = [distance.distance(tuple(c[::-1]), transmitters["ustar"]).m for c in locs]
    dist3_var = [distance.distance(tuple(c[::-1]), transmitters["guesthouse"]).m for c in locs]
    dist4_var = [distance.distance(tuple(c[::-1]), transmitters["mario"]).m for c in locs]
    dist5_var = [distance.distance(tuple(c[::-1]), transmitters["moran"]).m for c in locs]
    dist6_var = [distance.distance(tuple(c[::-1]), transmitters["wasatch"]).m for c in locs]

    print("Creating standard deviation plots...")
    fig2, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(18, 4))

    ax1.scatter(np.log10(dist2_var), np.sqrt(var_2), s=20, c='blue', alpha=0.7, edgecolors="k")
    ax1.grid(visible=True)
    ax1.set_xlabel("Log. Distance (log10(d[m]))")
    ax1.set_ylabel("Std. [dB]")
    ax1.set_title("Ustar")

    ax2.scatter(np.log10(dist3_var), np.sqrt(var_3), s=20, c='blue', alpha=0.7, edgecolors="k")
    ax2.grid(visible=True)
    ax2.set_xlabel("Log. Distance (log10(d[m]))")
    ax2.set_title("Guesthouse")

    ax3.scatter(np.log10(dist4_var), np.sqrt(var_4), s=20, c='blue', alpha=0.7, edgecolors="k")
    ax3.grid(visible=True)
    ax3.set_xlabel("Log. Distance (log10(d[m]))")
    ax3.set_title("Mario")

    ax4.scatter(np.log10(dist5_var), np.sqrt(var_5), s=20, c='blue', alpha=0.7, edgecolors="k")
    ax4.grid(visible=True)
    ax4.set_xlabel("Log. Distance (log10(d[m]))")
    ax4.set_title("Moran")

    ax5.scatter(np.log10(dist6_var), np.sqrt(var_6), s=20, c='blue', alpha=0.7, edgecolors="k")
    ax5.grid(visible=True)
    ax5.set_xlabel("Log. Distance (log10(d[m]))")
    ax5.set_title("Wasatch")

    plt.suptitle(f"Standard Deviation of {dataset_name} Measurements")
    plt.tight_layout()

    plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize RF measurement data"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["mobile", "stationary", "rotation"],
        required=True,
        help="Type of dataset to visualize"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory containing .npy files (default: files_generated_by_process_data_scripts/)"
    )

    args = parser.parse_args()

    # Default input directory
    if args.input_dir is None:
        args.input_dir = settings.LEGACY_OUTPUT_DIR

    if args.dataset_type == "mobile":
        visualize_mobile(args.input_dir)
    elif args.dataset_type == "stationary":
        visualize_stationary(args.input_dir, "stat", "Stationary")
    elif args.dataset_type == "rotation":
        visualize_stationary(args.input_dir, "rot", "Rotationary")


if __name__ == "__main__":
    main()
