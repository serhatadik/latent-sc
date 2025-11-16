#!/usr/bin/env python
"""
Pipeline Script 02: Estimate Signal Strength

This script performs the core localization algorithm to estimate signal strength
across the geographic region using the likelihood-based method with covariance modeling.

Implements:
- Transmit power estimation (Equation 3)
- Covariance matrix construction (Equation 5)
- Probability mass function computation (Equation 4)
- Signal strength prediction (Equation 6)

Usage:
    python 02_estimate_signals.py --config config/parameters.yaml
    python 02_estimate_signals.py --config config/parameters.yaml --band "3610-3650"
"""

import argparse
import os
import sys
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.localization import (
    estimate_transmit_power_map,
    build_covariance_matrix,
    compute_transmitter_pmf,
    estimate_received_power_map
)
from src.utils import load_slc_map


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_monitoring_locations(locations_path):
    """Load monitoring station locations from YAML file."""
    with open(locations_path, 'r') as f:
        locations = yaml.safe_load(f)
    return locations['data_points']


def load_occupancy_metrics(metrics_path):
    """Load occupancy metrics from previous pipeline step."""
    data = np.load(metrics_path)
    return data


def estimate_signals_for_band(band_name, band_config, monitoring_locations,
                              occupancy_metrics, map_data, localization_config):
    """
    Estimate signal strength for a single frequency band.

    Parameters
    ----------
    band_name : str
        Name of frequency band
    band_config : dict
        Band configuration
    monitoring_locations : list
        List of monitoring station configurations
    occupancy_metrics : dict
        Loaded occupancy metrics from step 01
    map_data : dict
        Loaded SLC map data
    localization_config : dict
        Localization parameters from config

    Returns
    -------
    results : dict
        Dictionary containing transmit_power_map, pmf, signal_estimates
    """
    print(f"\n{'='*60}")
    print(f"Estimating signals for {band_config['name']}")
    print(f"{'='*60}\n")

    # Extract monitoring station coordinates and observed powers
    data_points = []
    observed_powers = []

    for location in monitoring_locations:
        monitor_name = location['name']
        coords = location['coordinates']
        data_points.append(coords)

        # Get observed power for this monitor/band
        key = f"{band_name}_{monitor_name}_avg_power"
        if key in occupancy_metrics:
            power = occupancy_metrics[key]
            observed_powers.append(power)
        else:
            print(f"WARNING: No data for {monitor_name} in {band_name}")

    data_points = np.array(data_points)
    observed_powers = np.array(observed_powers)

    print(f"Monitoring stations: {len(data_points)}")
    print(f"Observed powers: {observed_powers}")

    # Get parameters
    scale = localization_config['proxel_size']
    np_exponent = localization_config['path_loss_exponent']
    sigma = localization_config['std_deviation']
    delta_c = localization_config['correlation_coeff']

    # Step 1: Estimate transmit power map (Equation 3)
    print("\nStep 1: Estimating transmit power at all locations...")
    transmit_power_map = estimate_transmit_power_map(
        map_shape=map_data['shape'],
        sensor_locations=data_points,
        observed_powers=observed_powers,
        scale=scale,
        np_exponent=np_exponent,
        n_jobs=-1  # Use all cores
    )
    print(f"✓ Transmit power map computed: shape {transmit_power_map.shape}")

    # Step 2: Build covariance matrix (Equation 5)
    print("\nStep 2: Building covariance matrix...")
    cov_matrix = build_covariance_matrix(
        sensor_locations=data_points,
        sigma=sigma,
        delta_c=delta_c,
        scale=scale
    )
    print(f"✓ Covariance matrix: shape {cov_matrix.shape}")

    # Step 3: Compute transmitter probability mass function (Equation 4)
    print("\nStep 3: Computing transmitter PMF...")
    pmf = compute_transmitter_pmf(
        transmit_power_map=transmit_power_map,
        sensor_locations=data_points,
        observed_powers=observed_powers,
        cov_matrix=cov_matrix,
        scale=scale,
        np_exponent=np_exponent
    )
    print(f"✓ PMF computed: shape {pmf.shape}")
    print(f"  PMF sum: {np.sum(pmf):.6f} (should be ≈ 1.0)")

    # Find most likely transmitter location
    most_likely_idx = np.unravel_index(pmf.argmax(), pmf.shape)
    print(f"  Most likely transmitter location: {most_likely_idx}")

    # Step 4: Estimate received power map (Equation 6)
    print("\nStep 4: Estimating received power at all locations...")

    # Create target grid (all pixels)
    rows, cols = map_data['shape']
    target_grid = []
    for i in range(rows):
        for j in range(cols):
            target_grid.append([i, j])
    target_grid = np.array(target_grid)

    signal_estimates = estimate_received_power_map(
        transmit_power_map=transmit_power_map,
        pmf=pmf,
        sensor_locations=data_points,
        target_grid=target_grid,
        scale=scale,
        np_exponent=np_exponent,
        probability_threshold=1e-6
    )

    # Reshape to 2D map
    signal_estimates_2d = signal_estimates.reshape(map_data['shape'])
    print(f"✓ Signal estimates computed: shape {signal_estimates_2d.shape}")

    # Calculate evaluation metrics (cross-validation style)
    print("\nStep 5: Calculating evaluation metrics...")
    mse_scores = []
    for i in range(len(data_points)):
        actual = observed_powers[i]
        predicted = signal_estimates_2d[data_points[i, 1], data_points[i, 0]]

        # Convert from linear to dB if needed
        from src.utils.conversions import lin_to_dB
        predicted_dB = lin_to_dB(predicted)

        error = (actual - predicted_dB) ** 2
        mse_scores.append(error)
        print(f"  {monitoring_locations[i]['name']}: Actual={actual:.2f}, Predicted={predicted_dB:.2f}, MSE={error:.2f}")

    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    print(f"\n  Mean MSE: {mean_mse:.2f} dB²")
    print(f"  Std MSE: {std_mse:.2f} dB²")

    # Calculate variance baseline (variance of observed powers)
    variance_baseline = np.var(observed_powers)
    print(f"  Baseline variance: {variance_baseline:.2f} dB²")
    print(f"  Variance reduction: {(1 - mean_mse/variance_baseline)*100:.1f}%")

    return {
        'transmit_power_map': transmit_power_map,
        'pmf': pmf,
        'signal_estimates': signal_estimates_2d,
        'mse_scores': np.array(mse_scores),
        'mean_mse': mean_mse,
        'std_mse': std_mse,
        'variance_baseline': variance_baseline
    }


def main(args):
    """Main processing function."""
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    locations_config = load_monitoring_locations(args.locations)

    # Load SLC map
    print("Loading SLC map...")
    map_data = load_slc_map(
        map_folder_dir=args.map_dir,
        downsample_factor=config['spatial']['downsample_factor']
    )
    print(f"✓ Map loaded: shape {map_data['shape']}")

    # Load occupancy metrics from previous step
    metrics_file = Path(args.metrics_path)
    if not metrics_file.exists():
        print(f"ERROR: Occupancy metrics file not found: {metrics_file}")
        print("Please run 01_process_occupancy.py first")
        sys.exit(1)

    print(f"Loading occupancy metrics from {metrics_file}...")
    occupancy_metrics = load_occupancy_metrics(metrics_file)
    print(f"✓ Loaded {len(occupancy_metrics.files)} metric values")

    # Get list of bands to process
    if args.band:
        bands_to_process = {args.band: config['frequency_bands'][args.band]}
    else:
        bands_to_process = config['frequency_bands']

    # Results storage
    all_results = {}

    # Process each band
    for band_name, band_config in bands_to_process.items():
        results = estimate_signals_for_band(
            band_name, band_config, locations_config,
            occupancy_metrics, map_data, config['localization']
        )
        all_results[band_name] = results

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for band_name, results in all_results.items():
        output_file = output_dir / f'signal_estimates_{band_name}.npz'
        print(f"\nSaving results for {band_name} to {output_file}")

        np.savez(
            output_file,
            transmit_power_map=results['transmit_power_map'],
            pmf=results['pmf'],
            signal_estimates=results['signal_estimates'],
            mse_scores=results['mse_scores'],
            mean_mse=results['mean_mse'],
            std_mse=results['std_mse'],
            variance_baseline=results['variance_baseline']
        )

    print(f"\n✓ All results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate signal strength using likelihood-based localization')
    parser.add_argument('--config', type=str, default='config/parameters.yaml',
                       help='Path to configuration file')
    parser.add_argument('--locations', type=str, default='config/monitoring_locations.yaml',
                       help='Path to monitoring locations file')
    parser.add_argument('--map-dir', type=str, default='./',
                       help='Directory containing SLC map file')
    parser.add_argument('--metrics-path', type=str, default='./data/processed/occupancy_metrics.npz',
                       help='Path to occupancy metrics from step 01')
    parser.add_argument('--output-dir', type=str, default='./data/processed/',
                       help='Output directory for signal estimates')
    parser.add_argument('--band', type=str, default=None,
                       help='Process only specific band (e.g., "3610-3650")')

    args = parser.parse_args()
    main(args)
