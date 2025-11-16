#!/usr/bin/env python
"""
Pipeline Script 01: Process Occupancy Metrics

This script extracts occupancy metrics (duty cycle, average power, signal variation)
from raw spectrum monitoring data for all configured frequency bands and monitors.

Usage:
    python 01_process_occupancy.py --config config/parameters.yaml
    python 01_process_occupancy.py --config config/parameters.yaml --band "3610-3650"
    python 01_process_occupancy.py --config config/parameters.yaml --monitor Bookstore
"""

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import load_monitoring_data, compute_occupancy_metrics
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


def process_single_monitor_band(monitor_name, band_config, data_path='./data/raw/rfbaseline/',
                                cutoff_date='2023-01-01'):
    """
    Process occupancy metrics for a single monitor and frequency band.

    Parameters
    ----------
    monitor_name : str
        Name of monitoring station
    band_config : dict
        Band configuration with 'start', 'end', 'threshold_start', 'threshold_end'
    data_path : str
        Path to raw data directory
    cutoff_date : str
        Cutoff date for data filtering

    Returns
    -------
    metrics : dict
        Computed occupancy metrics
    """
    band_start = band_config['start']
    band_end = band_config['end']
    threshold_start = band_config['threshold_start']
    threshold_end = band_config['threshold_end']

    print(f"  Loading data for {monitor_name} @ {band_start}-{band_end} MHz...")

    try:
        # Load data
        df = load_monitoring_data(monitor_name, band_start, band_end,
                                 base_path=data_path, cutoff_date=cutoff_date)

        if df is None or len(df) == 0:
            print(f"  WARNING: No data found for {monitor_name}")
            return None

        # Compute metrics
        metrics = compute_occupancy_metrics(df, band_start, band_end,
                                           threshold_start, threshold_end)

        print(f"  ✓ Duty Cycle: {metrics['duty_cycle']:.2f}%")
        print(f"  ✓ Avg Power (occupied): {metrics['avg_power_occupied']:.2f} dB")
        print(f"  ✓ Signal Variation: {metrics['signal_variation']:.2f} dB²")

        return metrics

    except Exception as e:
        print(f"  ERROR processing {monitor_name}: {e}")
        return None


def main(args):
    """Main processing function."""
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    locations_config = load_monitoring_locations(args.locations)

    # Get list of monitors to process
    if args.monitor:
        monitors_to_process = [args.monitor]
    else:
        monitors_to_process = [loc['name'] for loc in locations_config]

    # Get list of bands to process
    if args.band:
        bands_to_process = {args.band: config['frequency_bands'][args.band]}
    else:
        bands_to_process = config['frequency_bands']

    print(f"\nProcessing {len(monitors_to_process)} monitors × {len(bands_to_process)} bands\n")

    # Results storage
    all_results = {}

    # Process each band
    for band_name, band_config in bands_to_process.items():
        print(f"\n{'='*60}")
        print(f"BAND: {band_config['name']}")
        print(f"{'='*60}")

        band_results = {}

        # Process each monitor
        for monitor_name in monitors_to_process:
            print(f"\nProcessing {monitor_name}...")

            metrics = process_single_monitor_band(
                monitor_name, band_config,
                data_path=args.data_path,
                cutoff_date=config['data']['cutoff_date']
            )

            if metrics:
                band_results[monitor_name] = metrics

        all_results[band_name] = band_results

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'occupancy_metrics.npz'
    print(f"\n{'='*60}")
    print(f"Saving results to {output_file}")
    print(f"{'='*60}\n")

    # Convert to numpy arrays for saving
    save_dict = {}
    for band_name, band_results in all_results.items():
        for monitor_name, metrics in band_results.items():
            key_prefix = f"{band_name}_{monitor_name}"
            save_dict[f"{key_prefix}_duty_cycle"] = metrics['duty_cycle']
            save_dict[f"{key_prefix}_avg_power"] = metrics['avg_power_occupied']
            save_dict[f"{key_prefix}_variation"] = metrics['signal_variation']

    np.savez(output_file, **save_dict)

    # Also save as human-readable CSV
    csv_rows = []
    for band_name, band_results in all_results.items():
        for monitor_name, metrics in band_results.items():
            csv_rows.append({
                'band': band_name,
                'monitor': monitor_name,
                'duty_cycle_%': metrics['duty_cycle'],
                'avg_power_dB': metrics['avg_power_occupied'],
                'signal_variation_dB2': metrics['signal_variation']
            })

    df_results = pd.DataFrame(csv_rows)
    csv_file = output_dir / 'occupancy_metrics.csv'
    df_results.to_csv(csv_file, index=False)
    print(f"Also saved as CSV: {csv_file}\n")

    # Print summary
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(df_results.to_string(index=False))
    print(f"\n✓ Processing complete! Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract occupancy metrics from spectrum monitoring data')
    parser.add_argument('--config', type=str, default='config/parameters.yaml',
                       help='Path to configuration file')
    parser.add_argument('--locations', type=str, default='config/monitoring_locations.yaml',
                       help='Path to monitoring locations file')
    parser.add_argument('--data-path', type=str, default='./data/raw/rfbaseline/',
                       help='Path to raw data directory')
    parser.add_argument('--output-dir', type=str, default='./data/processed/',
                       help='Output directory for processed metrics')
    parser.add_argument('--band', type=str, default=None,
                       help='Process only specific band (e.g., "3610-3650")')
    parser.add_argument('--monitor', type=str, default=None,
                       help='Process only specific monitor (e.g., Bookstore)')

    args = parser.parse_args()
    main(args)
