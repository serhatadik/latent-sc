#!/usr/bin/env python
"""
Pipeline Script 03: Temporal Analysis

This script performs time-of-day and seasonal analysis of spectrum occupancy metrics.
Creates Figure 5 in the paper.

Usage:
    python 03_temporal_analysis.py --config config/parameters.yaml
    python 03_temporal_analysis.py --config config/parameters.yaml --monitor Bookstore
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

from src.data_processing import load_monitoring_data, compute_temporal_metrics
from src.analysis import analyze_metric_correlations, train_variance_regression


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


def analyze_single_monitor_band(monitor_name, band_config, data_path='./data/raw/rfbaseline/',
                                cutoff_date='2023-01-01'):
    """
    Perform temporal analysis for a single monitor and frequency band.

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
    temporal_metrics : dict
        Temporal analysis results
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

        # Compute temporal metrics
        temporal_metrics = compute_temporal_metrics(df, band_start, band_end,
                                                   threshold_start, threshold_end)

        # Print summary
        print(f"\n  Time of Day Analysis:")
        for period, metrics in temporal_metrics['time_of_day'].items():
            print(f"    {period:12s}: DC={metrics['duty_cycle']:6.2f}%, "
                  f"Pow={metrics['avg_power_occupied']:7.2f} dB, "
                  f"Var={metrics['signal_variation']:6.2f} dB²")

        print(f"\n  Seasonal Analysis:")
        for season, metrics in temporal_metrics['season'].items():
            print(f"    {season:12s}: DC={metrics['duty_cycle']:6.2f}%, "
                  f"Pow={metrics['avg_power_occupied']:7.2f} dB, "
                  f"Var={metrics['signal_variation']:6.2f} dB²")

        return temporal_metrics

    except Exception as e:
        print(f"  ERROR processing {monitor_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_regression_models(occupancy_metrics_path, bands_to_process):
    """
    Train variance regression models for each band (Figure 6).

    Parameters
    ----------
    occupancy_metrics_path : str
        Path to occupancy metrics file
    bands_to_process : dict
        Dictionary of bands to process

    Returns
    -------
    models : dict
        Trained regression models for each band
    """
    print(f"\n{'='*60}")
    print("Training Variance Regression Models (Figure 6)")
    print(f"{'='*60}\n")

    # Load occupancy metrics
    metrics = np.load(occupancy_metrics_path)

    models = {}

    for band_name in bands_to_process.keys():
        print(f"\nBand: {band_name}")

        # Extract RSSI and variance data for this band
        rssi_data = []
        variance_data = []

        for key in metrics.files:
            if band_name in key:
                if '_avg_power' in key:
                    rssi_data.append(metrics[key])
                elif '_variation' in key:
                    variance_data.append(metrics[key])

        rssi_data = np.array(rssi_data)
        variance_data = np.array(variance_data)

        if len(rssi_data) == 0 or len(variance_data) == 0:
            print(f"  WARNING: No data for {band_name}")
            continue

        print(f"  Training on {len(rssi_data)} monitoring stations")
        print(f"  RSSI range: [{rssi_data.min():.2f}, {rssi_data.max():.2f}] dB")
        print(f"  Variance range: [{variance_data.min():.2f}, {variance_data.max():.2f}] dB²")

        # Train regression model
        model = train_variance_regression(rssi_data, variance_data, degree=3, alpha=0.05)

        # Calculate R² score
        from sklearn.metrics import r2_score
        variance_pred = model.predict(rssi_data.reshape(-1, 1))
        r2 = r2_score(variance_data, variance_pred)

        print(f"  ✓ Model trained: R² = {r2:.3f}")

        models[band_name] = {
            'model': model,
            'rssi_data': rssi_data,
            'variance_data': variance_data,
            'r2_score': r2
        }

    return models


def calculate_correlations(occupancy_metrics_path, bands_to_process):
    """
    Calculate correlation coefficients between metrics (Table III).

    Parameters
    ----------
    occupancy_metrics_path : str
        Path to occupancy metrics file
    bands_to_process : dict
        Dictionary of bands to process

    Returns
    -------
    correlations : dict
        Correlation coefficients for each band
    """
    print(f"\n{'='*60}")
    print("Calculating Metric Correlations (Table III)")
    print(f"{'='*60}\n")

    # Load occupancy metrics
    metrics = np.load(occupancy_metrics_path)

    correlations = {}

    for band_name in bands_to_process.keys():
        print(f"\nBand: {band_name}")

        # Extract data for this band
        rssi_data = []
        variance_data = []
        duty_cycle_data = []

        for key in metrics.files:
            if band_name in key:
                if '_avg_power' in key:
                    rssi_data.append(metrics[key])
                elif '_variation' in key:
                    variance_data.append(metrics[key])
                elif '_duty_cycle' in key:
                    duty_cycle_data.append(metrics[key])

        rssi_data = np.array(rssi_data)
        variance_data = np.array(variance_data)
        duty_cycle_data = np.array(duty_cycle_data)

        if len(rssi_data) == 0:
            print(f"  WARNING: No data for {band_name}")
            continue

        # Calculate correlations
        corr = analyze_metric_correlations(rssi_data, variance_data, duty_cycle_data)

        print(f"  Variance vs RSSI:       {corr['variance_mean_power']:7.3f}")
        print(f"  Variance vs Duty Cycle: {corr['variance_duty_cycle']:7.3f}")
        print(f"  Duty Cycle vs RSSI:     {corr['duty_cycle_mean_power']:7.3f}")

        correlations[band_name] = corr

    return correlations


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
    all_temporal_results = {}

    # Process temporal analysis for each band and monitor
    for band_name, band_config in bands_to_process.items():
        print(f"\n{'='*60}")
        print(f"BAND: {band_config['name']}")
        print(f"{'='*60}")

        band_results = {}

        for monitor_name in monitors_to_process:
            print(f"\nProcessing {monitor_name}...")

            temporal_metrics = analyze_single_monitor_band(
                monitor_name, band_config,
                data_path=args.data_path,
                cutoff_date=config['data']['cutoff_date']
            )

            if temporal_metrics:
                band_results[monitor_name] = temporal_metrics

        all_temporal_results[band_name] = band_results

    # Train regression models (Figure 6)
    regression_models = train_regression_models(args.metrics_path, bands_to_process)

    # Calculate correlations (Table III)
    correlations = calculate_correlations(args.metrics_path, bands_to_process)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save temporal metrics
    temporal_file = output_dir / 'temporal_metrics.npz'
    print(f"\n{'='*60}")
    print(f"Saving temporal results to {temporal_file}")
    print(f"{'='*60}\n")

    # Convert to saveable format
    save_dict = {}
    for band_name, band_results in all_temporal_results.items():
        for monitor_name, temporal_metrics in band_results.items():
            for period_type in ['time_of_day', 'season']:
                for period, metrics in temporal_metrics[period_type].items():
                    key_prefix = f"{band_name}_{monitor_name}_{period_type}_{period}"
                    save_dict[f"{key_prefix}_duty_cycle"] = metrics['duty_cycle']
                    save_dict[f"{key_prefix}_avg_power"] = metrics['avg_power_occupied']
                    save_dict[f"{key_prefix}_variation"] = metrics['signal_variation']

    np.savez(temporal_file, **save_dict)

    # Save regression models
    import pickle
    for band_name, model_data in regression_models.items():
        model_file = output_dir / f'variance_regression_model_{band_name}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Saved regression model: {model_file}")

    # Save correlations
    corr_file = output_dir / 'correlations.npz'
    corr_dict = {}
    for band_name, corr in correlations.items():
        for key, val in corr.items():
            corr_dict[f"{band_name}_{key}"] = val
    np.savez(corr_file, **corr_dict)
    print(f"Saved correlations: {corr_file}")

    print(f"\n✓ Temporal analysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform temporal analysis of spectrum occupancy')
    parser.add_argument('--config', type=str, default='config/parameters.yaml',
                       help='Path to configuration file')
    parser.add_argument('--locations', type=str, default='config/monitoring_locations.yaml',
                       help='Path to monitoring locations file')
    parser.add_argument('--data-path', type=str, default='./data/raw/rfbaseline/',
                       help='Path to raw data directory')
    parser.add_argument('--metrics-path', type=str, default='./data/processed/occupancy_metrics.npz',
                       help='Path to occupancy metrics from step 01')
    parser.add_argument('--output-dir', type=str, default='./data/processed/',
                       help='Output directory for temporal analysis results')
    parser.add_argument('--band', type=str, default=None,
                       help='Process only specific band (e.g., "3610-3650")')
    parser.add_argument('--monitor', type=str, default=None,
                       help='Process only specific monitor (e.g., Bookstore)')

    args = parser.parse_args()
    main(args)
