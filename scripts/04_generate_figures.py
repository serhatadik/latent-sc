#!/usr/bin/env python
"""
Pipeline Script 04: Generate Figures

This script generates all publication-quality figures from the processed data:
- Figure 2: Power histograms with thresholds
- Figure 3: Signal estimation process (tx power, PMF, predictions)
- Figure 4: Combined power/duty cycle spatial maps
- Figure 5: Temporal analysis plots
- Figure 6: Variance regression plots
- Figure 7: Signal variation and confidence maps

Usage:
    python 04_generate_figures.py --config config/parameters.yaml
    python 04_generate_figures.py --config config/parameters.yaml --figure 3
    python 04_generate_figures.py --config config/parameters.yaml --band "3610-3650"
"""

import argparse
import os
import sys
import yaml
import numpy as np
import pickle
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.spatial_plots import (
    plot_transmit_power_map, plot_pmf_map, plot_signal_estimates_map,
    plot_power_duty_cycle_combined, plot_variation_confidence_combined
)
from src.visualization.temporal_plots import (
    plot_all_temporal_metrics
)
from src.visualization.analysis_plots import (
    plot_power_histogram, plot_variance_regression, plot_correlation_heatmap,
    plot_mse_comparison
)
from src.data_processing import load_monitoring_data
from src.interpolation import idw_interpolation_map, compute_confidence_map
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


def generate_figure_2(band_name, band_config, monitor_name, data_path, cutoff_date, output_dir):
    """Generate Figure 2: Power histogram with threshold."""
    print(f"\nGenerating Figure 2 for {band_name} @ {monitor_name}...")

    from src.data_processing import load_monitoring_data

    # Load data
    df = load_monitoring_data(
        monitor_name, band_config['start'], band_config['end'],
        base_path=data_path, cutoff_date=cutoff_date
    )

    if df is None or len(df) == 0:
        print(f"  WARNING: No data for {monitor_name}")
        return

    # Plot histogram
    fig, ax = plot_power_histogram(
        power_data=df['power'],
        threshold=band_config['threshold_start'],
        band_start=band_config['start'],
        band_end=band_config['end'],
        monitor_name=monitor_name,
        bins=20,
        save_path=output_dir / f'figure_2_{band_name}_{monitor_name}.png'
    )
    print(f"  ✓ Saved: figure_2_{band_name}_{monitor_name}.png")


def generate_figure_3(band_name, band_config, processed_dir, locations, map_data, output_dir):
    """Generate Figure 3: Signal estimation process (a, b, c)."""
    print(f"\nGenerating Figure 3 for {band_name}...")

    # Load signal estimates
    estimates_file = processed_dir / f'signal_estimates_{band_name}.npz'
    if not estimates_file.exists():
        print(f"  WARNING: Signal estimates not found: {estimates_file}")
        return

    estimates = np.load(estimates_file)

    # Load occupancy metrics for observed powers
    metrics_file = processed_dir / 'occupancy_metrics.npz'
    metrics = np.load(metrics_file)

    # Get monitoring station coordinates and observed powers
    data_points = np.array([loc['coordinates'] for loc in locations])
    observed_powers = []
    for loc in locations:
        key = f"{band_name}_{loc['name']}_avg_power"
        if key in metrics:
            observed_powers.append(metrics[key])
    observed_powers = np.array(observed_powers)

    # Figure 3a: Transmit power map
    fig, ax = plot_transmit_power_map(
        transmit_power_map=estimates['transmit_power_map'],
        data_points=data_points,
        observed_powers=observed_powers,
        UTM_lat=map_data['UTM_lat'],
        UTM_long=map_data['UTM_long'],
        band_name=band_name,
        save_path=output_dir / f'figure_3a_tx_power_{band_name}.png'
    )
    print(f"  ✓ Saved: figure_3a_tx_power_{band_name}.png")

    # Figure 3b: PMF map
    fig, ax = plot_pmf_map(
        pmf=estimates['pmf'],
        data_points=data_points,
        observed_powers=observed_powers,
        UTM_lat=map_data['UTM_lat'],
        UTM_long=map_data['UTM_long'],
        band_name=band_name,
        save_path=output_dir / f'figure_3b_pmf_{band_name}.png'
    )
    print(f"  ✓ Saved: figure_3b_pmf_{band_name}.png")

    # Figure 3c: Signal estimates map
    fig, ax = plot_signal_estimates_map(
        signal_estimates=estimates['signal_estimates'],
        data_points=data_points,
        observed_powers=observed_powers,
        UTM_lat=map_data['UTM_lat'],
        UTM_long=map_data['UTM_long'],
        band_name=band_name,
        save_path=output_dir / f'figure_3c_predictions_{band_name}.png'
    )
    print(f"  ✓ Saved: figure_3c_predictions_{band_name}.png")


def generate_figure_4(band_name, processed_dir, locations, map_data, output_dir):
    """Generate Figure 4: Combined power/duty cycle map."""
    print(f"\nGenerating Figure 4 for {band_name}...")

    # Load signal estimates (for power)
    estimates_file = processed_dir / f'signal_estimates_{band_name}.npz'
    if not estimates_file.exists():
        print(f"  WARNING: Signal estimates not found: {estimates_file}")
        return

    estimates = np.load(estimates_file)

    # Load occupancy metrics (for duty cycle)
    metrics_file = processed_dir / 'occupancy_metrics.npz'
    metrics = np.load(metrics_file)

    # Get data points
    data_points = np.array([loc['coordinates'] for loc in locations])

    # Get duty cycle values
    duty_cycles = []
    for loc in locations:
        key = f"{band_name}_{loc['name']}_duty_cycle"
        if key in metrics:
            duty_cycles.append(metrics[key])
    duty_cycles = np.array(duty_cycles)

    # Interpolate duty cycle to full map using IDW
    duty_cycle_map = idw_interpolation_map(
        x_known=data_points[:, 0],
        y_known=data_points[:, 1],
        z_known=duty_cycles,
        map_shape=map_data['shape'],
        power=2
    )

    # Convert signal estimates to dB for visualization
    from src.utils.conversions import lin_to_dB
    power_map = lin_to_dB(estimates['signal_estimates'])

    # Plot combined visualization
    fig, ax = plot_power_duty_cycle_combined(
        power_map=power_map,
        duty_cycle_map=duty_cycle_map,
        data_points=data_points,
        buildings_map=map_data['buildings'],
        UTM_lat=map_data['UTM_lat'],
        UTM_long=map_data['UTM_long'],
        band_name=band_name,
        block_size=5,
        save_path=output_dir / f'figure_4_combined_{band_name}.png'
    )
    print(f"  ✓ Saved: figure_4_combined_{band_name}.png")


def generate_figure_5(band_name, monitor_name, processed_dir, output_dir):
    """Generate Figure 5: Temporal analysis plots."""
    print(f"\nGenerating Figure 5 for {band_name} @ {monitor_name}...")

    # Load temporal metrics
    temporal_file = processed_dir / 'temporal_metrics.npz'
    if not temporal_file.exists():
        print(f"  WARNING: Temporal metrics not found: {temporal_file}")
        return

    metrics = np.load(temporal_file)

    # Extract temporal metrics for this band/monitor
    temporal_data = {'time_of_day': {}, 'season': {}}

    for period_type in ['time_of_day', 'season']:
        periods = ['morning', 'afternoon', 'night'] if period_type == 'time_of_day' else ['spring', 'summer', 'autumn', 'winter']

        for period in periods:
            key_prefix = f"{band_name}_{monitor_name}_{period_type}_{period}"
            if f"{key_prefix}_duty_cycle" in metrics:
                temporal_data[period_type][period] = {
                    'duty_cycle': metrics[f"{key_prefix}_duty_cycle"],
                    'avg_power_occupied': metrics[f"{key_prefix}_avg_power"],
                    'signal_variation': metrics[f"{key_prefix}_variation"]
                }

    if len(temporal_data['time_of_day']) == 0:
        print(f"  WARNING: No temporal data for {monitor_name}")
        return

    # Plot all temporal metrics
    fig, axes = plot_all_temporal_metrics(
        temporal_metrics=temporal_data,
        monitor_name=monitor_name,
        band_name=band_name,
        save_path=output_dir / f'figure_5_temporal_{band_name}_{monitor_name}.png'
    )
    print(f"  ✓ Saved: figure_5_temporal_{band_name}_{monitor_name}.png")


def generate_figure_6(band_name, processed_dir, output_dir):
    """Generate Figure 6: Variance regression plot."""
    print(f"\nGenerating Figure 6 for {band_name}...")

    # Load regression model
    model_file = processed_dir / f'variance_regression_model_{band_name}.pkl'
    if not model_file.exists():
        print(f"  WARNING: Regression model not found: {model_file}")
        return

    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)

    # Plot regression
    fig, ax = plot_variance_regression(
        rssi_data=model_data['rssi_data'],
        variance_data=model_data['variance_data'],
        model=model_data['model'],
        band_name=band_name,
        save_path=output_dir / f'figure_6_regression_{band_name}.png'
    )
    print(f"  ✓ Saved: figure_6_regression_{band_name}.png")


def generate_figure_7(band_name, processed_dir, locations, map_data, output_dir):
    """Generate Figure 7: Signal variation and confidence maps."""
    print(f"\nGenerating Figure 7 for {band_name}...")

    # Load occupancy metrics
    metrics_file = processed_dir / 'occupancy_metrics.npz'
    metrics = np.load(metrics_file)

    # Get data points and variance values
    data_points = np.array([loc['coordinates'] for loc in locations])
    variance_values = []
    for loc in locations:
        key = f"{band_name}_{loc['name']}_variation"
        if key in metrics:
            variance_values.append(metrics[key])
    variance_values = np.array(variance_values)

    # Interpolate variance to full map using IDW
    variance_map = idw_interpolation_map(
        x_known=data_points[:, 0],
        y_known=data_points[:, 1],
        z_known=variance_values,
        map_shape=map_data['shape'],
        power=2
    )

    # Compute confidence map
    confidence_map = compute_confidence_map(
        data_points=data_points,
        map_shape=map_data['shape'],
        alpha=0.01  # From paper
    )

    # Plot combined visualization
    fig, axes = plot_variation_confidence_combined(
        variation_map=variance_map,
        confidence_map=confidence_map,
        data_points=data_points,
        UTM_lat=map_data['UTM_lat'],
        UTM_long=map_data['UTM_long'],
        band_name=band_name,
        save_path=output_dir / f'figure_7_variation_confidence_{band_name}.png'
    )
    print(f"  ✓ Saved: figure_7_variation_confidence_{band_name}.png")


def main(args):
    """Main figure generation function."""
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

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of bands to process
    if args.band:
        bands_to_process = {args.band: config['frequency_bands'][args.band]}
    else:
        bands_to_process = config['frequency_bands']

    # Process each band
    for band_name, band_config in bands_to_process.items():
        print(f"\n{'='*60}")
        print(f"Generating figures for {band_config['name']}")
        print(f"{'='*60}")

        # Generate requested figures
        if args.figure is None or args.figure == 2:
            # Figure 2: One example monitor
            monitor_name = locations_config[0]['name']
            generate_figure_2(band_name, band_config, monitor_name,
                            args.data_path, config['data']['cutoff_date'], output_dir)

        if args.figure is None or args.figure == 3:
            generate_figure_3(band_name, band_config, Path(args.processed_dir),
                            locations_config, map_data, output_dir)

        if args.figure is None or args.figure == 4:
            generate_figure_4(band_name, Path(args.processed_dir),
                            locations_config, map_data, output_dir)

        if args.figure is None or args.figure == 5:
            # Figure 5: One example monitor
            monitor_name = locations_config[0]['name']
            generate_figure_5(band_name, monitor_name, Path(args.processed_dir), output_dir)

        if args.figure is None or args.figure == 6:
            generate_figure_6(band_name, Path(args.processed_dir), output_dir)

        if args.figure is None or args.figure == 7:
            generate_figure_7(band_name, Path(args.processed_dir),
                            locations_config, map_data, output_dir)

    print(f"\n{'='*60}")
    print(f"✓ All figures saved to {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate publication-quality figures')
    parser.add_argument('--config', type=str, default='config/parameters.yaml',
                       help='Path to configuration file')
    parser.add_argument('--locations', type=str, default='config/monitoring_locations.yaml',
                       help='Path to monitoring locations file')
    parser.add_argument('--map-dir', type=str, default='./',
                       help='Directory containing SLC map file')
    parser.add_argument('--data-path', type=str, default='./data/raw/rfbaseline/',
                       help='Path to raw data directory')
    parser.add_argument('--processed-dir', type=str, default='./data/processed/',
                       help='Directory with processed results from previous steps')
    parser.add_argument('--output-dir', type=str, default='./data/results/figures/',
                       help='Output directory for figures')
    parser.add_argument('--band', type=str, default=None,
                       help='Process only specific band (e.g., "3610-3650")')
    parser.add_argument('--figure', type=int, default=None,
                       help='Generate only specific figure (2-7)')

    args = parser.parse_args()
    main(args)
