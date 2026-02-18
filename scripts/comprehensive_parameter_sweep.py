"""
Comprehensive Reconstruction Parameter Sweep Script

Automatically processes all datasets in data/processed/, sweeping over various
reconstruction parameters (sigma_noise strategies, selection methods, feature_rho),
and generates comprehensive analysis reports by transmitter count (1-5) and universally.

This script:
1. Auto-discovers all data directories in data/processed/
2. Groups directories by TX count (1-5)
3. Runs reconstruction with various parameter combinations
4. Generates GLRT visualization for each unique data directory
5. Analyzes results per TX count and universally
6. Generates comprehensive reports and visualizations

Usage:
    python scripts/comprehensive_parameter_sweep.py
    python scripts/comprehensive_parameter_sweep.py --test  # Quick test mode
    python scripts/comprehensive_parameter_sweep.py --tx-counts 1,2,3  # Specific TX counts
    python scripts/comprehensive_parameter_sweep.py --nloc 30  # Only nloc30 directories
"""

# IMPORTANT: Set threading env vars BEFORE importing numpy to prevent deadlocks
# when using multiprocessing on Windows
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import argparse
import multiprocessing
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Re-export everything from the sweep package for backward compatibility.
# Downstream scripts (rerun_analysis.py, reanalyze_legacy_results.py,
# run_bic_analysis.py) do ``from scripts.comprehensive_parameter_sweep import ...``
from scripts.sweep import *  # noqa: F401,F403

# Explicit imports used directly in main()
from scripts.sweep.constants import AVAILABLE_WHITENING_CONFIGS, DESIRED_COLUMN_ORDER
from scripts.sweep.discovery import discover_data_directories
from scripts.sweep.orchestration import run_comprehensive_sweep
from scripts.sweep.results_io import save_bic_results_csv
from scripts.sweep.analysis import (
    analyze_by_tx_count,
    analyze_universal,
    analyze_by_tx_set,
    create_final_results,
)
from scripts.sweep.reporting import (
    generate_bic_analysis_report,
    generate_final_analysis_report,
    generate_analysis_report,
)
from scripts.sweep.plotting import generate_plots

from src.utils import (
    load_slc_map,
    load_transmitter_locations,
)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive parameter sweep across all data directories.'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run in test mode with reduced parameter set and limited directories'
    )
    parser.add_argument(
        '--tx-counts', type=str, default=None,
        help='Comma-separated list of TX counts to process (e.g., 1,2,3). Default: all'
    )
    parser.add_argument(
        '--nloc', type=int, default=None,
        help='Only process directories with this specific num_locations value (e.g., 30 for nloc30)'
    )
    parser.add_argument(
        '--max-dirs', type=int, default=None,
        help='Maximum directories to process per TX count (for testing)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save results (default: results/comprehensive_sweep_<timestamp>)'
    )
    parser.add_argument(
        '--model-type', type=str, default='tirem',
        choices=['tirem', 'log_distance', 'raytracing'],
        help='Default propagation model for both localization and reconstruction (default: tirem)'
    )
    parser.add_argument(
        '--localization-model', type=str, default=None,
        choices=['tirem', 'log_distance', 'raytracing'],
        help='Propagation model for localization/GLRT (overrides --model-type)'
    )
    parser.add_argument(
        '--reconstruction-model', type=str, default=None,
        choices=['tirem', 'log_distance', 'raytracing'],
        help='Propagation model for power recomputation & reconstruction validation (overrides --model-type)'
    )
    parser.add_argument(
        '--eta', type=float, default=0.1,
        help='Eta parameter for heteroscedastic whitening (default: 0.1)'
    )
    parser.add_argument(
        '--no-visualizations', action='store_true',
        help='Disable GLRT visualization generation'
    )
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Number of parallel workers (default: 1 for sequential, -1 for all CPUs minus 1)'
    )
    parser.add_argument(
        '--power-thresholds', type=str, default=None,
        help='Comma-separated list of power density thresholds to sweep (e.g., "0.01,0.1,0.3")'
    )
    parser.add_argument(
        '--whitening-methods', type=str, default=None,
        help='Comma-separated list of whitening methods to sweep (e.g., "hetero_diag,hetero_spatial")'
    )
    parser.add_argument(
        '--beam-width', type=int, default=1,
        help='Beam width for GLRT search (default: 1)'
    )
    parser.add_argument(
        '--max-pool-size', type=int, default=50,
        help='Max candidates for pool refinement (default: 50)'
    )
    parser.add_argument(
        '--use-edf', action='store_true',
        help='Enable Consensus-Based Scoring (EDF) penalty'
    )
    parser.add_argument(
        '--edf-threshold', type=float, default=1.5,
        help='Threshold for EDF penalty (default: 1.5)'
    )
    parser.add_argument(
        '--use-robust-scoring', action='store_true',
        help='Enable Robust GLRT scoring (Huber-like loss)'
    )
    parser.add_argument(
        '--robust-threshold', type=float, default=6.0,
        help='Threshold for robust clipping (default: 6.0)'
    )
    parser.add_argument(
        '--pooling-lambda', type=float, default=0.01,
        help='Regularization constant for pooling refinement (active only if refinement enabled)'
    )
    parser.add_argument(
        '--save-iterations', action='store_true',
        help='Save visualization for each GLRT iteration (default: False)'
    )
    parser.add_argument(
        '--dedupe-distance', type=float, default=60.0,
        help='Distance threshold for post-search transmitter deduplication (default: 60.0 m)'
    )

    # Combinatorial selection arguments
    parser.add_argument(
        '--combo-min-distance', type=float, default=100.0,
        help='Minimum distance between paired TXs in combinatorial selection (default: 100.0 m)'
    )
    parser.add_argument(
        '--combo-max-size', type=int, default=5,
        help='Maximum number of TXs in a combination (default: 5)'
    )
    parser.add_argument(
        '--combo-max-candidates', type=int, default=10,
        help='Maximum number of top candidates to consider for combinations (default: 10)'
    )
    parser.add_argument(
        '--combo-bic-weight', type=float, default=0.05,
        help='BIC penalty weight for model complexity (default: 0.05)'
    )
    parser.add_argument(
        '--combo-max-power-diff', type=float, default=20.0,
        help='Maximum TX power difference in dB for combinations (default: 20.0)'
    )
    parser.add_argument(
        '--combo-sensor-proximity-threshold', type=float, default=100.0,
        help='Distance threshold (m) for sensor proximity penalty (default: 100.0)'
    )
    parser.add_argument(
        '--combo-sensor-proximity-penalty', type=float, default=10.0,
        help='Constant BIC penalty for each TX within proximity threshold of a sensor (default: 10.0)'
    )

    args = parser.parse_args()

    # Resolve model types: specific flags override --model-type
    localization_model = args.localization_model or args.model_type
    reconstruction_model = args.reconstruction_model or args.model_type

    # Parse thresholds list
    if args.power_thresholds:
        power_thresholds = [float(x) for x in args.power_thresholds.split(',')]
    else:
        # Default changed: If flag omitted, disable power filtering entirely
        power_thresholds = None

    # Parse TX counts filter
    tx_counts_filter = None
    if args.tx_counts:
        tx_counts_filter = [int(x.strip()) for x in args.tx_counts.split(',')]

    # Parse whitening methods
    whitening_configs = None
    if args.whitening_methods:
        methods = [x.strip() for x in args.whitening_methods.split(',')]
        whitening_configs = {}
        for m in methods:
            if m in AVAILABLE_WHITENING_CONFIGS:
                whitening_configs[m] = AVAILABLE_WHITENING_CONFIGS[m]
            else:
                print(f"Warning: Unknown whitening method '{m}', skipping. Available: {list(AVAILABLE_WHITENING_CONFIGS.keys())}")

        if not whitening_configs:
            print("Error: No valid whitening methods selected. Exiting.")
            return

    # Set max dirs for test mode
    max_dirs = args.max_dirs
    if args.test and max_dirs is None:
        max_dirs = 2  # Only 2 dirs per TX count in test mode

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f'results/comprehensive_sweep_{timestamp}')

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE RECONSTRUCTION PARAMETER SWEEP")
    print("=" * 70)

    # Load base configuration
    print("\nLoading configuration...")
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load SLC map
    print("Loading SLC map...")
    map_data = load_slc_map(
        map_folder_dir="./",
        downsample_factor=config['spatial']['downsample_factor']
    )
    print(f"  Map shape: {map_data['shape']}")

    # Load transmitter locations
    print("Loading transmitter locations...")
    all_tx_locations = load_transmitter_locations(
        config_path='config/transmitter_locations.yaml',
        map_data=map_data
    )
    print(f"  Transmitters: {list(all_tx_locations.keys())}")

    # Discover data directories
    print("\nDiscovering data directories...")
    base_dir = Path('data/processed')
    grouped_dirs = discover_data_directories(base_dir)

    for tx_count in sorted(grouped_dirs.keys()):
        print(f"  TX count {tx_count}: {len(grouped_dirs[tx_count])} directories")

    # If using custom whitening configs, print them
    if whitening_configs:
        print(f"Custom whitening configs: {list(whitening_configs.keys())}")

    # Run comprehensive sweep
    results_df = run_comprehensive_sweep(
        grouped_dirs=grouped_dirs,
        config=config,
        map_data=map_data,
        all_tx_locations=all_tx_locations,
        output_dir=output_dir,
        test_mode=args.test,
        tx_counts_filter=tx_counts_filter,
        nloc_filter=args.nloc,
        max_dirs_per_count=max_dirs,
        model_type=localization_model,
        recon_model_type=reconstruction_model,
        eta=args.eta,
        save_visualizations=not args.no_visualizations,
        verbose=True,
        n_workers=args.workers,
        power_thresholds=power_thresholds,
        whitening_configs=whitening_configs,
        beam_width=args.beam_width,
        max_pool_size=args.max_pool_size,
        use_edf_penalty=args.use_edf,
        edf_threshold=args.edf_threshold,
        use_robust_scoring=args.use_robust_scoring,
        robust_threshold=args.robust_threshold,

        save_iterations=args.save_iterations,

        pooling_lambda=args.pooling_lambda,
        dedupe_distance_m=args.dedupe_distance,

        # Combinatorial selection parameters
        combo_min_distance_m=args.combo_min_distance,
        combo_max_size=args.combo_max_size,
        combo_max_candidates=args.combo_max_candidates,
        combo_bic_weight=args.combo_bic_weight,
        combo_max_power_diff_dB=args.combo_max_power_diff,
        combo_sensor_proximity_threshold_m=args.combo_sensor_proximity_threshold,
        combo_sensor_proximity_penalty=args.combo_sensor_proximity_penalty,
    )

    if len(results_df) == 0:
        print("\nNo results collected. Exiting.")
        return

    # Sort results for consistent grouping
    results_df = results_df.sort_values(
        by=['dir_name', 'transmitters', 'seed', 'strategy', 'selection_method', 'power_filtering'],
        na_position='last'
    )

    # Reorder columns as requested
    existing_cols = list(results_df.columns)
    ordered_cols = []

    # Add desired columns if they are present in the dataframe
    for col in DESIRED_COLUMN_ORDER:
        if col in existing_cols:
            ordered_cols.append(col)

    # Add remaining columns
    for col in existing_cols:
        if col not in ordered_cols:
            ordered_cols.append(col)

    # Apply reordering
    results_df = results_df[ordered_cols]


    # Save raw results
    results_path = output_dir / 'all_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nRaw results saved to: {results_path}")

    # Save BIC-only results CSV and generate BIC analysis report
    print("\nSaving BIC results...")
    bic_df = save_bic_results_csv(results_df, output_dir)
    if bic_df is not None and len(bic_df) > 0:
        generate_bic_analysis_report(bic_df, output_dir)

    # Create final results (best strategy per directory based on lowest BIC)
    print("\nCreating final results (best strategy per directory)...")
    final_df = create_final_results(results_df, output_dir)
    if final_df is not None and len(final_df) > 0:
        generate_final_analysis_report(final_df, output_dir)
        # Note: Visualization cleanup no longer needed - visualizations are only created
        # for the best BIC strategy per directory during processing

    # Generate analysis
    print("\nGenerating analysis...")
    tx_count_summaries = analyze_by_tx_count(results_df)
    universal_summary = analyze_universal(results_df)

    # Save summaries
    universal_summary.to_csv(output_dir / 'universal_summary.csv', index=False)
    for tx_count, summary in tx_count_summaries.items():
        summary.to_csv(output_dir / f'summary_tx{tx_count}.csv', index=False)

    # Generate report
    report_path = generate_analysis_report(
        results_df=results_df,
        tx_count_summaries=tx_count_summaries,
        universal_summary=universal_summary,
        output_dir=output_dir,
    )
    print(f"Analysis report saved to: {report_path}")

    # Generate per-tx set analysis
    print("\nGenerating per-transmitter set analysis...")
    tx_set_summary = analyze_by_tx_set(results_df)
    tx_set_summary.to_csv(output_dir / 'summary_by_tx_set.csv', index=False)

    # Re-generate report with new data
    report_path = generate_analysis_report(
        results_df=results_df,
        tx_count_summaries=tx_count_summaries,
        universal_summary=universal_summary,
        tx_set_summary=tx_set_summary,
        output_dir=output_dir,
    )
    print(f"Extended analysis report saved to: {report_path}")

    # Generate plots
    generate_plots(
        results_df=results_df,
        tx_count_summaries=tx_count_summaries,
        universal_summary=universal_summary,
        output_dir=output_dir,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"\nTotal experiments: {len(results_df)}")
    print(f"Results directory: {output_dir}")

    # Print best overall
    best = universal_summary.iloc[0]
    print(f"\nBest overall strategy: {best['strategy']} ({best['selection_method']}, {best['whitening_config']})")
    print(f"  Mean ALE: {best['ale_mean']:.2f} m")
    print(f"  Mean Pd: {best['pd_mean']*100:.1f}%")

    return results_df


if __name__ == "__main__":
    # Required for Windows multiprocessing support
    multiprocessing.freeze_support()
    main()
