"""
Ablation Study for TX Localization/Reconstruction Pipeline

Systematically measures the contribution of each algorithmic component
to localization and reconstruction performance.  For each ablation factor,
all other components are held at baseline values while the factor under
test is varied.

Factors tested:
  Binary (on/off):  pool_refinement, physics_filters, hard_filtering,
                    bic_selection, beam_search
  Comparison:       whitening, localization_model, reconstruction_model
  Cross-model:      full localization x reconstruction model matrix

Usage:
    python scripts/run_ablation_study.py --test
    python scripts/run_ablation_study.py --factors whitening,beam_search
    python scripts/run_ablation_study.py --workers 4
"""

# Threading env vars BEFORE numpy (Windows deadlock prevention)
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import copy
import time
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.sweep.discovery import discover_data_directories, define_sigma_noise_strategies
from scripts.sweep.experiment import run_single_experiment
from scripts.sweep.constants import _PROJECT_ROOT
from scripts.ablation_plotting import generate_all_ablation_plots, ABLATION_FACTORS

from src.utils import (
    load_slc_map,
    load_transmitter_locations,
    load_monitoring_locations,
    get_sensor_locations_array,
)
from src.sparse_reconstruction import dbm_to_linear


# ===========================================================================
# Baseline Configuration
# ===========================================================================
BASELINE_CONFIG = {
    # Whitening
    'whitening_method': 'hetero_spatial',
    'whitening_config_name': 'hetero_spatial',
    'feature_rho': None,
    # Beam search
    'beam_width': 3,
    # Selection method
    'selection_method': 'max',
    'use_power_filtering': False,
    'power_density_threshold': 0.3,
    # Pool refinement & dedup
    'pool_refinement': True,
    'dedupe_distance_m': 60.0,
    'max_pool_size': 50,
    # Physics filters
    'max_tx_power_dbm': 40.0,
    'veto_margin_db': 5.0,
    'ceiling_penalty_weight': 0.1,
    # Advanced penalties
    'use_edf_penalty': False,
    'edf_threshold': 1.5,
    'use_robust_scoring': False,
    'robust_threshold': 6.0,
    # Hard filtering
    'rmse_threshold': 20.0,
    'max_error_threshold': 38.0,
    # BIC selection
    'skip_bic_selection': False,
    # Propagation models
    'model_type': 'tirem',
    'recon_model_type': 'tirem',
    # Combinatorial selection
    'combo_min_distance_m': 100.0,
    'combo_max_size': 5,
    'combo_max_candidates': 10,
    'combo_bic_weight': 0.05,
    'combo_max_power_diff_dB': 20.0,
    'combo_sensor_proximity_threshold_m': 100.0,
    'combo_sensor_proximity_penalty': 10.0,
    # Misc
    'pooling_lambda': 0.01,
    'save_visualization': False,
    'save_iterations': False,
    'verbose': False,
}

# ===========================================================================
# Ablation Factor Definitions
# ===========================================================================
# Moved to ablation_plotting.py as ABLATION_FACTORS for shared access.
# Cumulative pipeline stages are defined here for the build-up experiment.

CUMULATIVE_STAGES = [
    # Each stage adds one component on top of the previous.
    # Order: raw greedy -> +physics -> +hard_filter -> +BIC -> +pool_refine -> +beam
    {
        'name': 'Raw GLRT (Greedy)',
        'overrides': {
            'beam_width': 1,
            'pool_refinement': False,
            'dedupe_distance_m': 0,
            'max_tx_power_dbm': 999.0,
            'veto_margin_db': 999.0,
            'ceiling_penalty_weight': 0.0,
            'rmse_threshold': 9999.0,
            'max_error_threshold': 9999.0,
            'skip_bic_selection': True,
        },
    },
    {
        'name': '+Physics Filters',
        'overrides': {
            'beam_width': 1,
            'pool_refinement': False,
            'dedupe_distance_m': 0,
            'max_tx_power_dbm': 40.0,
            'veto_margin_db': 5.0,
            'ceiling_penalty_weight': 0.1,
            'rmse_threshold': 9999.0,
            'max_error_threshold': 9999.0,
            'skip_bic_selection': True,
        },
    },
    {
        'name': '+Hard Filtering',
        'overrides': {
            'beam_width': 1,
            'pool_refinement': False,
            'dedupe_distance_m': 0,
            'max_tx_power_dbm': 40.0,
            'veto_margin_db': 5.0,
            'ceiling_penalty_weight': 0.1,
            'rmse_threshold': 20.0,
            'max_error_threshold': 38.0,
            'skip_bic_selection': True,
        },
    },
    {
        'name': '+BIC Selection',
        'overrides': {
            'beam_width': 1,
            'pool_refinement': False,
            'dedupe_distance_m': 0,
            'max_tx_power_dbm': 40.0,
            'veto_margin_db': 5.0,
            'ceiling_penalty_weight': 0.1,
            'rmse_threshold': 20.0,
            'max_error_threshold': 38.0,
            'skip_bic_selection': False,
        },
    },
    {
        'name': '+Pool Refinement',
        'overrides': {
            'beam_width': 1,
            'pool_refinement': True,
            'dedupe_distance_m': 60.0,
            'max_tx_power_dbm': 40.0,
            'veto_margin_db': 5.0,
            'ceiling_penalty_weight': 0.1,
            'rmse_threshold': 20.0,
            'max_error_threshold': 38.0,
            'skip_bic_selection': False,
        },
    },
    {
        'name': '+Beam Search',
        'overrides': {
            'beam_width': 3,
            'pool_refinement': True,
            'dedupe_distance_m': 60.0,
            'max_tx_power_dbm': 40.0,
            'veto_margin_db': 5.0,
            'ceiling_penalty_weight': 0.1,
            'rmse_threshold': 20.0,
            'max_error_threshold': 38.0,
            'skip_bic_selection': False,
        },
    },
]


# ===========================================================================
# Worker function (top-level for pickling)
# ===========================================================================
def _worker_init():
    """Initialize worker process."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


def _run_variant_for_directory(args):
    """
    Run all sigma strategies for one (factor, variant, directory) combination.

    Returns list of result dicts, each tagged with factor/variant metadata.
    """
    (data_info_ser, config, map_data, all_tx_locations,
     experiment_config, factor_name, variant_name, eta,
     output_dir_str, test_mode) = args

    # Reconstruct Path
    data_info = data_info_ser.copy()
    data_info['path'] = Path(data_info['path_str'])
    output_dir = Path(output_dir_str)

    transmitters = data_info['transmitters']
    tx_underscore = "_".join(transmitters)

    # Load power data
    powers_file = data_info['path'] / f"{tx_underscore}_avg_powers.npy"
    if not powers_file.exists():
        return []

    observed_powers_dB = np.load(powers_file)
    observed_powers_linear = dbm_to_linear(observed_powers_dB)

    # Define sigma strategies
    strategies = define_sigma_noise_strategies(observed_powers_linear, test_mode=test_mode)

    results = []
    for strategy_name, sigma_noise in strategies.items():
        try:
            result = run_single_experiment(
                data_info=data_info,
                config=config,
                map_data=map_data,
                all_tx_locations=all_tx_locations,
                sigma_noise=sigma_noise,
                eta=eta,
                output_dir=output_dir,
                strategy_name=strategy_name,
                # Unpack all experiment config params
                **experiment_config,
            )
            if result is not None:
                result.update({
                    'factor': factor_name,
                    'variant': variant_name,
                    'dir_name': data_info['name'],
                    'tx_count': len(transmitters),
                    'transmitters': ','.join(transmitters),
                    'seed': data_info['seed'],
                    'strategy': strategy_name,
                    'sigma_noise': sigma_noise,
                })
                results.append(result)
        except Exception as e:
            print(f"    FAILED [{factor_name}/{variant_name}/{data_info['name']}/{strategy_name}]: {e}")

    return results


def _build_experiment_config(baseline: dict, overrides: dict) -> dict:
    """Merge variant overrides into baseline to create experiment config."""
    cfg = copy.deepcopy(baseline)
    cfg.update(overrides)

    # Build the kwargs dict for run_single_experiment
    # (exclude keys that are metadata, not function params)
    experiment_params = {
        'selection_method': cfg['selection_method'],
        'use_power_filtering': cfg['use_power_filtering'],
        'whitening_method': cfg['whitening_method'],
        'feature_rho': cfg.get('feature_rho'),
        'whitening_config_name': cfg['whitening_config_name'],
        'power_density_threshold': cfg['power_density_threshold'],
        'model_type': cfg['model_type'],
        'recon_model_type': cfg['recon_model_type'],
        'save_visualization': cfg.get('save_visualization', False),
        'verbose': cfg.get('verbose', False),
        'beam_width': cfg['beam_width'],
        'max_pool_size': cfg['max_pool_size'],
        'use_edf_penalty': cfg['use_edf_penalty'],
        'edf_threshold': cfg['edf_threshold'],
        'use_robust_scoring': cfg['use_robust_scoring'],
        'robust_threshold': cfg['robust_threshold'],
        'save_iterations': cfg.get('save_iterations', False),
        'pooling_lambda': cfg['pooling_lambda'],
        'dedupe_distance_m': cfg['dedupe_distance_m'],
        'combo_min_distance_m': cfg['combo_min_distance_m'],
        'combo_max_size': cfg['combo_max_size'],
        'combo_max_candidates': cfg['combo_max_candidates'],
        'combo_bic_weight': cfg['combo_bic_weight'],
        'combo_max_power_diff_dB': cfg['combo_max_power_diff_dB'],
        'combo_sensor_proximity_threshold_m': cfg['combo_sensor_proximity_threshold_m'],
        'combo_sensor_proximity_penalty': cfg['combo_sensor_proximity_penalty'],
        # Ablation toggles
        'pool_refinement': cfg['pool_refinement'],
        'max_tx_power_dbm': cfg['max_tx_power_dbm'],
        'veto_margin_db': cfg['veto_margin_db'],
        'ceiling_penalty_weight': cfg['ceiling_penalty_weight'],
        'rmse_threshold': cfg['rmse_threshold'],
        'max_error_threshold': cfg['max_error_threshold'],
        'skip_bic_selection': cfg['skip_bic_selection'],
    }
    return experiment_params


def _select_best_per_directory(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (factor, variant, dir_name), select the best strategy by:
    - lowest combo_bic if BIC is available
    - lowest combo_ale otherwise
    """
    if results_df.empty:
        return results_df

    group_cols = ['factor', 'variant', 'dir_name']
    best_indices = []

    for _, group in results_df.groupby(group_cols):
        valid_bic = group[group['combo_bic'].notna() & np.isfinite(group['combo_bic'])]
        if not valid_bic.empty:
            best_indices.append(valid_bic['combo_bic'].idxmin())
        else:
            valid_ale = group[group['combo_ale'].notna()]
            if not valid_ale.empty:
                best_indices.append(valid_ale['combo_ale'].idxmin())
            else:
                best_indices.append(group.index[0])

    return results_df.loc[best_indices].reset_index(drop=True)


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Ablation study for TX localization/reconstruction pipeline.'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Results directory (default: results/ablation_study_<timestamp>)'
    )
    parser.add_argument(
        '--factors', type=str, default=None,
        help='Comma-separated factors to ablate (default: all). '
             'Options: whitening,beam_search,pool_refinement,physics_filters,'
             'hard_filtering,bic_selection,localization_model,reconstruction_model,'
             'cross_model,cumulative'
    )
    parser.add_argument(
        '--tx-counts', type=str, default=None,
        help='Comma-separated TX counts to process (e.g., 1,2,3)'
    )
    parser.add_argument(
        '--nloc', type=int, default=None,
        help='Filter to specific num_locations value'
    )
    parser.add_argument(
        '--max-dirs', type=int, default=None,
        help='Max directories per TX count'
    )
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Parallel workers (default: 1, -1 = all CPUs minus 1)'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Quick test mode (2 dirs max, reduced strategies)'
    )
    parser.add_argument(
        '--eta', type=float, default=0.1,
        help='Eta for heteroscedastic whitening (default: 0.1)'
    )
    parser.add_argument(
        '--baseline-model', type=str, default='tirem',
        choices=['tirem', 'log_distance', 'raytracing'],
        help='Baseline propagation model for both localization and reconstruction (default: tirem)'
    )
    parser.add_argument(
        '--baseline-localization-model', type=str, default=None,
        choices=['tirem', 'log_distance', 'raytracing'],
        help='Baseline localization model (overrides --baseline-model for localization)'
    )
    parser.add_argument(
        '--baseline-reconstruction-model', type=str, default=None,
        choices=['tirem', 'log_distance', 'raytracing'],
        help='Baseline reconstruction model (overrides --baseline-model for reconstruction)'
    )
    args = parser.parse_args()

    # -- Output directory --
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Encode CLI args into directory name
        name_parts = [f'ablation_study_{timestamp}']
        if args.factors:
            name_parts.append(f'factors_{args.factors.replace(",", "-")}')
        if args.tx_counts:
            name_parts.append(f'tx_{args.tx_counts.replace(",", "-")}')
        if args.nloc is not None:
            name_parts.append(f'nloc_{args.nloc}')
        if args.max_dirs is not None:
            name_parts.append(f'maxdirs_{args.max_dirs}')
        loc_model = args.baseline_localization_model or args.baseline_model
        recon_model = args.baseline_reconstruction_model or args.baseline_model
        if loc_model == recon_model:
            if loc_model != 'tirem':
                name_parts.append(f'model_{loc_model}')
        else:
            name_parts.append(f'loc_{loc_model}_recon_{recon_model}')
        if args.eta != 0.1:
            name_parts.append(f'eta_{args.eta}')
        if args.test:
            name_parts.append('test')
        output_dir = Path('results') / '_'.join(name_parts)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Parse filters --
    tx_counts_filter = None
    if args.tx_counts:
        tx_counts_filter = [int(x.strip()) for x in args.tx_counts.split(',')]

    max_dirs = args.max_dirs
    if args.test and max_dirs is None:
        max_dirs = 2

    # -- Select factors --
    all_factor_keys = list(ABLATION_FACTORS.keys()) + ['cross_model', 'cumulative']
    if args.factors:
        selected_factors = [f.strip() for f in args.factors.split(',')]
        invalid = [f for f in selected_factors if f not in all_factor_keys]
        if invalid:
            print(f"Warning: Unknown factors {invalid}. Available: {all_factor_keys}")
        selected_factors = [f for f in selected_factors if f in all_factor_keys]
    else:
        selected_factors = all_factor_keys

    # Update baseline models (specific flags override --baseline-model)
    baseline_loc_model = args.baseline_localization_model or args.baseline_model
    baseline_recon_model = args.baseline_reconstruction_model or args.baseline_model
    BASELINE_CONFIG['model_type'] = baseline_loc_model
    BASELINE_CONFIG['recon_model_type'] = baseline_recon_model

    # -- Load data --
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    print("\nLoading configuration...")
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Loading SLC map...")
    map_data = load_slc_map(
        map_folder_dir="./",
        downsample_factor=config['spatial']['downsample_factor']
    )
    print(f"  Map shape: {map_data['shape']}")

    print("Loading transmitter locations...")
    all_tx_locations = load_transmitter_locations(
        config_path='config/transmitter_locations.yaml',
        map_data=map_data
    )
    print(f"  Transmitters: {list(all_tx_locations.keys())}")

    print("\nDiscovering data directories...")
    base_dir = Path('data/processed')
    grouped_dirs = discover_data_directories(base_dir)

    # Apply filters
    if tx_counts_filter:
        grouped_dirs = {k: v for k, v in grouped_dirs.items() if k in tx_counts_filter}
    if args.nloc is not None:
        grouped_dirs = {
            k: [d for d in v if d.get('num_locations') == args.nloc]
            for k, v in grouped_dirs.items()
        }
        grouped_dirs = {k: v for k, v in grouped_dirs.items() if v}

    # Flatten
    all_dirs = []
    for tc in sorted(grouped_dirs.keys()):
        dirs = grouped_dirs[tc]
        if max_dirs:
            dirs = dirs[:max_dirs]
        all_dirs.extend(dirs)

    total_dirs = len(all_dirs)
    print(f"  Total directories: {total_dirs}")
    print(f"  TX counts: {sorted(grouped_dirs.keys())}")
    print(f"  Selected factors: {selected_factors}")
    print(f"  Baseline localization model: {baseline_loc_model}")
    print(f"  Baseline reconstruction model: {baseline_recon_model}")
    print(f"  Workers: {args.workers}")
    print(f"  Test mode: {args.test}")

    if total_dirs == 0:
        print("\nNo data directories found. Exiting.")
        return

    # -- Build task list --
    print("\n" + "-" * 70)
    print("Building experiment configurations...")
    print("-" * 70)

    tasks = []  # List of (factor_name, variant_name, data_info, experiment_config)

    # Serializable data_info list
    all_dirs_ser = []
    for d in all_dirs:
        d_ser = {
            'name': d['name'],
            'transmitters': d['transmitters'],
            'num_locations': d.get('num_locations'),
            'seed': d['seed'],
            'path_str': str(d['path']),
        }
        all_dirs_ser.append(d_ser)

    # Standard factors (binary + comparison)
    standard_factors = [f for f in selected_factors
                        if f in ABLATION_FACTORS and f not in ('cross_model', 'cumulative')]

    for factor_name in standard_factors:
        finfo = ABLATION_FACTORS[factor_name]
        for variant in finfo['variants']:
            vname = variant['name']
            overrides = {k: v for k, v in variant.items() if k != 'name'}
            experiment_config = _build_experiment_config(BASELINE_CONFIG, overrides)

            for d_ser in all_dirs_ser:
                tasks.append((
                    d_ser, config, map_data, all_tx_locations,
                    experiment_config, factor_name, vname, args.eta,
                    str(output_dir), args.test,
                ))

    # Cross-model factor (3x3 matrix)
    if 'cross_model' in selected_factors:
        loc_models = ['log_distance', 'tirem', 'raytracing']
        recon_models = ['log_distance', 'tirem', 'raytracing']
        for lm in loc_models:
            for rm in recon_models:
                vname = f'{lm}/{rm}'
                overrides = {'model_type': lm, 'recon_model_type': rm}
                experiment_config = _build_experiment_config(BASELINE_CONFIG, overrides)
                for d_ser in all_dirs_ser:
                    tasks.append((
                        d_ser, config, map_data, all_tx_locations,
                        experiment_config, 'cross_model', vname, args.eta,
                        str(output_dir), args.test,
                    ))

    # Cumulative pipeline build-up
    if 'cumulative' in selected_factors:
        for stage in CUMULATIVE_STAGES:
            experiment_config = _build_experiment_config(BASELINE_CONFIG, stage['overrides'])
            for d_ser in all_dirs_ser:
                tasks.append((
                    d_ser, config, map_data, all_tx_locations,
                    experiment_config, 'cumulative', stage['name'], args.eta,
                    str(output_dir), args.test,
                ))

    total_tasks = len(tasks)
    print(f"  Total experiment tasks: {total_tasks}")

    # -- Execute --
    print("\n" + "-" * 70)
    print("Running experiments...")
    print("-" * 70)

    n_workers = args.workers
    if n_workers == -1:
        n_workers = max(1, os.cpu_count() - 1)
    n_workers = min(n_workers, total_tasks)

    all_results = []
    start_time = time.time()

    if n_workers <= 1:
        # Sequential execution
        for i, task_args in enumerate(tasks):
            factor = task_args[5]
            variant = task_args[6]
            dir_name = task_args[0]['name']

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total_tasks - i - 1) / rate / 60 if rate > 0 else 0
                print(f"  [{i+1}/{total_tasks}] {factor}/{variant}/{dir_name} "
                      f"({elapsed/60:.1f}min elapsed, ~{remaining:.1f}min remaining)")

            results = _run_variant_for_directory(task_args)
            all_results.extend(results)
    else:
        print(f"  Using {n_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as executor:
            futures = {executor.submit(_run_variant_for_directory, t): t for t in tasks}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"  Task failed: {e}")

                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Completed {completed}/{total_tasks} tasks "
                          f"({elapsed/60:.1f}min elapsed, "
                          f"{len(all_results)} results collected)")

    elapsed_total = time.time() - start_time
    print(f"\nAll experiments completed in {elapsed_total/60:.1f} minutes")
    print(f"Total raw results: {len(all_results)}")

    if not all_results:
        print("No results collected. Check data directories and TIREM cache.")
        return

    # -- Aggregate results --
    print("\n" + "-" * 70)
    print("Aggregating results...")
    print("-" * 70)

    raw_df = pd.DataFrame(all_results)
    raw_csv_path = output_dir / 'ablation_raw_results.csv'
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"  Raw results saved: {raw_csv_path} ({len(raw_df)} rows)")

    # Select best strategy per (factor, variant, directory)
    best_df = _select_best_per_directory(raw_df)
    best_csv_path = output_dir / 'ablation_best_results.csv'
    best_df.to_csv(best_csv_path, index=False)
    print(f"  Best-per-directory results: {best_csv_path} ({len(best_df)} rows)")

    # Separate cross-model results
    cross_model_df = None
    if 'cross_model' in best_df['factor'].unique():
        cm_df = best_df[best_df['factor'] == 'cross_model'].copy()
        # Parse loc_model and recon_model from variant name
        cm_df[['loc_model', 'recon_model']] = cm_df['variant'].str.split('/', expand=True)
        cross_model_df = cm_df
        cross_csv = output_dir / 'ablation_cross_model.csv'
        cross_model_df.to_csv(cross_csv, index=False)
        print(f"  Cross-model results: {cross_csv}")

    # Separate cumulative results
    cumulative_df = None
    if 'cumulative' in best_df['factor'].unique():
        cumulative_df = best_df[best_df['factor'] == 'cumulative'].copy()

    # Summary statistics per (factor, variant)
    summary_rows = []
    for (factor, variant), group in best_df.groupby(['factor', 'variant']):
        row = {
            'factor': factor,
            'variant': variant,
            'n_dirs': len(group),
        }
        for metric in ['combo_ale', 'combo_pd', 'combo_precision', 'combo_count_error',
                        'recon_rmse', 'recon_mae', 'recon_bias', 'recon_max_error',
                        'ale', 'pd', 'precision', 'f1_score']:
            vals = group[metric].dropna()
            if len(vals) > 0:
                row[f'{metric}_mean'] = vals.mean()
                row[f'{metric}_std'] = vals.std()
                row[f'{metric}_median'] = vals.median()
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
                row[f'{metric}_median'] = np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / 'ablation_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Summary: {summary_csv}")

    # Print summary table
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)

    for factor in summary_df['factor'].unique():
        fdf = summary_df[summary_df['factor'] == factor]
        print(f"\n  {factor}:")
        print(f"  {'Variant':<25s} {'ALE (m)':<12s} {'Pd':<10s} {'R-RMSE (dB)':<14s} {'R-MAE (dB)':<12s}")
        print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*14} {'-'*12}")
        for _, row in fdf.iterrows():
            ale = f"{row['combo_ale_mean']:.1f}±{row['combo_ale_std']:.1f}" if np.isfinite(row['combo_ale_mean']) else 'N/A'
            pd_val = f"{row['combo_pd_mean']:.2f}±{row['combo_pd_std']:.2f}" if np.isfinite(row['combo_pd_mean']) else 'N/A'
            rmse = f"{row['recon_rmse_mean']:.1f}±{row['recon_rmse_std']:.1f}" if np.isfinite(row['recon_rmse_mean']) else 'N/A'
            mae = f"{row['recon_mae_mean']:.1f}±{row['recon_mae_std']:.1f}" if np.isfinite(row['recon_mae_mean']) else 'N/A'
            print(f"  {row['variant']:<25s} {ale:<12s} {pd_val:<10s} {rmse:<14s} {mae:<12s}")

    # -- Generate plots --
    print("\n" + "-" * 70)
    print("Generating plots...")
    print("-" * 70)

    plots_dir = output_dir / 'plots'
    generate_all_ablation_plots(
        results_df=best_df,
        factors=ABLATION_FACTORS,
        output_dir=plots_dir,
        cross_model_df=cross_model_df,
        cumulative_df=cumulative_df,
    )

    print(f"\n{'=' * 70}")
    print(f"Ablation study complete!")
    print(f"  Results: {output_dir}")
    print(f"  Plots:   {plots_dir}")
    print(f"  Runtime: {elapsed_total/60:.1f} minutes")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
