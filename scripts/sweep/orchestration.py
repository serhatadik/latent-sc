"""Multiprocessing orchestration: worker init, per-directory processing, sweep loop."""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

from src.utils import (
    load_monitoring_locations,
    get_sensor_locations_array,
)
from src.sparse_reconstruction import dbm_to_linear

from .constants import AVAILABLE_WHITENING_CONFIGS
from .discovery import check_tirem_cache_exists, define_sigma_noise_strategies
from .experiment import run_single_experiment
from .results_io import append_results_to_csv


def _worker_init():
    """Initialize worker process - disable numpy threading to prevent deadlocks."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


def process_single_directory(args: Tuple) -> Tuple[List[Dict], str]:
    """
    Process a single data directory with all parameter combinations.

    This is a top-level function for pickling compatibility with multiprocessing.
    All Path objects are converted to strings before being passed here.

    Parameters
    ----------
    args : tuple
        (data_info_serializable, config, map_data, all_tx_locations, output_dir_str,
         test_mode, model_type, recon_model_type, eta, save_visualizations,
         whitening_configs, selection_methods, power_thresholds, beam_width,
         max_pool_size, use_edf_penalty, edf_threshold, ...)

    Returns
    -------
    tuple
        (list of result dictionaries, skip_reason or None)
    """
    (data_info_serializable, config, map_data, all_tx_locations, output_dir_str,
     test_mode, model_type, recon_model_type, eta, save_visualizations, whitening_configs,
     selection_configs, power_thresholds, beam_width, max_pool_size,
     use_edf_penalty, edf_threshold, use_robust_scoring, robust_threshold, save_iterations,
     pooling_lambda, dedupe_distance_m,
     combo_min_distance_m, combo_max_size, combo_max_candidates, combo_bic_weight, combo_max_power_diff_dB,
     combo_sensor_proximity_threshold_m, combo_sensor_proximity_penalty) = args

    # Reconstruct data_info with Path object
    data_info = data_info_serializable.copy()
    data_info['path'] = Path(data_info['path_str'])
    output_dir = Path(output_dir_str)

    results = []
    dir_name = data_info['name']
    transmitters = data_info['transmitters']
    tx_underscore = "_".join(transmitters)
    tx_count = len(transmitters)

    # Load power data for strategy definition
    powers_file = data_info['path'] / f"{tx_underscore}_avg_powers.npy"
    if not powers_file.exists():
        return results, "no powers file"

    observed_powers_dB = np.load(powers_file)
    observed_powers_linear = dbm_to_linear(observed_powers_dB)

    # Check if TIREM cache exists (needed if either model uses TIREM)
    if model_type == 'tirem' or recon_model_type == 'tirem':
        seed = data_info['seed']
        num_locations = data_info.get('num_locations')
        # Build config path (matching the directory naming convention)
        config_id = tx_underscore
        if num_locations is not None:
            config_id = f"{config_id}_nloc{num_locations}"
        if seed is not None:
            config_id = f"{config_id}_seed_{seed}"
        config_path = f'config/monitoring_locations_{config_id}.yaml'

        if not Path(config_path).exists():
            return results, "no config file"

        locations_config = load_monitoring_locations(
            config_path=config_path,
            map_data=map_data
        )
        sensor_locations = get_sensor_locations_array(locations_config)

        features_cached, prop_cached = check_tirem_cache_exists(
            sensor_locations=sensor_locations,
            map_shape=map_data['shape'],
            scale=config['spatial']['proxel_size'],
            tirem_config_path='config/tirem_parameters.yaml',
        )

        # Features cache only needed for localization with hetero_geo_aware whitening
        needs_features = ('hetero_geo_aware' in whitening_configs) and (model_type == 'tirem')
        if not prop_cached or (needs_features and not features_cached):
            return results, f"no TIREM cache (features={features_cached}, prop={prop_cached}, needs_features={needs_features})"

    # Define strategies based on this dataset's observations
    strategies = define_sigma_noise_strategies(observed_powers_linear, test_mode=test_mode)

    attempted = 0
    failed = 0

    try:
        for strategy_name, sigma_noise in strategies.items():
            for sel_method, use_pf in selection_configs:
                # Determine thresholds to test for this selection config
                # If PF is enabled, test all thresholds. If disabled, test only one (value doesn't matter).
                if use_pf:
                    thresholds_to_test = power_thresholds
                else:
                    thresholds_to_test = [0.0] # Dummy value when PF is disabled

                for threshold in thresholds_to_test:
                    for config_name, (whitening_method, feature_rho) in whitening_configs.items():
                        attempted += 1
                        try:
                            # First pass: run WITHOUT visualizations to find best BIC
                            result = run_single_experiment(
                                data_info=data_info,
                                config=config,
                                map_data=map_data,
                                all_tx_locations=all_tx_locations,
                                sigma_noise=sigma_noise,
                                selection_method=sel_method,
                                use_power_filtering=use_pf,
                                whitening_method=whitening_method,
                                feature_rho=feature_rho,
                                whitening_config_name=config_name,
                                strategy_name=strategy_name,
                                model_type=model_type,
                                recon_model_type=recon_model_type,
                                eta=eta,
                                output_dir=output_dir,
                                save_visualization=False,  # Always False in first pass
                                verbose=False,
                                power_density_threshold=threshold,

                                beam_width=beam_width,
                                max_pool_size=max_pool_size,
                                use_edf_penalty=use_edf_penalty,
                                edf_threshold=edf_threshold,
                                use_robust_scoring=use_robust_scoring,
                                robust_threshold=robust_threshold,

                                save_iterations=save_iterations,

                                pooling_lambda=pooling_lambda,
                                dedupe_distance_m=dedupe_distance_m,

                                # Combinatorial selection parameters
                                combo_min_distance_m=combo_min_distance_m,
                                combo_max_size=combo_max_size,
                                combo_max_candidates=combo_max_candidates,
                                combo_bic_weight=combo_bic_weight,
                                combo_max_power_diff_dB=combo_max_power_diff_dB,
                                combo_sensor_proximity_threshold_m=combo_sensor_proximity_threshold_m,
                                combo_sensor_proximity_penalty=combo_sensor_proximity_penalty,
                            )


                            if result is not None:
                                result.update({
                                    'dir_name': dir_name,
                                    'tx_count': tx_count,
                                    'transmitters': ','.join(transmitters),
                                    'seed': data_info['seed'],
                                    'strategy': strategy_name,
                                    'selection_method': sel_method,
                                    'power_filtering': use_pf,
                                    'power_threshold': threshold if use_pf else float('nan'),
                                    'whitening_config': config_name,
                                    'sigma_noise': sigma_noise,
                                    'whitening_config': config_name,
                                    'sigma_noise': sigma_noise,
                                    'sigma_noise_dB': 10 * np.log10(sigma_noise) if sigma_noise > 0 else -np.inf,
                                    'use_edf': use_edf_penalty,
                                    'edf_thresh': edf_threshold if use_edf_penalty else float('nan'),
                                    'use_robust': use_robust_scoring,
                                    'robust_thresh': robust_threshold if use_robust_scoring else float('nan'),
                                    'robust_thresh': robust_threshold if use_robust_scoring else float('nan'),
                                    'pooling_lambda': pooling_lambda,
                                    'dedupe_dist': dedupe_distance_m,
                                    # Store whitening params for potential re-run
                                    '_whitening_method': whitening_method,
                                    '_feature_rho': feature_rho,
                                })
                                results.append(result)
                            else:
                                failed += 1
                        except Exception as inner_exc:
                            failed += 1
                        # Continue to next experiment

                        # --- Intermediate Logging (every 5 experiments) ---
                        if attempted % 5 == 0:
                            n_curr = len(results)
                            best_curr = min(r['ale'] for r in results) if results else float('inf')
                            msg = f"[{dir_name}] Progress: {n_curr}/{attempted} exps | Best ALE: {best_curr:.2f}m"
                            print(f"  {msg}", flush=True)
                            try:
                                log_path = output_dir / "sweep_progress.log"
                                with open(log_path, "a", encoding="utf-8") as f:
                                    timestamp = datetime.now().strftime("%H:%M:%S")
                                    f.write(f"[{timestamp}] {msg}\n")
                            except:
                                pass
    except Exception as e:
        import traceback
        return results, f"EXCEPTION after {attempted} attempts ({failed} failed): {e}\n{traceback.format_exc()}"

    if failed > 0 and len(results) == 0:
        return results, f"all {attempted} experiments failed"

    # --- Logging Results (before visualization re-run) ---
    n_completed = len(results)
    best_ale = float('inf')
    max_prec = 0.0
    max_pd = 0.0
    best_count_err = 0.0

    if n_completed > 0:
        best_ale = min(r['ale'] for r in results)
        max_prec = max(r['precision'] for r in results)
        max_pd = max(r['pd'] for r in results)

        # for count error, we want the one with min absolute value
        errors = [r['count_error'] for r in results]
        best_count_err = min(errors, key=abs)

    # Format message
    msg = (f"[{dir_name}] Comp: {n_completed}/{attempted} | "
           f"Best ALE: {best_ale:.1f}m | "
           f"Max Pd: {max_pd:.2f} | "
           f"Max Prec: {max_prec:.2f} | "
           f"Best C.Err: {best_count_err:.1f}")

    if n_completed > 0:
        # Identify best result
        best_result = min(results, key=lambda r: r['ale'])
        best_params = f"{best_result['strategy']}, {best_result['selection_method']}, {best_result['whitening_config']}"
        if best_result['power_filtering']:
             best_params += f", PF={best_result['power_threshold']}"
        msg += f" | Best Params: [{best_params}]"

    if failed > 0:
        msg += f" | Failed: {failed}"

    # 1. Print to stdout (will be captured by joblib and shown in main process)
    print(f"  {msg}", flush=True)

    # 2. Append to log file
    try:
        log_path = output_dir / "sweep_progress.log"
        with open(log_path, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass # Don't fail the worker just because logging failed

    # --- Re-run best BIC experiment with visualizations (if requested) ---
    if not save_visualizations:
        print(f"  [{dir_name}] Skipping visualizations (disabled)", flush=True)
    elif len(results) == 0:
        print(f"  [{dir_name}] Skipping visualizations (no results)", flush=True)
    if save_visualizations and len(results) > 0:
        # Find result with lowest combo_bic
        valid_bic_results = [r for r in results if 'combo_bic' in r and not np.isnan(r.get('combo_bic', np.nan))]
        if not valid_bic_results:
            # Debug: check what BIC values we have
            bic_values = [r.get('combo_bic', 'MISSING') for r in results[:3]]  # First 3
            print(f"  [{dir_name}] Warning: No valid BIC results for visualization. Sample BIC values: {bic_values}", flush=True)
        if valid_bic_results:
            best_bic_result = min(valid_bic_results, key=lambda r: r['combo_bic'])
            best_bic_strategy = best_bic_result['strategy']
            best_bic_whitening = best_bic_result['whitening_config']
            best_bic_sel_method = best_bic_result['selection_method']
            best_bic_pf = best_bic_result['power_filtering']
            best_bic_threshold = best_bic_result['power_threshold']
            best_bic_sigma = best_bic_result['sigma_noise']
            best_bic_whitening_method = best_bic_result.get('_whitening_method', 'none')
            best_bic_feature_rho = best_bic_result.get('_feature_rho', None)

            print(f"  [{dir_name}] Re-running best BIC experiment for visualizations: "
                  f"{best_bic_strategy}, {best_bic_whitening}, BIC={best_bic_result['combo_bic']:.2f}", flush=True)

            try:
                # Re-run with save_visualization=True
                _ = run_single_experiment(
                    data_info=data_info,
                    config=config,
                    map_data=map_data,
                    all_tx_locations=all_tx_locations,
                    sigma_noise=best_bic_sigma,
                    selection_method=best_bic_sel_method,
                    use_power_filtering=best_bic_pf,
                    whitening_method=best_bic_whitening_method,
                    feature_rho=best_bic_feature_rho,
                    whitening_config_name=best_bic_whitening,
                    strategy_name=best_bic_strategy,
                    model_type=model_type,
                    recon_model_type=recon_model_type,
                    eta=eta,
                    output_dir=output_dir,
                    save_visualization=True,  # NOW enable visualizations
                    verbose=False,
                    power_density_threshold=best_bic_threshold if best_bic_pf else 0.0,
                    beam_width=beam_width,
                    max_pool_size=max_pool_size,
                    use_edf_penalty=use_edf_penalty,
                    edf_threshold=edf_threshold,
                    use_robust_scoring=use_robust_scoring,
                    robust_threshold=robust_threshold,
                    save_iterations=save_iterations,
                    pooling_lambda=pooling_lambda,
                    dedupe_distance_m=dedupe_distance_m,
                    combo_min_distance_m=combo_min_distance_m,
                    combo_max_size=combo_max_size,
                    combo_max_candidates=combo_max_candidates,
                    combo_bic_weight=combo_bic_weight,
                    combo_max_power_diff_dB=combo_max_power_diff_dB,
                    combo_sensor_proximity_threshold_m=combo_sensor_proximity_threshold_m,
                    combo_sensor_proximity_penalty=combo_sensor_proximity_penalty,
                )
            except Exception as viz_exc:
                print(f"  [{dir_name}] Warning: Failed to generate visualizations: {viz_exc}", flush=True)

    return results, None  # None = no skip reason, processing succeeded


def run_comprehensive_sweep(
    grouped_dirs: Dict[int, List[Dict]],
    config: Dict,
    map_data: Dict,
    all_tx_locations: Dict,
    output_dir: Path,
    test_mode: bool = False,
    tx_counts_filter: Optional[List[int]] = None,
    nloc_filter: Optional[int] = None,
    max_dirs_per_count: Optional[int] = None,
    model_type: str = 'tirem',
    recon_model_type: str = 'tirem',
    eta: float = 0.01,
    save_visualizations: bool = True,
    verbose: bool = True,
    n_workers: int = 1,
    power_thresholds: List[float] = None,
    whitening_configs: Dict = None,
    beam_width: int = 1,
    max_pool_size: int = 50,
    use_edf_penalty: bool = False,
    edf_threshold: float = 1.5,
    use_robust_scoring: bool = False,
    robust_threshold: float = 6.0,
    save_iterations: bool = False,

    pooling_lambda: float = 0.01,
    dedupe_distance_m: float = 60.0,

    # Combinatorial selection parameters
    combo_min_distance_m: float = 100.0,
    combo_max_size: int = 5,
    combo_max_candidates: int = 10,
    combo_bic_weight: float = 0.2,
    combo_max_power_diff_dB: float = 20.0,
    combo_sensor_proximity_threshold_m: float = 100.0,
    combo_sensor_proximity_penalty: float = 10.0,
) -> pd.DataFrame:
    """
    Run comprehensive parameter sweep across all directories.

    Parameters
    ----------
    n_workers : int, optional
        Number of parallel workers. Default: 1 (sequential).
        Set to -1 to use all CPUs minus 1.

    Returns
    -------
    pd.DataFrame
        Results dataframe with all experiments
    """
    results = []
    # Selection configurations: (method, use_power_filtering)
    selection_configs = [
        ('max', False),
        ('cluster', False),
        ('max', True),
        ('cluster', True),
    ]

    # If Beam Search is enabled (width > 1), 'max' and 'cluster' inputs yield identical Hybrid results.
    # To avoid 2x redundant computation, prune the list to only use 'max' as the representative input.
    if beam_width > 1:
        if verbose:
             print(f"  Beam Search (width={beam_width}) enabled: Pruning redundant 'cluster' selection methods.")
        selection_configs = [c for c in selection_configs if c[0] == 'max']

    # If no power thresholds are provided, disable power filtering experiments
    if not power_thresholds:
        if verbose:
            print("  No power density thresholds provided: Disabling power filtering experiments.")
        selection_configs = [c for c in selection_configs if not c[1]]

    # Whitening configurations
    if whitening_configs is None:
        whitening_configs = AVAILABLE_WHITENING_CONFIGS
        if test_mode:
             # Only use diagonal in test mode (faster) if not explicitly provided
             whitening_configs = {'hetero_diag': ('hetero_diag', None)}

    print(f"Whitening configs to run: {list(whitening_configs.keys())}")

    # Filter TX counts if specified
    if tx_counts_filter:
        grouped_dirs = {k: v for k, v in grouped_dirs.items() if k in tx_counts_filter}

    # Filter by nloc if specified
    if nloc_filter is not None:
        grouped_dirs = {
            k: [d for d in v if d.get('num_locations') == nloc_filter]
            for k, v in grouped_dirs.items()
        }
        # Remove empty groups
        grouped_dirs = {k: v for k, v in grouped_dirs.items() if v}

    # Flatten directories list with TX count info
    all_dirs = []
    for tx_count in sorted(grouped_dirs.keys()):
        dirs = grouped_dirs[tx_count]
        if max_dirs_per_count:
            dirs = dirs[:max_dirs_per_count]
        all_dirs.extend(dirs)

    total_dirs = len(all_dirs)

    # Determine actual worker count
    if n_workers == -1:
        n_workers = max(1, os.cpu_count() - 1)
    n_workers = min(n_workers, total_dirs)  # Don't use more workers than dirs

    print(f"\n{'='*70}")
    print("COMPREHENSIVE PARAMETER SWEEP")
    print(f"{'='*70}")
    print(f"TX counts to process: {sorted(grouped_dirs.keys())}")
    if nloc_filter is not None:
        print(f"nloc filter: {nloc_filter}")
    print(f"Total directories: {total_dirs}")
    print(f"Workers: {n_workers}")
    print(f"Test mode: {test_mode}")
    print(f"Localization model: {model_type}")
    print(f"Reconstruction model: {recon_model_type}")
    print(f"Whitening configs: {list(whitening_configs.keys())}")
    print(f"EDF Penalty: {use_edf_penalty} (Threshold: {edf_threshold})")
    print(f"Robust Scoring: {use_robust_scoring} (Threshold: {robust_threshold})")
    print(f"Save Iterations: {save_iterations}")
    print(f"Pooling Refinement Lambda: {pooling_lambda}")
    print(f"Dedupe Distance: {dedupe_distance_m}m")
    print(f"\nCombinatorial Selection:")
    print(f"  Min TX Distance: {combo_min_distance_m}m")
    print(f"  Max Combination Size: {combo_max_size}")
    print(f"  Max Candidates: {combo_max_candidates}")
    print(f"  BIC Penalty Weight: {combo_bic_weight}")
    print(f"  Max Power Diff: {combo_max_power_diff_dB}dB")
    print(f"  Sensor Proximity Threshold: {combo_sensor_proximity_threshold_m}m")
    print(f"  Sensor Proximity Penalty: {combo_sensor_proximity_penalty}")

    start_time = time.time()

    # Prepare serializable arguments for each directory
    # Convert Path objects to strings to avoid pickling issues on Windows
    output_dir_str = str(output_dir)

    task_args = []
    for data_info in all_dirs:
        # Create serializable copy of data_info
        data_info_serializable = {
            'name': data_info['name'],
            'transmitters': data_info['transmitters'],
            'num_locations': data_info.get('num_locations'),
            'seed': data_info['seed'],
            'path_str': str(data_info['path']),  # Convert Path to string
        }
        task_args.append((
            data_info_serializable,
            config,
            map_data,
            all_tx_locations,
            output_dir_str,
            test_mode,
            model_type,
            recon_model_type,
            eta,
            save_visualizations,
            whitening_configs,
            selection_configs,
            power_thresholds,
            beam_width,
            max_pool_size,
            use_edf_penalty,
            edf_threshold,
            use_robust_scoring,
            robust_threshold,
            save_iterations,
            pooling_lambda,
            dedupe_distance_m,
            # Combinatorial selection parameters
            combo_min_distance_m,
            combo_max_size,
            combo_max_candidates,
            combo_bic_weight,
            combo_max_power_diff_dB,
            combo_sensor_proximity_threshold_m,
            combo_sensor_proximity_penalty,
        ))

    if n_workers == 1:
        # Sequential execution
        print("\nRunning in sequential mode...")
        for i, args in enumerate(task_args):
            dir_name = args[0]['name']
            print(f"  [{i+1}/{total_dirs}] Processing {dir_name}...", end=" ", flush=True)

            dir_results, skip_reason = process_single_directory(args)

            if skip_reason:
                print(f"SKIPPED ({skip_reason})")
            else:
                print(f"{len(dir_results)} experiments")
                # Incremental save
                results.extend(dir_results)
                append_results_to_csv(dir_results, output_dir)

            # Progress estimate
            elapsed = time.time() - start_time
            if i > 0:
                avg_time = elapsed / (i + 1)
                remaining = (total_dirs - i - 1) * avg_time
                print(f"      Progress: {i+1}/{total_dirs} | "
                      f"Elapsed: {elapsed/60:.1f}min | "
                      f"Remaining: ~{remaining/60:.1f}min")
    else:
        # Parallel execution using concurrent.futures for incremental processing
        print(f"\nRunning in parallel mode with {n_workers} workers...")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(process_single_directory, args): args for args in task_args}

            n_skipped = 0
            n_success = 0

            # Process as they complete
            for i, future in enumerate(as_completed(future_to_args)):
                try:
                    dir_results, skip_reason = future.result()

                    if skip_reason:
                        n_skipped += 1
                    else:
                        n_success += 1
                        results.extend(dir_results)
                        # Incremental save
                        append_results_to_csv(dir_results, output_dir)

                except Exception as exc:
                    print(f"Generated an exception: {exc}")
                    n_skipped += 1

        print(f"  Directories processed: {n_success} success, {n_skipped} skipped")

    elapsed_total = time.time() - start_time
    print(f"\nSweep completed in {elapsed_total/60:.1f} minutes")
    print(f"Total experiments collected: {len(results)}")

    return pd.DataFrame(results)
