"""
Reconstruction Parameter Sweep Script

Evaluates localization performance by sweeping over various reconstruction parameters,
such as sigma_noise strategies and candidate selection methods.

This script:
1. Loads data (from cache or creates from scratch)
2. Defines parameters to sweep (sigma_noise strategies, selection_methods)
3. Runs sparse reconstruction for each configuration
4. Computes localization metrics for comparison
5. Outputs summary table and visualization

Usage:
    python scripts/reconstruction_parameter_sweep.py --transmitters mario,moran --seed 32
    python scripts/reconstruction_parameter_sweep.py --test  # Quick test mode
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import utility functions
from src.utils import (
    load_slc_map,
    load_monitoring_locations,
    get_sensor_locations_array,
    load_transmitter_locations,
)

# Import sparse reconstruction
from src.sparse_reconstruction import (
    joint_sparse_reconstruction,
    dbm_to_linear,
)

# Import evaluation metrics
from src.evaluation.metrics import compute_localization_metrics


def define_sigma_noise_strategies(observed_powers_linear, test_mode=False):
    """
    Define sigma_noise strategies to test.
    
    Parameters
    ----------
    observed_powers_linear : ndarray
        Observed powers in linear scale (mW)
    test_mode : bool
        If True, return a reduced set of strategies for quick testing
        
    Returns
    -------
    dict
        Dictionary of {strategy_name: sigma_noise_value}
    """
    min_power = np.min(observed_powers_linear)
    max_power = np.max(observed_powers_linear)
    mean_power = np.mean(observed_powers_linear)
    power_range = max_power - min_power
    
    if test_mode:
        # Minimal set for quick testing
        return {
            'fixed_1e-9': 1e-9,
            'min_power': min_power,
            '5x_min': 5.0 * min_power,
        }
    
    # Full set of strategies
    strategies = {
        # Fixed values (common choices)
        'fixed_1e-10': 1e-10,
        'fixed_1e-9': 1e-9,
        'fixed_5e-9': 5e-9,
        'fixed_1e-8': 1e-8,
        'fixed_1e-7': 1e-7,
        
        # Min-power based strategies (multiples of min observed power)
        '0.5x_min': 0.5 * min_power,
        'min_power': min_power,
        '2x_min': 2.0 * min_power,
        '5x_min': 5.0 * min_power,
        '10x_min': 10.0 * min_power,
        '20x_min': 20.0 * min_power,
        
        # Sqrt-based (often used for Poisson-like noise)
        'sqrt_min': np.sqrt(min_power),
        'sqrt_mean': np.sqrt(mean_power),
        
        # Mean-power based
        '0.01_mean': 0.01 * mean_power,
        '0.1_mean': 0.1 * mean_power,
    }
    
    return strategies


def run_investigation(transmitters, seed=None, test_mode=False, output_dir=None,
                      whitening_method='hetero_diag', eta=0.01, model_type='tirem', verbose=True):
    """
    Run the sigma_noise investigation.
    
    Parameters
    ----------
    transmitters : list of str
        List of transmitter names (e.g., ['mario', 'moran'])
    seed : int, optional
        Random seed suffix for data directory
    test_mode : bool
        Quick test mode with reduced parameter set
    output_dir : Path, optional
        Directory to save results
    whitening_method : str
        Whitening method to use ('hetero_diag' or 'hetero_geo_aware')
    eta : float
        Eta parameter for hetero_diag whitening
    model_type : str
        Propagation model: 'tirem' (accurate, slow) or 'log_distance' (fast)
    verbose : bool
        Print detailed progress
        
    Returns
    -------
    pd.DataFrame
        Results dataframe with metrics for each strategy
    """
    print("=" * 70)
    print("SIGMA NOISE INVESTIGATION")
    print("=" * 70)
    
    # --- Configuration ---
    tx_underscore = "_".join(transmitters)
    
    if seed is not None:
        data_dir = Path(f'data/processed/{tx_underscore}_seed_{seed}/')
        config_path = f'config/monitoring_locations_{tx_underscore}_seed_{seed}.yaml'
    else:
        data_dir = Path(f'data/processed/{tx_underscore}/')
        config_path = f'config/monitoring_locations_{tx_underscore}.yaml'
    
    print(f"\nConfiguration:")
    print(f"  Transmitters: {transmitters}")
    print(f"  Data directory: {data_dir}")
    print(f"  Config path: {config_path}")
    print(f"  Whitening method: {whitening_method}")
    print(f"  Model type: {model_type}")
    print(f"  Test mode: {test_mode}")
    
    # --- Load Data ---
    print("\n" + "-" * 40)
    print("LOADING DATA")
    print("-" * 40)
    
    # Load main config
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Loaded parameters.yaml")
    
    # Load SLC map
    map_data = load_slc_map(
        map_folder_dir="./",
        downsample_factor=config['spatial']['downsample_factor']
    )
    print(f"✓ Loaded SLC map: shape {map_data['shape']}")
    
    # Load monitoring locations
    locations_config = load_monitoring_locations(
        config_path=config_path,
        map_data=map_data
    )
    sensor_locations = get_sensor_locations_array(locations_config)
    print(f"✓ Loaded {len(sensor_locations)} monitoring locations")
    
    # Load power measurements
    observed_powers_dB = np.load(data_dir / f"{tx_underscore}_avg_powers.npy")
    observed_powers_linear = dbm_to_linear(observed_powers_dB)
    
    # Compute observation statistics
    obs_min_dbm = np.min(observed_powers_dB)
    obs_mean_dbm = np.mean(observed_powers_dB)
    obs_max_dbm = np.max(observed_powers_dB)
    
    print(f"✓ Loaded observed powers:")
    print(f"    dBm range: [{obs_min_dbm:.2f}, {obs_max_dbm:.2f}] (Mean: {obs_mean_dbm:.2f})")
    print(f"    Linear range: [{observed_powers_linear.min():.2e}, {observed_powers_linear.max():.2e}] mW")
    
    # Load transmitter locations
    all_tx_locations = load_transmitter_locations(
        config_path='config/transmitter_locations.yaml',
        map_data=map_data
    )
    tx_locations = {name: all_tx_locations[name] for name in transmitters if name in all_tx_locations}
    true_locs_pixels = np.array([tx['coordinates'] for tx in tx_locations.values()])
    print(f"✓ Loaded {len(tx_locations)} true transmitter location(s)")
    
    # --- Define Strategies ---
    print("\n" + "-" * 40)
    print("DEFINING SIGMA_NOISE STRATEGIES")
    print("-" * 40)
    
    strategies = define_sigma_noise_strategies(observed_powers_linear, test_mode=test_mode)
    
    print(f"Testing {len(strategies)} strategies:")
    for name, value in strategies.items():
        print(f"  {name:20s}: {value:.2e}")
    
    # --- Run Reconstruction Loop ---
    print("\n" + "-" * 40)
    print("RUNNING RECONSTRUCTION EXPERIMENTS")
    print("-" * 40)
    
    results = []
    
    selection_methods = ['max', 'cluster']
    
    for i, (strategy_name, sigma_noise) in enumerate(strategies.items()):
        for sel_method in selection_methods:
            print(f"\n[{i+1}/{len(strategies)}] Strategy: {strategy_name} (σ_noise = {sigma_noise:.2e}) | Selection: {sel_method}")
            
            try:
                start_time = time.time()
                
                print(f"    Starting reconstruction...")
                
                # Run reconstruction
                tx_map, info = joint_sparse_reconstruction(
                    sensor_locations=sensor_locations,
                    observed_powers_dBm=observed_powers_dB,
                    input_is_linear=False,
                    solve_in_linear_domain=True,
                    map_shape=map_data['shape'],
                    scale=config['spatial']['proxel_size'],
                    np_exponent=config['localization']['path_loss_exponent'],
                    lambda_reg=0,  # Use GLRT solver
                    norm_exponent=0,
                    whitening_method=whitening_method,
                    sigma_noise=sigma_noise,
                    eta=eta,
                    solver='glrt',
                    selection_method=sel_method,
                    cluster_max_candidates=30,
                    glrt_max_iter=len(transmitters) + 1,  # Extra iteration to check stopping
                    glrt_threshold=4.0,
                    dedupe_distance_m=25.0,
                    return_linear_scale=False,
                    verbose=True,  # Enable verbose to see progress
                    model_type=model_type,
                    tirem_config_path='config/tirem_parameters.yaml' if model_type == 'tirem' else None,
                    n_jobs=-1
                )
                print(f"    Reconstruction complete.")
            
                elapsed = time.time() - start_time
                
                # Extract estimated locations from solver info
                if 'solver_info' in info and 'support' in info['solver_info']:
                    support_indices = info['solver_info']['support']
                    height, width = map_data['shape']
                    
                    # Filter out transmitters with power below -190 dBm
                    valid_indices = []
                    for idx in support_indices:
                        r, c = idx // width, idx % width
                        power_dbm = tx_map[r, c]
                        if power_dbm > -190:
                            valid_indices.append(idx)
                    
                    est_rows = [idx // width for idx in valid_indices]
                    est_cols = [idx % width for idx in valid_indices]
                    est_locs_pixels = np.column_stack((est_cols, est_rows)) if valid_indices else np.empty((0, 2))
                else:
                    # Fallback: extract from map
                    from src.evaluation.metrics import extract_locations_from_map
                    est_locs_pixels = extract_locations_from_map(tx_map, threshold=1e-10)
                
                # Compute metrics
                metrics = compute_localization_metrics(
                    true_locations=true_locs_pixels,
                    estimated_locations=est_locs_pixels,
                    scale=config['spatial']['proxel_size'],
                    tolerance=200.0  # meters
                )
                
                # Store results
                result = {
                    'strategy': strategy_name,
                    'selection_method': sel_method,
                    'sigma_noise': sigma_noise,
                    'sigma_noise_dB': 10 * np.log10(sigma_noise) if sigma_noise > 0 else -np.inf,
                    'ale': metrics['ale'],
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'fn': metrics['fn'],
                    'pd': metrics['pd'],
                    'precision': metrics['precision'],
                    'f1_score': metrics['f1_score'],
                    'n_estimated': metrics['n_est'],
                    'runtime_s': elapsed,
                    'obs_min_dbm': obs_min_dbm,
                    'obs_mean_dbm': obs_mean_dbm,
                    'obs_max_dbm': obs_max_dbm,
                }
                results.append(result)
                
                print(f"    ALE: {metrics['ale']:.2f} m | TP: {metrics['tp']} | FP: {metrics['fp']} | "
                      f"FN: {metrics['fn']} | Pd: {metrics['pd']*100:.0f}% | "
                      f"Precision: {metrics['precision']*100:.0f}% | Time: {elapsed:.1f}s")
            
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # --- Compile Results ---
    results_df = pd.DataFrame(results)
    
    # Sort by ALE (lower is better), then by F1 (higher is better)
    results_df = results_df.sort_values(['ale', 'f1_score'], ascending=[True, False])
    
    # --- Print Summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\nAll strategies (sorted by ALE):")
    print(results_df.to_string(index=False))
    
    # Find best strategies
    if len(results_df) > 0:
        best_ale = results_df.iloc[0]
        best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
        
        print("\n" + "-" * 40)
        print("BEST STRATEGIES")
        print("-" * 40)
        print(f"\nBest by ALE: {best_ale['strategy']} (σ_noise = {best_ale['sigma_noise']:.2e})")
        print(f"    ALE: {best_ale['ale']:.2f} m, Pd: {best_ale['pd']*100:.0f}%, Precision: {best_ale['precision']*100:.0f}%")
        
        print(f"\nBest by F1: {best_f1['strategy']} (σ_noise = {best_f1['sigma_noise']:.2e})")
        print(f"    F1: {best_f1['f1_score']:.4f}, ALE: {best_f1['ale']:.2f} m")
        
        # Check if dynamic is better than fixed
        fixed_strategies = results_df[results_df['strategy'].str.startswith('fixed')]
        dynamic_strategies = results_df[~results_df['strategy'].str.startswith('fixed')]
        
        if len(fixed_strategies) > 0 and len(dynamic_strategies) > 0:
            best_fixed_ale = fixed_strategies['ale'].min()
            best_dynamic_ale = dynamic_strategies['ale'].min()
            
            print("\n" + "-" * 40)
            print("FIXED vs DYNAMIC COMPARISON")
            print("-" * 40)
            print(f"Best fixed strategy ALE: {best_fixed_ale:.2f} m")
            print(f"Best dynamic strategy ALE: {best_dynamic_ale:.2f} m")
            
            if best_dynamic_ale < best_fixed_ale:
                improvement = (best_fixed_ale - best_dynamic_ale) / best_fixed_ale * 100
                print(f"\n✓ Dynamic strategies OUTPERFORM fixed by {improvement:.1f}%")
            elif best_dynamic_ale > best_fixed_ale:
                degradation = (best_dynamic_ale - best_fixed_ale) / best_fixed_ale * 100
                print(f"\n✗ Dynamic strategies underperform fixed by {degradation:.1f}%")
            else:
                print("\n= Fixed and dynamic strategies perform equally")
    
    # --- Save Results ---
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_dir / "sigma_noise_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")
        
        # Generate plots
        if len(results_df) > 1:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot 1: ALE by strategy
            ax = axes[0]
            ax.barh(results_df['strategy'], results_df['ale'], color='steelblue')
            ax.set_xlabel('Average Localization Error (m)')
            ax.set_title('ALE by Strategy')
            ax.invert_yaxis()
            
            # Plot 2: Detection metrics
            ax = axes[1]
            x = np.arange(len(results_df))
            width = 0.25
            ax.bar(x - width, results_df['tp'], width, label='TP', color='green')
            ax.bar(x, results_df['fp'], width, label='FP', color='red')
            ax.bar(x + width, results_df['fn'], width, label='FN', color='orange')
            ax.set_xticks(x)
            ax.set_xticklabels(results_df['strategy'], rotation=45, ha='right')
            ax.set_ylabel('Count')
            ax.set_title('Detection Metrics')
            ax.legend()
            
            # Plot 3: Pd and Precision
            ax = axes[2]
            ax.bar(x - width/2, results_df['pd'] * 100, width, label='Pd', color='blue')
            ax.bar(x + width/2, results_df['precision'] * 100, width, label='Precision', color='purple')
            ax.set_xticks(x)
            ax.set_xticklabels(results_df['strategy'], rotation=45, ha='right')
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Pd and Precision')
            ax.set_ylim(0, 105)
            ax.legend()
            
            plt.tight_layout()
            plot_path = output_dir / "sigma_noise_comparison.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"✓ Plots saved to: {plot_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Investigate optimal sigma_noise selection for sparse reconstruction.'
    )
    parser.add_argument(
        '--transmitters', type=str, default='mario,moran',
        help='Comma-separated list of transmitters (default: mario,moran)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed suffix for data directory (optional)'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run in test mode with reduced parameter set'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save results (default: results/sigma_noise_investigation_<timestamp>)'
    )
    parser.add_argument(
        '--whitening-method', type=str, default='hetero_diag',
        choices=['hetero_diag', 'hetero_geo_aware'],
        help='Whitening method to use (default: hetero_diag)'
    )
    parser.add_argument(
        '--eta', type=float, default=0.01,
        help='Eta parameter for heteroscedastic whitening (default: 0.01)'
    )
    parser.add_argument(
        '--model-type', type=str, default='tirem',
        choices=['tirem', 'log_distance'],
        help='Propagation model: tirem (accurate, slow) or log_distance (fast, default: tirem)'
    )
    
    args = parser.parse_args()
    
    # Parse transmitters
    transmitters = [t.strip() for t in args.transmitters.split(',')]
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f'results/sigma_noise_investigation_{timestamp}')
    
    # Run investigation
    results_df = run_investigation(
        transmitters=transmitters,
        seed=args.seed,
        test_mode=args.test,
        output_dir=output_dir,
        whitening_method=args.whitening_method,
        eta=args.eta,
        model_type=args.model_type,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    main()
