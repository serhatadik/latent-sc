
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import time
import itertools
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path.cwd()))

# Import utility functions
from src.utils import (
    load_slc_map, 
    load_monitoring_locations, 
    get_sensor_locations_array,
    load_transmitter_locations
)

# Import SPARSE RECONSTRUCTION modules
from src.sparse_reconstruction import (
    joint_sparse_reconstruction,
    dbm_to_linear
)
from src.visualization.spatial_plots import plot_transmit_power_map

import argparse

def run_hyperparam_sweep(test_mode=False):
    print("="*70)
    print("STARTING HYPERPARAMETER SWEEP")
    if test_mode:
        print("  [TEST MODE ENABLED: Running with minimal parameter set]")
    print("="*70)
    
    # --- Configuration ---
    TRANSMITTERS = ['mario', 'guesthouse']
    TRANSMITTER_UNDERSCORE = "_".join(TRANSMITTERS)
    DATA_DIR = Path(f'data/processed/{TRANSMITTER_UNDERSCORE}/')
    CONFIG_PATH = f'config/monitoring_locations_{TRANSMITTER_UNDERSCORE}.yaml'
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = Path(f'results/hyperparam_sweep_{timestamp}')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")

    # --- Load Data ---
    print("\nLoading data...")
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    map_data = load_slc_map(
        map_folder_dir="./",
        downsample_factor=config['spatial']['downsample_factor']
    )
    
    locations_config = load_monitoring_locations(
        config_path=CONFIG_PATH,
        map_data=map_data
    )
    sensor_locations = get_sensor_locations_array(locations_config)
    
    observed_powers_dB = np.load(DATA_DIR / f"{TRANSMITTER_UNDERSCORE}_avg_powers.npy")
    
    # Load transmitter locations for plotting
    all_tx_locations = load_transmitter_locations(
        config_path='config/transmitter_locations.yaml',
        map_data=map_data
    )
    tx_locations = {name: all_tx_locations[name] for name in TRANSMITTERS if name in all_tx_locations}

    # --- Parameter Grid ---
    if test_mode:
        # Minimal grid for verification
        param_grid = {
            'lambda_reg': [1e8],
            'gamma': [0],
            'exclusion_radius': [50.0],
            'max_l2_norm': [1e-7],
            'norm_exponent': [1.0],
            'whitening_method': ['covariance'],
            'enable_reweighting': [False],
            'proximity_weight': [50.0]
        }
    else:
        # Full grid as defined in implementation plan
        param_grid = {
            'lambda_reg': [1e8, 2e8],
            'gamma': [0],
            'exclusion_radius': [0],
            'max_l2_norm': [1e-7],
            'proximity_weight': [0.0, 500.0],
            'norm_exponent': [0.001, 0.01, 0.1, 0.5],
            'whitening_method': ['covariance', 'diagonal_observation'],
            'enable_reweighting': [False, True]
        }
    
    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_combinations = len(combinations)
    
    print(f"\nTotal combinations to test: {total_combinations}")
    if total_combinations == 0:
        print("WARNING: No combinations generated!")
    if not test_mode:
        print("Note: This may take a long time. Press Ctrl+C to stop.")
    
    # --- Execution Loop ---
    start_time_total = time.time()
    
    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{total_combinations}] Testing parameters: {params}")
        
        try:
            # Setup reweighting params if enabled
            reweight_kwargs = {}
            if params['enable_reweighting']:
                reweight_kwargs = {
                    'max_reweight_iter': 4,
                    'reweight_epsilon': 1e-12,
                    'convergence_tol': 0
                }
            
            # Run reconstruction
            tx_map, info = joint_sparse_reconstruction(
                sensor_locations=sensor_locations,
                observed_powers_dBm=observed_powers_dB,
                map_shape=map_data['shape'],
                scale=config['spatial']['proxel_size'],
                np_exponent=config['localization']['path_loss_exponent'],
                sigma=config['localization']['std_deviation'],
                delta_c=config['localization']['correlation_coeff'],
                lambda_reg=params['lambda_reg'],
                gamma=params['gamma'],
                max_l2_norm=params['max_l2_norm'],
                norm_exponent=params['norm_exponent'],
                whitening_method=params['whitening_method'],
                exclusion_radius=params['exclusion_radius'],
                proximity_weight=params['proximity_weight'],
                enable_reweighting=params['enable_reweighting'],
                solver='scipy',
                return_linear_scale=False, # Return dBm for plotting
                verbose=False, # Reduce output
                **reweight_kwargs
            )
            
            # map_data has UTM_lat and UTM_long arrays
            
            # Generate filename
            w_short = 'cov' if params['whitening_method'] == 'covariance' else 'diag'
            rw_short = 'T' if params['enable_reweighting'] else 'F'
            
            filename = (
                f"L{params['lambda_reg']:.0e}_"
                f"G{params['gamma']:.0e}_"
                f"R{int(params['exclusion_radius'])}_"
                f"P{int(params['proximity_weight'])}_"
                f"M{params['max_l2_norm']:.0e}_"
                f"N{params['norm_exponent']}_"
                f"W{w_short}_"
                f"RW{rw_short}.png"
            )
            
            save_path = RESULTS_DIR / filename
            
            plot_transmit_power_map(
                transmit_power_map=tx_map,
                data_points=sensor_locations, # This expects pixels, which is what we have
                observed_powers=observed_powers_dB,
                UTM_lat=map_data['UTM_lat'],
                UTM_long=map_data['UTM_long'],
                band_name=f"Sweep {i+1}",
                save_path=str(save_path),
                transmitter_locations=tx_locations
            )
            plt.close('all') # Close figure to free memory
            
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    elapsed_total = time.time() - start_time_total
    print("\n" + "="*70)
    print(f"SWEEP COMPLETE in {elapsed_total:.2f} seconds")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep for sparse reconstruction.')
    parser.add_argument('--test', action='store_true', help='Run in test mode with minimal parameters')
    args = parser.parse_args()
    
    run_hyperparam_sweep(test_mode=args.test)
