"""Core experiment execution: run_single_experiment and GLRT visualization."""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from src.utils import (
    load_monitoring_locations,
    get_sensor_locations_array,
)
from src.sparse_reconstruction import (
    joint_sparse_reconstruction,
    dbm_to_linear,
)
from src.evaluation.metrics import compute_localization_metrics
from src.evaluation.reconstruction_validation import compute_reconstruction_error
from scripts.candidate_analysis import (
    compute_candidate_power_rmse,
    filter_candidates_by_rmse,
    save_candidate_power_analysis,
    run_combinatorial_selection,
    recompute_powers_with_propagation_model,
    refit_with_per_tx_exponents,
)


def save_glrt_visualization(
    info: Dict,
    map_data: Dict,
    sensor_locations: np.ndarray,
    observed_powers_dB: np.ndarray,
    tx_locations: Dict,
    output_dir: Path,
    experiment_name: str,
    rmse_filtered_support: Optional[List[int]] = None,
    save_iterations: bool = False,
):
    """
    Save GLRT iteration history visualization to files.

    Parameters
    ----------
    info : dict
        Reconstruction info containing solver_info with candidates_history
    map_data : dict
        Map data with shape and UTM coordinates
    sensor_locations : ndarray
        Sensor locations in pixel coordinates
    observed_powers_dB : ndarray
        Observed powers in dBm
    tx_locations : dict
        True transmitter locations
    output_dir : Path
        Directory to save visualization figures
    experiment_name : str
        Name for this experiment (used in filenames)
    rmse_filtered_support : list, optional
        List of grid indices that passed RMSE filtering. If provided, these
        are shown as magenta stars with numbers, while other candidates are
        shown as black stars (filtered out).
    save_iterations : bool
        If True, save visualization for each GLRT iteration. Default False.
    """

    if 'solver_info' not in info or 'candidates_history' not in info['solver_info']:
        return

    solver_info = info['solver_info']
    history = solver_info['candidates_history']

    if len(history) == 0:
        return

    # Create output directory
    vis_dir = output_dir / 'glrt_visualizations' / experiment_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    whitening_method = solver_info.get('whitening_method', 'unknown')

    # Get true TX coordinates
    tx_coords = np.array([tx['coordinates'] for tx in tx_locations.values()])

    # Only save iteration history if requested
    if save_iterations:
        for item in history:
            height, width = map_data['shape']
            score_map = np.zeros((height, width))


        top_indices = item['top_indices']
        top_scores = item['top_scores']

        # Fill scores
        rows, cols = np.unravel_index(top_indices, (height, width))
        score_map[rows, cols] = top_scores

        # Determine score label
        if whitening_method == 'hetero_geo_aware':
            display_score = item.get('normalized_score', item['selected_score'])
            score_label = "Corrected Score"
        else:
            display_score = item['selected_score']
            score_label = "GLRT Score"

        # Create figure
        fig = plt.figure(figsize=(13, 8))
        ax = fig.gca()

        # Plot sensors
        scatter = ax.scatter(sensor_locations[:, 0], sensor_locations[:, 1],
                             c=observed_powers_dB, s=150, edgecolor='green',
                             linewidth=2, cmap='hot', label='Monitoring Locations', zorder=6)

        # Plot true TX locations
        if len(tx_coords) > 0:
            ax.scatter(tx_coords[:, 0], tx_coords[:, 1],
                       marker='x', s=200, c='blue', linewidth=3,
                       label='True Transmitter Locations', zorder=10)

        # Plot candidates (sparse score map)
        nonzero_mask = score_map > 0
        if np.sum(nonzero_mask) > 0:
            nonzero_indices = np.argwhere(nonzero_mask)
            nonzero_row = nonzero_indices[:, 0]
            nonzero_col = nonzero_indices[:, 1]
            nonzero_values = score_map[nonzero_mask]

            sparse_scatter = ax.scatter(nonzero_col, nonzero_row,
                                        c=nonzero_values, s=300, marker='s',
                                        cmap='viridis', edgecolor='black', linewidth=1,
                                        label='Candidates', zorder=5)

            cbar = plt.colorbar(sparse_scatter, label=score_label)
            cbar.ax.tick_params(labelsize=18)
            cbar.set_label(label=score_label, size=18)
            cbar.ax.set_position([0.77, 0.1, 0.04, 0.8])

        # Highlight selected (All candidates in the beam for this iteration)
        # Check for new 'beam_selected_indices' key, fall back to single 'selected_index'
        beam_indices = item.get('beam_selected_indices', [item.get('selected_index')])
        beam_indices = [idx for idx in beam_indices if idx is not None]

        if beam_indices:
            sel_rows, sel_cols = np.unravel_index(beam_indices, map_data['shape'])
            ax.scatter(sel_cols, sel_rows, c='magenta', marker='*', s=400, label='Selected Candidate(s)', zorder=11)

        # UTM ticks
        UTM_lat = map_data['UTM_lat']
        UTM_long = map_data['UTM_long']
        interval = max(1, len(UTM_lat) // 5)
        tick_values = list(range(0, len(UTM_lat), interval))
        tick_labels = ['{:.1f}'.format(lat) for lat in UTM_lat[::interval]]
        plt.xticks(ticks=tick_values, labels=tick_labels, fontsize=14, rotation=0)

        interval = max(1, len(UTM_long) // 5)
        tick_values = list(range(0, len(UTM_long), interval))
        tick_labels = ['{:.1f}'.format(lat) for lat in UTM_long[::interval]]
        plt.yticks(ticks=[0] + tick_values[1:], labels=[""] + tick_labels[1:], fontsize=14, rotation=90)

        ax.set_xlim([0, map_data['shape'][1]])
        ax.set_ylim([0, map_data['shape'][0]])

        plt.xlabel('UTM$_E$ [m]', fontsize=18, labelpad=10)
        plt.ylabel('UTM$_N$ [m]', fontsize=18, labelpad=10)

        plt.title(f"GLRT Iteration {item['iteration']} ({score_label}: {display_score:.4f})", fontsize=20)

        scatter_cbar = plt.colorbar(scatter, label='Observed Signal [dBm]', location='left')
        scatter_cbar.ax.tick_params(labelsize=18)
        scatter_cbar.set_label(label='Observed Signal [dBm]', size=18)
        scatter_cbar.ax.set_position([0.18, 0.1, 0.04, 0.8])

        plt.legend(loc='upper right')

        # Save figure
        fig_path = vis_dir / f"iter_{item['iteration']:02d}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # === Power Density Visualization (separate figure) ===
        power_density_info = item.get('power_density_info')
        if power_density_info is not None and 'power_density' in power_density_info:
            # Create power density map visualization
            power_density = power_density_info['power_density']
            density_mask = power_density_info['density_mask']
            threshold = power_density_info['threshold']

            # Reshape to 2D
            density_map = power_density.reshape((height, width))
            mask_map = density_mask.reshape((height, width))

            # Create masked density (areas below threshold shown as NaN)
            density_thresholded = density_map.copy()
            density_thresholded[mask_map] = np.nan  # Mask out low-density areas

            fig = plt.figure(figsize=(14, 8))
            ax = fig.gca()

            # Plot the full density map with transparency
            im_full = ax.imshow(density_map, origin='lower', cmap='Blues', alpha=0.3,
                               vmin=0, vmax=1, aspect='auto')

            # Overlay the thresholded (valid) density regions
            im_valid = ax.imshow(density_thresholded, origin='lower', cmap='Reds',
                                vmin=threshold, vmax=1, aspect='auto')

            # Plot sensors with power indication
            scatter = ax.scatter(sensor_locations[:, 0], sensor_locations[:, 1],
                                c=observed_powers_dB, s=200, edgecolor='black',
                                linewidth=2, cmap='hot', label='Sensors (by power)', zorder=8)

            # Plot true TX locations
            if len(tx_coords) > 0:
                ax.scatter(tx_coords[:, 0], tx_coords[:, 1],
                          marker='x', s=250, c='blue', linewidth=4,
                          label='True TX Locations', zorder=10)

            # Highlight selected candidate
            sel_row, sel_col = np.unravel_index(item['selected_index'], (height, width))
            ax.scatter([sel_col], [sel_row], c='magenta', marker='*', s=500,
                      edgecolor='white', linewidth=2, label='Selected', zorder=11)

            # Colorbar for density
            cbar = plt.colorbar(im_valid, ax=ax, label='Power Density (thresholded)', shrink=0.8)
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('Power Density (above threshold)', size=14)

            # UTM ticks
            UTM_lat = map_data['UTM_lat']
            UTM_long = map_data['UTM_long']
            interval = max(1, len(UTM_lat) // 5)
            tick_values = list(range(0, len(UTM_lat), interval))
            tick_labels = ['{:.1f}'.format(lat) for lat in UTM_lat[::interval]]
            plt.xticks(ticks=tick_values, labels=tick_labels, fontsize=12, rotation=0)

            interval = max(1, len(UTM_long) // 5)
            tick_values = list(range(0, len(UTM_long), interval))
            tick_labels = ['{:.1f}'.format(lat) for lat in UTM_long[::interval]]
            plt.yticks(ticks=[0] + tick_values[1:], labels=[""] + tick_labels[1:], fontsize=12, rotation=90)

            ax.set_xlim([0, width])
            ax.set_ylim([0, height])

            plt.xlabel('UTM$_E$ [m]', fontsize=16, labelpad=10)
            plt.ylabel('UTM$_N$ [m]', fontsize=16, labelpad=10)

            n_masked = power_density_info['n_masked']
            n_total = len(power_density)
            pct_valid = (n_total - n_masked) / n_total * 100

            plt.title(f"Power Density Map - Iter {item['iteration']} | "
                     f"Threshold: {threshold:.0%} | Valid: {pct_valid:.1f}%", fontsize=16)

            plt.legend(loc='upper right', fontsize=11)

            # Save figure
            fig_path = vis_dir / f"power_density_iter_{item['iteration']:02d}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    # === Final Refined Selection Visualization ===
    if 'final_support' in solver_info and len(solver_info['final_support']) > 0:
        final_indices = solver_info['final_support']

        # Determine which candidates are kept vs filtered
        # Use rmse_filtered_support directly to preserve RMSE-sorted order (lowest to highest)
        if rmse_filtered_support is not None:
            kept_set = set(rmse_filtered_support)
            kept_indices = list(rmse_filtered_support)  # Already sorted by RMSE (lowest first)
            filtered_indices = [idx for idx in final_indices if idx not in kept_set]
        else:
            kept_indices = list(final_indices)
            filtered_indices = []


        fig = plt.figure(figsize=(13, 8))
        ax = fig.gca()

        # Plot sensors
        ax.scatter(sensor_locations[:, 0], sensor_locations[:, 1],
                   c=observed_powers_dB, s=150, edgecolor='green',
                   linewidth=2, cmap='hot', label='Monitoring Locations', zorder=6)

        # Plot true TX
        if len(tx_coords) > 0:
            ax.scatter(tx_coords[:, 0], tx_coords[:, 1],
                       marker='x', s=200, c='blue', linewidth=3,
                       label='True Transmitter Locations', zorder=10)

        # Plot filtered candidates (black stars, no numbers)
        if len(filtered_indices) > 0:
            filt_rows, filt_cols = np.unravel_index(filtered_indices, map_data['shape'])
            ax.scatter(filt_cols, filt_rows, c='black', marker='*', s=400,
                       edgecolor='white', linewidth=1.5,
                       label=f'Filtered by RMSE ({len(filtered_indices)})', zorder=10)

        # Plot kept candidates (magenta stars with numbers)
        if len(kept_indices) > 0:
            kept_rows, kept_cols = np.unravel_index(kept_indices, map_data['shape'])
            ax.scatter(kept_cols, kept_rows, c='magenta', marker='*', s=500,
                       edgecolor='white', linewidth=1.5,
                       label=f'Kept ({len(kept_indices)})', zorder=11)

            # Add numbered annotations to kept candidates for association with power analysis plots
            for idx, (col, row) in enumerate(zip(kept_cols, kept_rows)):
                ax.annotate(f'{idx+1}', (col, row), textcoords='offset points',
                           xytext=(8, 8), fontsize=12, fontweight='bold',
                           color='black', ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor='magenta', alpha=0.9),
                           zorder=12)

        # Formatting
        UTM_lat = map_data['UTM_lat']
        UTM_long = map_data['UTM_long']
        interval = max(1, len(UTM_lat) // 5)
        tick_values = list(range(0, len(UTM_lat), interval))
        tick_labels = ['{:.1f}'.format(lat) for lat in UTM_lat[::interval]]
        plt.xticks(ticks=tick_values, labels=tick_labels, fontsize=14, rotation=0)

        interval = max(1, len(UTM_long) // 5)
        tick_values = list(range(0, len(UTM_long), interval))
        tick_labels = ['{:.1f}'.format(lat) for lat in UTM_long[::interval]]
        plt.yticks(ticks=[0] + tick_values[1:], labels=[""] + tick_labels[1:], fontsize=14, rotation=90)

        ax.set_xlim([0, map_data['shape'][1]])
        ax.set_ylim([0, map_data['shape'][0]])

        # Title shows total, kept, and filtered counts
        title = f"Final Selection: {len(kept_indices)} Kept"
        if len(filtered_indices) > 0:
            title += f", {len(filtered_indices)} Filtered (RMSE > 20 dB)"
        plt.title(title, fontsize=18)
        plt.legend(loc='upper right')

        # Save figure
        fig_path = vis_dir / "final_selection.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def run_single_experiment(
    data_info: Dict,
    config: Dict,
    map_data: Dict,
    all_tx_locations: Dict,
    sigma_noise: float,
    selection_method: str,
    use_power_filtering: bool,
    whitening_method: str,
    feature_rho: Optional[List[float]],
    whitening_config_name: str,
    power_density_threshold: float = 0.3,
    strategy_name: str = '',
    model_type: str = 'tirem',
    recon_model_type: str = 'tirem',
    eta: float = 0.01,
    output_dir: Optional[Path] = None,
    save_visualization: bool = False,
    verbose: bool = False,
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
) -> Optional[Dict]:
    """
    Run a single reconstruction experiment.

    Returns
    -------
    dict or None
        Result dictionary with metrics, or None if failed
    """
    try:
        transmitters = data_info['transmitters']
        tx_underscore = "_".join(transmitters)
        data_dir = data_info['path']
        seed = data_info['seed']
        num_locations = data_info.get('num_locations')

        # Build config path (matching the directory naming convention)
        # Format: {transmitters}_nloc{N}_seed_{S} or legacy format without nloc
        config_id = tx_underscore
        if num_locations is not None:
            config_id = f"{config_id}_nloc{num_locations}"
        if seed is not None:
            config_id = f"{config_id}_seed_{seed}"
        config_path = f'config/monitoring_locations_{config_id}.yaml'

        # Check if config exists
        if not Path(config_path).exists():
            if verbose:
                print(f"    Config not found: {config_path}")
            return None

        # Load monitoring locations
        locations_config = load_monitoring_locations(
            config_path=config_path,
            map_data=map_data
        )
        sensor_locations = get_sensor_locations_array(locations_config)

        # Load power measurements
        powers_file = data_dir / f"{tx_underscore}_avg_powers.npy"
        if not powers_file.exists():
            if verbose:
                print(f"    Powers file not found: {powers_file}")
            return None

        observed_powers_dB = np.load(powers_file)
        observed_powers_linear = dbm_to_linear(observed_powers_dB)

        # Load observed standard deviations (for hetero_diag_obs whitening)
        stds_file = data_dir / f"{tx_underscore}_std_powers.npy"
        observed_stds_dB = np.load(stds_file) if stds_file.exists() else None

        # Get true transmitter locations
        tx_locations = {name: all_tx_locations[name] for name in transmitters if name in all_tx_locations}
        true_locs_pixels = np.array([tx['coordinates'] for tx in tx_locations.values()])

        start_time = time.time()

        # Determine model config paths (localization and reconstruction)
        if model_type == 'tirem':
            model_config_path = 'config/tirem_parameters.yaml'
        elif model_type == 'raytracing':
            model_config_path = 'config/sionna_parameters.yaml'
        else:
            model_config_path = None

        if recon_model_type == 'tirem':
            recon_model_config_path = 'config/tirem_parameters.yaml'
        elif recon_model_type == 'raytracing':
            recon_model_config_path = 'config/sionna_parameters.yaml'
        else:
            recon_model_config_path = None

        # Run reconstruction
        reconstruction_kwargs = {
            'sensor_locations': sensor_locations,
            'observed_powers_dBm': observed_powers_dB,
            'input_is_linear': False,
            'solve_in_linear_domain': True,
            'map_shape': map_data['shape'],
            'scale': config['spatial']['proxel_size'],
            'np_exponent': config['localization']['path_loss_exponent'],
            'lambda_reg': pooling_lambda,
            'norm_exponent': 0,
            'whitening_method': whitening_method,
            'sigma_noise': sigma_noise,
            'eta': eta,
            'solver': 'glrt',
            'selection_method': selection_method,
            'use_power_filtering': use_power_filtering,
            'power_density_threshold': power_density_threshold,
            'cluster_max_candidates': 30,
            'glrt_max_iter': max(10, len(transmitters) + 5),
            'glrt_threshold': 4.0,
            'glrt_max_iter': max(10, len(transmitters) + 5),
            'glrt_threshold': 4.0,
            'dedupe_distance_m': dedupe_distance_m,
            'return_linear_scale': False,
            'verbose': False,
            'model_type': model_type,
            'model_config_path': model_config_path,
            'n_jobs': -1,
            'beam_width': beam_width,
            'max_pool_size': max_pool_size,
            'max_pool_size': max_pool_size,
            'pool_refinement': True, # Always enable refinement for sweep
            'use_edf_penalty': use_edf_penalty,
            'edf_threshold': edf_threshold,
            'use_robust_scoring': use_robust_scoring,
            'robust_threshold': robust_threshold,
            # Observed std for hetero_diag_obs whitening (None for other methods)
            'observed_stds_dB': observed_stds_dB,
        }

        # Add feature_rho only for hetero_geo_aware
        if feature_rho is not None:
            reconstruction_kwargs['feature_rho'] = feature_rho

        tx_map, info = joint_sparse_reconstruction(**reconstruction_kwargs)

        elapsed = time.time() - start_time

        # Build experiment name for any file outputs
        pf_suffix = ''
        if use_power_filtering:
            pf_suffix = f'_pf_thresh{power_density_threshold}'
        experiment_name = f"{data_info['name']}_{strategy_name}_{whitening_config_name}_{selection_method}{pf_suffix}"

        # Always do RMSE-based candidate filtering and combinatorial selection (needed for BIC metrics)
        filtered_support = None
        if 'solver_info' in info and 'final_support' in info['solver_info']:
            final_support = info['solver_info']['final_support']

            if len(final_support) > 0:
                scale = config['spatial']['proxel_size']
                np_exponent = config['localization']['path_loss_exponent']

                # Step 1: Compute RMSE for all candidates (with bias correction)
                rmse_values, mae_values, max_error_values, optimal_tx_powers, slope_values = compute_candidate_power_rmse(
                    final_support=final_support,
                    tx_map=tx_map,
                    map_shape=map_data['shape'],
                    sensor_locations=sensor_locations,
                    observed_powers_dB=observed_powers_dB,
                    scale=scale,
                    np_exponent=np_exponent,
                )

                # Step 2: Filter candidates by RMSE threshold
                filtered_support, filtered_rmse, cutoff_rmse = filter_candidates_by_rmse(
                    final_support=final_support,
                    rmse_values=rmse_values,
                    max_error_values=max_error_values,
                    slope_values=slope_values,
                    output_dir=output_dir,
                    experiment_name=experiment_name,
                    min_candidates=1,
                    rmse_threshold=20.0,
                    max_error_threshold=38.0,
                    save_plot=save_visualization,  # Only save plot if visualizing
                )

                # Store filtered support for metrics computation
                info['solver_info']['rmse_filtered_support'] = filtered_support
                info['solver_info']['rmse_cutoff'] = cutoff_rmse
                info['solver_info']['n_filtered_by_rmse'] = len(final_support) - len(filtered_support)

                # Step 3: Generate power analysis plots only if visualizations requested
                if save_visualization and output_dir is not None:
                    save_candidate_power_analysis(
                        info=info,
                        tx_map=tx_map,
                        map_data=map_data,
                        sensor_locations=sensor_locations,
                        observed_powers_dB=observed_powers_dB,
                        tx_locations=tx_locations,
                        output_dir=output_dir,
                        experiment_name=experiment_name,
                        scale=scale,
                        np_exponent=np_exponent,
                        candidate_indices=filtered_support,
                    )

                # Step 4: Run combinatorial TX selection optimization (always, for BIC)
                # Find optimal combination of TXs that best explains observations
                # Only generate plots if save_visualization is True
                combination_result = run_combinatorial_selection(
                    info=info,
                    tx_map=tx_map,
                    map_data=map_data,
                    sensor_locations=sensor_locations,
                    observed_powers_dB=observed_powers_dB,
                    tx_locations=tx_locations,
                    output_dir=output_dir,
                    experiment_name=experiment_name,
                    filtered_support=filtered_support,
                    scale=scale,
                    np_exponent=np_exponent,
                    min_distance_m=combo_min_distance_m,
                    max_combination_size=combo_max_size,
                    max_candidates_to_consider=combo_max_candidates,
                    bic_penalty_weight=combo_bic_weight,
                    max_power_diff_dB=combo_max_power_diff_dB,
                    sensor_proximity_threshold_m=combo_sensor_proximity_threshold_m,
                    sensor_proximity_penalty=combo_sensor_proximity_penalty,
                    max_plots=10,
                    save_plots=save_visualization,  # Only generate plots if visualizing
                    verbose=False,
                )

                # Store combination result in info
                info['solver_info']['combination_result'] = combination_result
                info['solver_info']['optimal_combination'] = combination_result.get('best_combination', [])
                info['solver_info']['optimal_powers_dBm'] = combination_result.get('best_powers_dBm', np.array([]))
                info['solver_info']['combination_rmse'] = combination_result.get('best_rmse', np.inf)
                info['solver_info']['combination_bic'] = combination_result.get('best_bic', np.inf)

                # Step 5: Recompute optimal TX powers using the reconstruction
                # propagation model instead of the log-distance approximation
                # used during candidate selection.  The TX locations are kept
                # fixed; only powers are re-optimized so that reconstruction
                # uses power estimates consistent with the reconstruction
                # propagation model.
                optimal_combo = info['solver_info']['optimal_combination']
                if len(optimal_combo) > 0:
                    # Get the propagation matrix for reconstruction
                    if recon_model_type == model_type:
                        # Same model -- reuse A_model from localization
                        A_recon = info.get('A_model')
                    else:
                        # Different model -- compute propagation matrix
                        # for reconstruction model with sensor locations
                        from src.sparse_reconstruction.propagation_matrix import compute_propagation_matrix as _compute_prop_matrix
                        A_recon = _compute_prop_matrix(
                            sensor_locations=sensor_locations,
                            map_shape=map_data['shape'],
                            scale=scale,
                            model_type=recon_model_type,
                            config_path=recon_model_config_path,
                            np_exponent=np_exponent,
                            n_jobs=-1,
                            verbose=False,
                        )

                    if A_recon is not None:
                        recomp_powers, recomp_rmse, recomp_mae, recomp_max_err, recomp_total = \
                            recompute_powers_with_propagation_model(
                                combo_grid_indices=optimal_combo,
                                A_model=A_recon,
                                observed_powers_dB=observed_powers_dB,
                                max_power_diff_dB=combo_max_power_diff_dB,
                            )
                        info['solver_info']['optimal_powers_dBm'] = recomp_powers
                        info['solver_info']['combination_rmse'] = recomp_rmse

                # Step 5.5: Per-TX exponent refit (log_distance reconstruction only)
                # After localization, fit a per-TX path loss exponent from
                # observed sensor data, rebuild path gains, and re-optimize
                # powers.  This improves reconstruction when different TXs
                # experience different propagation conditions.
                if recon_model_type == 'log_distance' and len(optimal_combo) > 0:
                    per_tx_exp, _refit_gains, refit_powers, refit_rmse, \
                        refit_mae, refit_max_err, refit_total = \
                        refit_with_per_tx_exponents(
                            combo_grid_indices=optimal_combo,
                            map_shape=map_data['shape'],
                            sensor_locations=sensor_locations,
                            observed_powers_dB=observed_powers_dB,
                            current_powers_dBm=info['solver_info']['optimal_powers_dBm'],
                            scale=scale,
                            np_exponent_global=np_exponent,
                            max_power_diff_dB=combo_max_power_diff_dB,
                        )
                    info['solver_info']['per_tx_exponents'] = per_tx_exp.tolist()
                    info['solver_info']['optimal_powers_dBm'] = refit_powers
                    info['solver_info']['combination_rmse'] = refit_rmse

        # Save GLRT visualization if requested
        if save_visualization and output_dir is not None:
            save_glrt_visualization(
                info=info,
                map_data=map_data,
                sensor_locations=sensor_locations,
                observed_powers_dB=observed_powers_dB,
                tx_locations=tx_locations,
                output_dir=output_dir,
                experiment_name=experiment_name,
                rmse_filtered_support=filtered_support,
                save_iterations=save_iterations,
            )



        # Extract estimated locations
        if 'solver_info' in info and 'support' in info['solver_info']:
            support_indices = info['solver_info']['support']
            height, width = map_data['shape']

            n_est_raw = len(support_indices)

            valid_indices = []
            for idx in support_indices:
                r, c = idx // width, idx % width
                power_dbm = tx_map[r, c]
                if power_dbm > -190:
                    valid_indices.append(idx)

            n_est_diff = n_est_raw - len(valid_indices)

            est_rows = [idx // width for idx in valid_indices]
            est_cols = [idx % width for idx in valid_indices]
            est_locs_pixels = np.column_stack((est_cols, est_rows)) if valid_indices else np.empty((0, 2))
        else:
            from src.evaluation.metrics import extract_locations_from_map
            est_locs_pixels = extract_locations_from_map(tx_map, threshold=1e-10)
            n_est_raw = len(est_locs_pixels)
            n_est_diff = 0

        # Compute metrics
        metrics = compute_localization_metrics(
            true_locations=true_locs_pixels,
            estimated_locations=est_locs_pixels,
            scale=config['spatial']['proxel_size'],
            tolerance=200.0
        )

        # Compute metrics for optimal combination (if available)
        combo_metrics = {'combo_ale': np.nan, 'combo_tp': 0, 'combo_fp': 0, 'combo_fn': 0, 'combo_pd': 0.0, 'combo_precision': 0.0}
        if 'solver_info' in info and 'optimal_combination' in info['solver_info']:
            optimal_combo = info['solver_info']['optimal_combination']
            if len(optimal_combo) > 0:
                height, width = map_data['shape']
                combo_rows = [idx // width for idx in optimal_combo]
                combo_cols = [idx % width for idx in optimal_combo]
                combo_locs_pixels = np.column_stack((combo_cols, combo_rows))

                combo_metrics_raw = compute_localization_metrics(
                    true_locations=true_locs_pixels,
                    estimated_locations=combo_locs_pixels,
                    scale=config['spatial']['proxel_size'],
                    tolerance=200.0
                )
                combo_metrics = {
                    'combo_ale': combo_metrics_raw['ale'],
                    'combo_tp': combo_metrics_raw['tp'],
                    'combo_fp': combo_metrics_raw['fp'],
                    'combo_fn': combo_metrics_raw['fn'],
                    'combo_pd': combo_metrics_raw['pd'],
                    'combo_precision': combo_metrics_raw['precision'],
                }

        # Extract GLRT score history from solver info
        glrt_score_history = []
        glrt_n_iterations = 0
        glrt_initial_score = 0.0
        glrt_final_score = 0.0
        glrt_score_reduction = 0.0

        if 'solver_info' in info and 'candidates_history' in info['solver_info']:
            candidates_history = info['solver_info']['candidates_history']
            whitening_method = info['solver_info'].get('whitening_method', 'unknown')

            if len(candidates_history) > 0:
                glrt_n_iterations = len(candidates_history)

                # Extract scores - use normalized_score which is geo_aware_score for hetero_geo_aware
                # and R^2 normalized score for other methods
                for item in candidates_history:
                    # Use corrected score for hetero_geo_aware, raw score for others
                    if whitening_method == 'hetero_geo_aware':
                        score = item.get('normalized_score', item.get('selected_score', 0.0))
                    else:
                        score = item.get('selected_score', 0.0)
                    glrt_score_history.append(float(score))

                if glrt_score_history:
                    glrt_initial_score = glrt_score_history[0]
                    glrt_final_score = glrt_score_history[-1]
                    glrt_score_reduction = glrt_initial_score - glrt_final_score

            # Find closest iteration for each true transmitter
            height, width = map_data['shape']
            best_match_iterations = []

            if len(candidates_history) > 0 and len(true_locs_pixels) > 0:
                # Get locations of all added candidates in order
                candidate_locs = []
                for item in candidates_history:
                    idx = item['selected_index']
                    r, c = idx // width, idx % width
                    candidate_locs.append([c, r]) # x, y for distance calc

                candidate_locs = np.array(candidate_locs)

                # For each true transmitter, find index of closest candidate
                for true_tx in true_locs_pixels:
                    dists = np.sqrt(np.sum((candidate_locs - true_tx)**2, axis=1))
                    best_iter = np.argmin(dists) + 1 # 1-based iteration index
                    best_match_iterations.append(int(best_iter))

        # Convert history to JSON string for CSV storage
        import json
        glrt_score_history_str = json.dumps([round(s, 6) for s in glrt_score_history])

        # === RECONSTRUCTION ERROR COMPUTATION ===
        # Compute how well the estimated TX locations/powers predict RSS at validation points
        # Pass observation data for noise floor computation (clamping predictions)
        from .constants import _PROJECT_ROOT
        recon_metrics = compute_reconstruction_error(
            combo_indices=info.get('solver_info', {}).get('optimal_combination', []),
            combo_powers_dBm=[float(p) for p in info.get('solver_info', {}).get('optimal_powers_dBm', [])],
            map_data=map_data,
            transmitters=transmitters,
            project_root=_PROJECT_ROOT,
            observed_powers_dB=observed_powers_dB,
            num_locations=num_locations,
            model_type=recon_model_type,
            model_config_path=recon_model_config_path,
            scale=config['spatial']['proxel_size'],
            auto_generate=False,  # Don't auto-generate during sweep
            verbose=False,
            output_dir=output_dir,
            experiment_name=experiment_name,
            save_plot=save_visualization,  # Generate validation plots when visualizations enabled
            true_tx_locations=tx_locations,  # For spatial plot visualization
            per_tx_exponents=info.get('solver_info', {}).get('per_tx_exponents'),
        )

        return {
            'ale': metrics['ale'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'pd': metrics['pd'],
            'precision': metrics['precision'],
            'f1_score': metrics['f1_score'],
            'n_estimated': metrics['n_est'],
            'n_est_raw': n_est_raw,
            'n_est_diff': n_est_diff,
            'count_error': metrics['n_est'] - len(true_locs_pixels),
            'runtime_s': elapsed,
            'obs_min_dbm': np.min(observed_powers_dB),
            'obs_mean_dbm': np.mean(observed_powers_dB),
            'obs_max_dbm': np.max(observed_powers_dB),
            'glrt_n_iterations': glrt_n_iterations,
            'glrt_initial_score': glrt_initial_score,
            'glrt_final_score': glrt_final_score,
            'glrt_score_reduction': glrt_score_reduction,
            'glrt_score_history': glrt_score_history_str,
            'best_match_iterations': json.dumps(best_match_iterations), # Store as JSON list
            # Combinatorial selection metrics
            'combo_n_tx': len(info.get('solver_info', {}).get('optimal_combination', [])),
            'combo_rmse': info.get('solver_info', {}).get('combination_rmse', np.nan),
            'combo_bic': info.get('solver_info', {}).get('combination_bic', np.nan),
            'combo_indices': json.dumps(info.get('solver_info', {}).get('optimal_combination', [])),
            'combo_powers_dBm': json.dumps(
                [float(p) for p in info.get('solver_info', {}).get('optimal_powers_dBm', [])]
            ),
            'per_tx_exponents': json.dumps(info.get('solver_info', {}).get('per_tx_exponents', [])),
            # Combinatorial selection localization metrics
            'combo_ale': combo_metrics['combo_ale'],
            'combo_tp': combo_metrics['combo_tp'],
            'combo_fp': combo_metrics['combo_fp'],
            'combo_fn': combo_metrics['combo_fn'],
            'combo_pd': combo_metrics['combo_pd'],
            'combo_precision': combo_metrics['combo_precision'],
            'combo_count_error': abs(len(info.get('solver_info', {}).get('optimal_combination', [])) - len(true_locs_pixels)),
            # Reconstruction error metrics
            'recon_rmse': recon_metrics['recon_rmse'],
            'recon_mae': recon_metrics['recon_mae'],
            'recon_bias': recon_metrics['recon_bias'],
            'recon_max_error': recon_metrics['recon_max_error'],
            'recon_n_val_points': recon_metrics['recon_n_val_points'],
            'recon_noise_floor': recon_metrics['recon_noise_floor'],
            'recon_status': recon_metrics['recon_status'],
        }

    except Exception as e:
        if verbose:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
        return None
