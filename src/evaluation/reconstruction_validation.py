"""
Reconstruction error computation for comprehensive parameter sweep.

This module provides utilities to compute reconstruction error metrics by comparing
predicted RSS (from estimated TX locations/powers) against observed RSS at independent
validation locations.
"""

import numpy as np
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.evaluation.validation import ReconstructionValidator
from src.sparse_reconstruction import dbm_to_linear


def normalize_tx_id(transmitters: List[str]) -> str:
    """Normalize transmitter list to canonical alphabetical order."""
    return "_".join(sorted(transmitters))


def get_validation_paths(transmitters: List[str], project_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Get validation config and data paths for a transmitter combination."""
    tx_id = normalize_tx_id(transmitters)
    config_path = project_root / "config" / f"validation_{tx_id}.yaml"
    data_dir = project_root / "data" / "processed" / f"validation_{tx_id}"

    if config_path.exists() and data_dir.exists():
        return config_path, data_dir
    return None, None


def check_validation_data_exists(transmitters: List[str], project_root: Path) -> bool:
    """Check if validation data exists for given transmitters."""
    val_config, val_data_dir = get_validation_paths(transmitters, project_root)
    return val_config is not None and val_data_dir is not None


def generate_validation_data(transmitters: List[str], project_root: Path, verbose: bool = False) -> bool:
    """
    Attempt to generate validation data by calling process_raw_data_to_validation.py.
    Returns True if successful, False otherwise.
    """
    # Construct command - uses the process_raw_data_to_validation.py script
    tx_arg = ",".join(transmitters)
    script_path = project_root / "scripts" / "process_raw_data_to_validation.py"

    if not script_path.exists():
        if verbose:
            print(f"  Validation generator script not found: {script_path}")
        return False

    # Determine input directories based on transmitter names
    # Default raw data location pattern
    raw_data_root = Path("C:/Users/serha/raw_data")
    input_dirs = []
    for tx in transmitters:
        # Try common raw data directory patterns
        stat_dir = raw_data_root / "stat_rot" / "stat"
        rot_dir = raw_data_root / "stat_rot" / "rot"
        if stat_dir.exists():
            input_dirs.append(str(stat_dir))
        if rot_dir.exists():
            input_dirs.append(str(rot_dir))

    if not input_dirs:
        if verbose:
            print(f"  No raw data directories found for {transmitters}")
        return False

    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--input-dir", *input_dirs,
        "--transmitter", tx_arg,
    ]

    try:
        if verbose:
            print(f"  Generating validation data for {transmitters}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0
    except Exception as e:
        if verbose:
            print(f"  Failed to generate validation data: {e}")
        return False


def compute_noise_floor(observed_powers_dB: np.ndarray, num_locations: int) -> float:
    """
    Compute the noise floor based on observation data.

    The noise floor accounts for the receiver's sensitivity limit. With fewer
    observation locations, the minimum observed power may not capture the true
    noise floor, so we apply a correction.

    Parameters
    ----------
    observed_powers_dB : ndarray
        Observed powers in dBm from localization observations
    num_locations : int
        Number of observation locations (nloc)

    Returns
    -------
    float
        Estimated noise floor in dBm
    """
    min_power_dB = np.min(observed_powers_dB)

    # Linear interpolation of adjustment based on nloc:
    # - nloc=10: adjustment = -10 dB (extend floor lower due to sparse sampling)
    # - nloc=30: adjustment = 0 dB (min power is good estimate of floor)
    # For nloc outside [10, 30], clamp the adjustment
    if num_locations <= 10:
        adjustment = -10.0
    elif num_locations >= 30:
        adjustment = 0.0
    else:
        # Linear interpolation between nloc=10 and nloc=30
        adjustment = -10.0 + (num_locations - 10) * (10.0 / 20.0)

    noise_floor = min_power_dB + adjustment
    return noise_floor


def save_validation_power_analysis_plot(
    combo_indices: List[int],
    combo_powers_dBm: List[float],
    val_points: np.ndarray,
    observed_val_powers_dBm: np.ndarray,
    predicted_val_powers_dBm: np.ndarray,
    map_shape: Tuple[int, int],
    scale: float,
    noise_floor: Optional[float],
    metrics: Dict,
    output_dir: Path,
    experiment_name: str,
) -> None:
    """
    Generate validation power analysis plot.

    Creates a plot showing observed vs predicted validation power as a function
    of average distance to all estimated transmitters.

    Parameters
    ----------
    combo_indices : list of int
        Grid indices of estimated TX locations
    combo_powers_dBm : list of float
        Estimated TX powers in dBm
    val_points : ndarray
        Validation point locations in pixel coordinates, shape (N, 2) as [col, row]
    observed_val_powers_dBm : ndarray
        Observed validation powers in dBm
    predicted_val_powers_dBm : ndarray
        Predicted validation powers in dBm (after clamping)
    map_shape : tuple
        (height, width) of the map
    scale : float
        Pixel-to-meter scaling factor
    noise_floor : float or None
        Noise floor used for clamping (dBm)
    metrics : dict
        Computed metrics (rmse, mae, bias, etc.)
    output_dir : Path
        Directory to save figure
    experiment_name : str
        Name for this experiment
    """
    if len(combo_indices) == 0 or len(val_points) == 0:
        return

    vis_dir = output_dir / 'glrt_visualizations' / experiment_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    height, width = map_shape
    n_val = len(val_points)
    n_tx = len(combo_indices)

    # Compute TX locations in pixel coordinates
    tx_coords = []
    for idx in combo_indices:
        tx_row = idx // width
        tx_col = idx % width
        tx_coords.append([tx_col, tx_row])
    tx_coords = np.array(tx_coords)

    # Compute average distance from each validation point to all TXs
    avg_distances = np.zeros(n_val)
    for j, val_loc in enumerate(val_points):
        # val_loc is [col, row]
        distances_to_txs = []
        for tx_coord in tx_coords:
            dist_pixels = np.sqrt((val_loc[0] - tx_coord[0])**2 + (val_loc[1] - tx_coord[1])**2)
            dist_m = max(dist_pixels * scale, 1.0)
            distances_to_txs.append(dist_m)
        avg_distances[j] = np.mean(distances_to_txs)

    # Compute error for coloring
    power_errors = predicted_val_powers_dBm - observed_val_powers_dBm

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use log scale for X axis
    ax.set_xscale('log')

    # For many points, use smaller markers and transparency
    marker_size = max(5, min(30, 500 / np.sqrt(n_val)))
    alpha = max(0.3, min(0.8, 100 / n_val))

    # Sort by distance for cleaner visualization
    sort_idx = np.argsort(avg_distances)

    # Plot observed validation powers
    ax.scatter(avg_distances[sort_idx], observed_val_powers_dBm[sort_idx],
               s=marker_size, c='blue', marker='o', alpha=alpha,
               edgecolor='none', label=f'Observed Validation ({n_val} pts)')

    # Plot predicted validation powers
    ax.scatter(avg_distances[sort_idx], predicted_val_powers_dBm[sort_idx],
               s=marker_size, c='red', marker='^', alpha=alpha,
               edgecolor='none', label='Predicted (clamped)')

    # Fit and plot trend lines using binned averages for cleaner visualization
    n_bins = min(20, n_val // 10)
    if n_bins >= 3:
        # Create log-spaced bins
        log_dist = np.log10(avg_distances)
        bin_edges = np.linspace(log_dist.min(), log_dist.max(), n_bins + 1)
        bin_centers = []
        obs_bin_means = []
        pred_bin_means = []

        for i in range(n_bins):
            mask = (log_dist >= bin_edges[i]) & (log_dist < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_centers.append(10 ** ((bin_edges[i] + bin_edges[i + 1]) / 2))
                obs_bin_means.append(np.mean(observed_val_powers_dBm[mask]))
                pred_bin_means.append(np.mean(predicted_val_powers_dBm[mask]))

        if len(bin_centers) >= 2:
            bin_centers = np.array(bin_centers)
            obs_bin_means = np.array(obs_bin_means)
            pred_bin_means = np.array(pred_bin_means)

            # Plot binned trend lines
            ax.plot(bin_centers, obs_bin_means, 'b-', linewidth=2.5, alpha=0.9,
                    label='Observed Trend (binned avg)', zorder=5)
            ax.plot(bin_centers, pred_bin_means, 'r-', linewidth=2.5, alpha=0.9,
                    label='Predicted Trend (binned avg)', zorder=5)

    # Draw noise floor line if available
    if noise_floor is not None:
        ax.axhline(y=noise_floor, color='gray', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'Noise Floor ({noise_floor:.1f} dBm)')

    # Formatting
    ax.set_xlabel('Average Distance to All Estimated TXs (m) [Log Scale]', fontsize=13)
    ax.set_ylabel('Received Power (dBm)', fontsize=13)

    # Build title with TX info and metrics
    tx_info_parts = [f"TX{i+1}:{combo_powers_dBm[i]:.0f}dBm" for i in range(n_tx)]
    tx_info_str = ", ".join(tx_info_parts)

    title = f"Validation Power Analysis ({n_tx} TX)\n"
    title += f"TX Powers: {tx_info_str}\n"
    title += f"RMSE: {metrics['rmse']:.1f} dB | MAE: {metrics['mae']:.1f} dB | Bias: {metrics['bias']:.1f} dB"
    ax.set_title(title, fontsize=11)

    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # Set reasonable y-axis limits
    y_min = min(observed_val_powers_dBm.min(), predicted_val_powers_dBm.min()) - 5
    y_max = max(observed_val_powers_dBm.max(), predicted_val_powers_dBm.max()) + 5
    ax.set_ylim(y_min, y_max)

    # Save figure
    fig_path = vis_dir / "validation_power_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_reconstruction_error(
    combo_indices: List[int],
    combo_powers_dBm: List[float],
    map_data: Dict,
    transmitters: List[str],
    project_root: Path,
    observed_powers_dB: Optional[np.ndarray] = None,
    num_locations: Optional[int] = None,
    model_type: str = 'tirem',
    model_config_path: Optional[str] = None,
    scale: float = 1.0,
    cache_dir: Optional[Path] = None,
    auto_generate: bool = False,
    verbose: bool = False,
    output_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    save_plot: bool = False,
) -> Dict:
    """
    Compute reconstruction error metrics for a given TX estimate.

    Parameters
    ----------
    combo_indices : list of int
        Grid indices of estimated transmitter locations
    combo_powers_dBm : list of float
        Estimated transmit powers in dBm
    map_data : dict
        Map data dictionary with 'shape' and georeferencing info
    transmitters : list of str
        List of transmitter names
    project_root : Path
        Project root directory
    observed_powers_dB : ndarray, optional
        Observed powers in dBm from localization (used to compute noise floor)
    num_locations : int, optional
        Number of observation locations (nloc). If None, inferred from observed_powers_dB length
    model_type : str
        Propagation model ('tirem', 'log_distance')
    model_config_path : str, optional
        Path to model config
    scale : float
        Pixel-to-meter scale
    cache_dir : Path, optional
        Cache directory for propagation matrices
    auto_generate : bool
        If True, attempt to generate missing validation data
    verbose : bool
        Print debug info
    output_dir : Path, optional
        Directory to save visualization (required if save_plot=True)
    experiment_name : str, optional
        Name for the experiment (required if save_plot=True)
    save_plot : bool
        If True, save validation power analysis plot

    Returns
    -------
    dict with keys:
        - 'recon_rmse': RMSE in dB
        - 'recon_mae': MAE in dB
        - 'recon_bias': Mean bias in dB
        - 'recon_max_error': Maximum absolute error in dB
        - 'recon_n_val_points': Number of validation points
        - 'recon_noise_floor': Noise floor used for clamping (dBm)
        - 'recon_status': 'success', 'no_validation_data', 'no_estimate', or 'error:...'
    """
    error_metrics = {
        'recon_rmse': np.nan,
        'recon_mae': np.nan,
        'recon_bias': np.nan,
        'recon_max_error': np.nan,
        'recon_n_val_points': 0,
        'recon_noise_floor': np.nan,
        'recon_status': 'unknown',
    }

    # Check for empty combo
    if not combo_indices or len(combo_indices) == 0:
        error_metrics['recon_status'] = 'no_estimate'
        return error_metrics

    # Get validation paths
    val_config, val_data_dir = get_validation_paths(transmitters, project_root)

    if val_config is None or val_data_dir is None:
        # Optionally try to generate validation data
        if auto_generate:
            success = generate_validation_data(transmitters, project_root, verbose)
            if success:
                val_config, val_data_dir = get_validation_paths(transmitters, project_root)

        if val_config is None:
            error_metrics['recon_status'] = 'no_validation_data'
            return error_metrics

    try:
        # Initialize validator (suppress verbose output during sweep)
        import io
        import sys as sys_module
        old_stdout = sys_module.stdout
        sys_module.stdout = io.StringIO() if not verbose else old_stdout

        try:
            validator = ReconstructionValidator(
                map_data=map_data,
                validation_config_path=str(val_config),
                validation_data_dir=str(val_data_dir)
            )

            # Load observed data
            tx_id = normalize_tx_id(transmitters)
            file_prefix = f"validation_{tx_id}"
            validator.load_observed_data(file_prefix)

            # Filter out-of-bounds points
            validator.filter_out_of_bounds(verbose=verbose)
        finally:
            sys_module.stdout = old_stdout

        # Get propagation matrix
        if cache_dir is None:
            cache_dir = project_root / "data" / "cache"

        if model_config_path is None and model_type == 'tirem':
            model_config_path = str(project_root / 'config' / 'tirem_parameters.yaml')

        validator.get_propagation_matrix(
            model_type=model_type,
            model_config_path=model_config_path,
            scale=scale,
            cache_dir=str(cache_dir),
            verbose=verbose
        )

        # Build TX power map from combo_indices and combo_powers_dBm
        height, width = map_data['shape']
        tx_map_linear = np.zeros((height, width), dtype=np.float64)

        for idx, power_dBm in zip(combo_indices, combo_powers_dBm):
            row = idx // width
            col = idx % width
            tx_map_linear[row, col] = dbm_to_linear(power_dBm)

        # Predict RSS at validation points
        predicted_dBm = validator.predict_rss(tx_map_linear)

        # Compute noise floor and clamp predictions
        # The propagation model can predict arbitrarily low values, but the
        # receiver has a noise floor that limits actual measurements
        noise_floor = None
        if observed_powers_dB is not None:
            nloc = num_locations if num_locations is not None else len(observed_powers_dB)
            noise_floor = compute_noise_floor(observed_powers_dB, nloc)

            # Clamp predictions to noise floor (predictions below floor are set to floor)
            predicted_dBm = np.maximum(predicted_dBm, noise_floor)

            if verbose:
                n_clamped = np.sum(predicted_dBm == noise_floor)
                print(f"  Noise floor: {noise_floor:.2f} dBm (nloc={nloc}, min_obs={np.min(observed_powers_dB):.2f})")
                print(f"  Predictions clamped: {n_clamped}/{len(predicted_dBm)}")

        # Compute metrics (after clamping)
        metrics = validator.compute_metrics(predicted_dBm, verbose=verbose)

        error_metrics['recon_rmse'] = metrics['rmse']
        error_metrics['recon_mae'] = metrics['mae']
        error_metrics['recon_bias'] = metrics['bias']
        error_metrics['recon_max_error'] = max(abs(metrics['min_error']), abs(metrics['max_error']))
        error_metrics['recon_n_val_points'] = metrics['n_samples']
        error_metrics['recon_noise_floor'] = noise_floor if noise_floor is not None else np.nan
        error_metrics['recon_status'] = 'success'

        # Generate validation power analysis plot if requested
        if save_plot and output_dir is not None and experiment_name is not None:
            try:
                save_validation_power_analysis_plot(
                    combo_indices=combo_indices,
                    combo_powers_dBm=combo_powers_dBm,
                    val_points=validator.val_points,
                    observed_val_powers_dBm=validator.observed_powers_dBm,
                    predicted_val_powers_dBm=predicted_dBm,
                    map_shape=map_data['shape'],
                    scale=scale,
                    noise_floor=noise_floor,
                    metrics=metrics,
                    output_dir=output_dir,
                    experiment_name=experiment_name,
                )
            except Exception as plot_err:
                if verbose:
                    print(f"  Failed to save validation plot: {plot_err}")

    except Exception as e:
        if verbose:
            print(f"  Reconstruction error computation failed: {e}")
        error_metrics['recon_status'] = f'error:{str(e)[:40]}'

    return error_metrics
