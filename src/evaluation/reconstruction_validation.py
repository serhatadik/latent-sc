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

    Creates a plot with subplots for each estimated transmitter, showing observed
    vs predicted validation power as a function of distance to that TX.

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

    # Compute distance from each validation point to each TX individually
    # distances_to_tx[i, j] = distance from validation point j to TX i
    distances_to_tx = np.zeros((n_tx, n_val))
    for i, tx_coord in enumerate(tx_coords):
        for j, val_loc in enumerate(val_points):
            dist_pixels = np.sqrt((val_loc[0] - tx_coord[0])**2 + (val_loc[1] - tx_coord[1])**2)
            distances_to_tx[i, j] = max(dist_pixels * scale, 1.0)

    # Determine subplot layout
    if n_tx == 1:
        n_rows, n_cols = 1, 1
        fig_width, fig_height = 10, 7
    elif n_tx == 2:
        n_rows, n_cols = 1, 2
        fig_width, fig_height = 16, 6
    elif n_tx <= 4:
        n_rows, n_cols = 2, 2
        fig_width, fig_height = 14, 10
    elif n_tx <= 6:
        n_rows, n_cols = 2, 3
        fig_width, fig_height = 18, 10
    else:
        n_cols = 3
        n_rows = (n_tx + n_cols - 1) // n_cols
        fig_width, fig_height = 18, 5 * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_tx == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Marker size and transparency - larger and more visible
    marker_size = max(8, min(25, 400 / np.sqrt(n_val)))
    scatter_alpha = max(0.4, min(0.7, 150 / n_val))

    # Shared y-axis limits
    y_min = min(observed_val_powers_dBm.min(), predicted_val_powers_dBm.min()) - 5
    y_max = max(observed_val_powers_dBm.max(), predicted_val_powers_dBm.max()) + 5

    n_bins = min(20, n_val // 10)

    for i in range(n_tx):
        ax = axes[i]
        distances = distances_to_tx[i, :]
        tx_power = combo_powers_dBm[i]

        # Use log scale for X axis
        ax.set_xscale('log')

        # Sort by distance for this TX
        sort_idx = np.argsort(distances)

        # Plot observed validation powers (blue circles)
        ax.scatter(distances[sort_idx], observed_val_powers_dBm[sort_idx],
                   s=marker_size, c='blue', marker='o', alpha=scatter_alpha,
                   edgecolor='none', label=f'Observed ({n_val} pts)')

        # Plot predicted validation powers (red triangles)
        ax.scatter(distances[sort_idx], predicted_val_powers_dBm[sort_idx],
                   s=marker_size, c='red', marker='^', alpha=scatter_alpha,
                   edgecolor='none', label='Predicted')

        # Compute and plot binned trend lines
        if n_bins >= 3:
            log_dist = np.log10(distances)
            bin_edges = np.linspace(log_dist.min(), log_dist.max(), n_bins + 1)
            bin_centers = []
            obs_bin_means = []
            pred_bin_means = []

            for b in range(n_bins):
                mask = (log_dist >= bin_edges[b]) & (log_dist < bin_edges[b + 1])
                if np.sum(mask) >= 3:  # Require at least 3 points per bin
                    bin_centers.append(10 ** ((bin_edges[b] + bin_edges[b + 1]) / 2))
                    obs_bin_means.append(np.mean(observed_val_powers_dBm[mask]))
                    pred_bin_means.append(np.mean(predicted_val_powers_dBm[mask]))

            if len(bin_centers) >= 2:
                bin_centers = np.array(bin_centers)
                obs_bin_means = np.array(obs_bin_means)
                pred_bin_means = np.array(pred_bin_means)

                # Plot observed trend line (solid blue)
                ax.plot(bin_centers, obs_bin_means, 'b-', linewidth=2.5,
                        alpha=0.9, label='Obs. Trend', zorder=5)

                # Plot predicted trend line (solid red)
                ax.plot(bin_centers, pred_bin_means, 'r-', linewidth=2.5,
                        alpha=0.9, label='Pred. Trend', zorder=5)

        # Draw noise floor line if available
        if noise_floor is not None:
            ax.axhline(y=noise_floor, color='gray', linestyle='--', linewidth=1.5,
                       alpha=0.7, label=f'Noise Floor ({noise_floor:.1f})')

        # Formatting for this subplot
        ax.set_xlabel('Distance to TX (m)', fontsize=11)
        ax.set_ylabel('Power (dBm)', fontsize=11)
        ax.set_title(f'TX{i+1}: {tx_power:.0f} dBm', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(y_min, y_max)

    # Hide unused subplots
    for i in range(n_tx, len(axes)):
        axes[i].set_visible(False)

    # Overall title
    suptitle = f"Validation Power Analysis ({n_val} validation points)\n"
    suptitle += f"RMSE: {metrics['rmse']:.1f} dB | MAE: {metrics['mae']:.1f} dB | Bias: {metrics['bias']:.1f} dB"
    fig.suptitle(suptitle, fontsize=13, y=1.02)

    plt.tight_layout()

    # Save figure
    fig_path = vis_dir / "validation_power_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_validation_spatial_plot(
    combo_indices: List[int],
    combo_powers_dBm: List[float],
    val_points: np.ndarray,
    observed_val_powers_dBm: np.ndarray,
    map_shape: Tuple[int, int],
    scale: float,
    true_tx_locations: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
) -> None:
    """
    Generate spatial plot showing validation measurements with power levels,
    true TX locations, and estimated TX locations.

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
    map_shape : tuple
        (height, width) of the map
    scale : float
        Pixel-to-meter scaling factor
    true_tx_locations : dict, optional
        Dictionary of true TX locations {name: {'coordinates': [col, row]}}
    metrics : dict, optional
        Computed metrics (rmse, mae, bias, etc.)
    output_dir : Path, optional
        Directory to save figure
    experiment_name : str, optional
        Name for this experiment
    """
    if len(val_points) == 0:
        return

    if output_dir is None or experiment_name is None:
        return

    vis_dir = output_dir / 'glrt_visualizations' / experiment_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    height, width = map_shape
    n_val = len(val_points)
    n_tx = len(combo_indices)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot validation points with power levels as color
    # val_points is [col, row] format
    val_cols = val_points[:, 0]
    val_rows = val_points[:, 1]

    # Scatter plot of validation points colored by observed power
    scatter = ax.scatter(val_cols, val_rows, c=observed_val_powers_dBm,
                         cmap='viridis', s=15, alpha=0.7, edgecolor='none',
                         label=f'Validation ({n_val} pts)')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Observed Power (dBm)', fontsize=11)

    # Plot estimated TX locations
    if n_tx > 0:
        est_tx_cols = []
        est_tx_rows = []
        for idx in combo_indices:
            tx_row = idx // width
            tx_col = idx % width
            est_tx_cols.append(tx_col)
            est_tx_rows.append(tx_row)

        ax.scatter(est_tx_cols, est_tx_rows, c='red', s=300, marker='*',
                   edgecolor='black', linewidth=1.5, zorder=10,
                   label=f'Estimated TX ({n_tx})')

        # Add power labels next to estimated TXs
        for i, (col, row) in enumerate(zip(est_tx_cols, est_tx_rows)):
            ax.annotate(f'{combo_powers_dBm[i]:.0f}dBm',
                        (col, row), xytext=(8, 8), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # Plot true TX locations
    if true_tx_locations is not None and len(true_tx_locations) > 0:
        true_cols = []
        true_rows = []
        true_names = []
        for name, loc in true_tx_locations.items():
            coords = loc['coordinates']
            true_cols.append(coords[0])
            true_rows.append(coords[1])
            true_names.append(name)

        ax.scatter(true_cols, true_rows, c='lime', s=400, marker='X',
                   edgecolor='black', linewidth=2, zorder=9,
                   label=f'True TX ({len(true_tx_locations)})')

        # Add name labels next to true TXs
        for name, col, row in zip(true_names, true_cols, true_rows):
            ax.annotate(name, (col, row), xytext=(-8, -15), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='darkgreen',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # Set axis limits to map bounds
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)  # Y axis not inverted to match other spatial plots

    # Formatting
    ax.set_xlabel(f'Column (pixel) [{scale:.1f} m/pixel]', fontsize=11)
    ax.set_ylabel(f'Row (pixel) [{scale:.1f} m/pixel]', fontsize=11)

    # Build title
    title = "Validation Measurements Spatial Distribution\n"
    if metrics is not None:
        title += f"RMSE: {metrics['rmse']:.1f} dB | MAE: {metrics['mae']:.1f} dB | Bias: {metrics['bias']:.1f} dB"
    ax.set_title(title, fontsize=12)

    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # Save figure
    fig_path = vis_dir / "validation_spatial_map.png"
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
    true_tx_locations: Optional[Dict] = None,
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
        If True, save validation power analysis and spatial plots
    true_tx_locations : dict, optional
        Dictionary of true TX locations {name: {'coordinates': [col, row]}}
        Used for spatial plot visualization

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

        # Generate validation plots if requested
        if save_plot and output_dir is not None and experiment_name is not None:
            # Power analysis plot (subplots per TX)
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
                    print(f"  Failed to save validation power plot: {plot_err}")

            # Spatial map plot
            try:
                save_validation_spatial_plot(
                    combo_indices=combo_indices,
                    combo_powers_dBm=combo_powers_dBm,
                    val_points=validator.val_points,
                    observed_val_powers_dBm=validator.observed_powers_dBm,
                    map_shape=map_data['shape'],
                    scale=scale,
                    true_tx_locations=true_tx_locations,
                    metrics=metrics,
                    output_dir=output_dir,
                    experiment_name=experiment_name,
                )
            except Exception as plot_err:
                if verbose:
                    print(f"  Failed to save validation spatial plot: {plot_err}")

    except Exception as e:
        if verbose:
            print(f"  Reconstruction error computation failed: {e}")
        error_metrics['recon_status'] = f'error:{str(e)[:40]}'

    return error_metrics
