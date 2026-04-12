#!/usr/bin/env python
"""
Run likelihood-based reconstruction on the same directories used in the ablation study.

Evaluates reconstruction performance (RMSE, MAE, bias, max_error) at validation
points using the likelihood marginalization approach, for direct comparison with
the GLRT sparse reconstruction results.

The likelihood method:
  1. Builds a propagation matrix (log_distance, tirem, or raytracing)
  2. Estimates transmit power at every grid cell (single-TX-per-hypothesis)
  3. Computes a PMF over transmitter locations using Gaussian likelihood
     with spatial covariance (exponential decay whitening)
  4. Marginalizes over all hypotheses to predict received power at validation points
"""

# Threading env vars BEFORE numpy (Windows deadlock prevention)
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import io
import sys
import time
import yaml
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.sweep.discovery import discover_data_directories
from scripts.sweep.constants import _PROJECT_ROOT

from src.utils import (
    load_slc_map,
    load_monitoring_locations,
    get_sensor_locations_array,
    load_transmitter_locations,
)
from src.sparse_reconstruction.propagation_matrix import compute_propagation_matrix
from src.sparse_reconstruction import dbm_to_linear, linear_to_dbm
from src.localization.likelihood import build_covariance_matrix
from src.localization.transmitter import estimate_transmit_power_map
from src.evaluation.reconstruction_validation import (
    compute_noise_floor,
    get_validation_paths,
    normalize_tx_id,
)
from src.evaluation.validation import ReconstructionValidator


# ---------------------------------------------------------------------------
# Vectorized PMF computation (replaces the slow double-loop in likelihood.py)
# ---------------------------------------------------------------------------

def compute_pmf_vectorized(
    transmit_power_map: np.ndarray,
    propagation_matrix: np.ndarray,
    observed_powers: np.ndarray,
    cov_matrix: np.ndarray,
    threshold: float = 1e-6,
) -> np.ndarray:
    """
    Compute transmitter location PMF using vectorized operations.

    Parameters
    ----------
    transmit_power_map : (H, W) estimated transmit power at each cell (dBm)
    propagation_matrix : (M, N) linear path gains from grid to sensors
    observed_powers    : (M,) observed sensor powers (dBm)
    cov_matrix         : (M, M) noise covariance matrix
    threshold          : minimum PMF value to retain

    Returns
    -------
    pmf : (N,) probability mass function over grid cells (flattened)
    """
    t_hat = transmit_power_map.ravel()  # (N,)

    # Path gain in dB: G[j, i] = 10*log10(A[j, i])
    with np.errstate(divide='ignore'):
        G = 10.0 * np.log10(np.maximum(propagation_matrix, 1e-20))  # (M, N)

    # Predicted power at each sensor for each hypothesis
    # predicted[j, i] = t_hat[i] + G[j, i]
    error = (t_hat[np.newaxis, :] + G) - observed_powers[:, np.newaxis]  # (M, N)

    # Inverse covariance
    try:
        V_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(cov_matrix)

    # Quadratic form: e_i^T V^{-1} e_i for each hypothesis i
    V_inv_error = V_inv @ error      # (M, N)
    quad_forms = np.sum(error * V_inv_error, axis=0)  # (N,)

    # Log-likelihood (up to shared constant)
    log_lik = -0.5 * quad_forms

    # Numerically stable softmax -> PMF
    log_lik -= np.max(log_lik)
    pmf = np.exp(log_lik)
    pmf /= np.sum(pmf)

    # Threshold and renormalize
    pmf[pmf < threshold] = 0.0
    total = np.sum(pmf)
    if total > 0:
        pmf /= total

    return pmf


# ---------------------------------------------------------------------------
# Marginalized prediction at validation points
# ---------------------------------------------------------------------------

def predict_at_validation_points(
    transmit_power_map: np.ndarray,
    pmf: np.ndarray,
    A_val: np.ndarray,
) -> np.ndarray:
    """
    Predict received power at validation points via likelihood marginalization.

    p_hat_linear(y) = sum_i  A_val[y, i] * 10^(t_hat_i / 10) * pmf(i)

    Parameters
    ----------
    transmit_power_map : (H, W) estimated transmit power (dBm)
    pmf                : (N,) PMF over grid cells (flattened, thresholded)
    A_val              : (M_val, N) linear path gains from grid to validation points

    Returns
    -------
    predicted_dBm : (M_val,) predicted power in dBm
    """
    t_hat = transmit_power_map.ravel()
    tx_linear = 10.0 ** (t_hat / 10.0)

    # Only compute for nonzero-PMF cells
    nz = pmf > 0
    weighted = tx_linear[nz] * pmf[nz]             # (K,)
    predicted_linear = A_val[:, nz] @ weighted      # (M_val,)

    with np.errstate(divide='ignore'):
        predicted_dBm = 10.0 * np.log10(np.maximum(predicted_linear, 1e-20))

    return predicted_dBm


def compute_pmf_sharpness_metrics(
    pmf: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Compute PMF sharpness metrics that are cheap to aggregate across experiments.

    Parameters
    ----------
    pmf : (N,) PMF over grid cells (flattened)
    threshold : float, optional
        Threshold used to define PMF support. Cells with pmf > threshold are
        counted as active support.

    Returns
    -------
    dict
        Entropy, normalized entropy, peak mass, top-k mass, effective support,
        and support size/fraction.
    """
    pmf = np.asarray(pmf, dtype=np.float64).ravel()
    n_cells = len(pmf)

    if n_cells == 0:
        return {
            'pmf_entropy': np.nan,
            'pmf_entropy_norm': np.nan,
            'pmf_peak_mass': np.nan,
            'pmf_top5_mass': np.nan,
            'pmf_effective_support': np.nan,
            'pmf_support_size': 0,
            'pmf_support_fraction': np.nan,
        }

    support_mask = pmf > threshold
    support_size = int(np.sum(support_mask))

    positive = pmf[pmf > 0]
    if len(positive) == 0:
        entropy = 0.0
    else:
        entropy = float(-np.sum(positive * np.log(positive)))

    entropy_norm = float(entropy / np.log(n_cells)) if n_cells > 1 else 0.0
    peak_mass = float(np.max(pmf))
    topk = min(5, n_cells)
    top5_mass = float(np.sum(np.partition(pmf, -topk)[-topk:]))
    effective_support = float(np.exp(entropy))
    support_fraction = float(support_size / n_cells)

    return {
        'pmf_entropy': entropy,
        'pmf_entropy_norm': entropy_norm,
        'pmf_peak_mass': peak_mass,
        'pmf_top5_mass': top5_mass,
        'pmf_effective_support': effective_support,
        'pmf_support_size': support_size,
        'pmf_support_fraction': support_fraction,
    }


def compute_peak_hypothesis_residual(
    transmit_power_map: np.ndarray,
    propagation_matrix: np.ndarray,
    observed_powers: np.ndarray,
    pmf: np.ndarray,
) -> np.ndarray:
    """
    Residual vector for the MAP/peak PMF hypothesis after closed-form power fit.

    This is useful for residual diagnostics because it isolates the residual
    structure of the single hypothesis that the PMF ranks highest.
    """
    t_hat = transmit_power_map.ravel()
    peak_idx = int(np.argmax(pmf))

    with np.errstate(divide='ignore'):
        gain_db = 10.0 * np.log10(np.maximum(propagation_matrix[:, peak_idx], 1e-20))

    predicted = t_hat[peak_idx] + gain_db
    return predicted - observed_powers


def summarize_residual_vector(residual: np.ndarray) -> Dict[str, float]:
    """Compact per-experiment summary of the peak-hypothesis residual vector."""
    residual = np.asarray(residual, dtype=np.float64).ravel()
    return {
        'peak_resid_rmse': float(np.sqrt(np.mean(residual**2))),
        'peak_resid_mae': float(np.mean(np.abs(residual))),
        'peak_resid_bias': float(np.mean(residual)),
        'peak_resid_std': float(np.std(residual)),
        'peak_resid_max_abs': float(np.max(np.abs(residual))),
    }


# ---------------------------------------------------------------------------
# Empirical path loss exponent estimation
# ---------------------------------------------------------------------------

def estimate_path_loss_exponent(
    sensor_locations: np.ndarray,
    observed_powers_dB: np.ndarray,
    map_shape: tuple,
    pmf: np.ndarray,
    scale: float = 1.0,
    n_min: float = 1.5,
    n_max: float = 6.0,
) -> float:
    """
    Estimate the path loss exponent from sensor data using the PMF peak
    as the assumed TX location.

    Fits:  P_obs = P_tx - 10*n*log10(d)
    via linear regression of P_obs vs log10(d), giving n = -slope / 10.

    Parameters
    ----------
    sensor_locations : (M, 2) sensor pixel coordinates [col, row]
    observed_powers_dB : (M,) observed powers in dBm
    map_shape : (height, width)
    pmf : (N,) PMF over grid cells (flattened)
    scale : pixel-to-meter scale
    n_min, n_max : clipping bounds for the estimated exponent

    Returns
    -------
    n_est : float  estimated path loss exponent
    """
    height, width = map_shape
    peak_idx = np.argmax(pmf)
    peak_row = peak_idx // width
    peak_col = peak_idx % width

    dx = (sensor_locations[:, 0] - peak_col) * scale
    dy = (sensor_locations[:, 1] - peak_row) * scale
    distances = np.maximum(np.sqrt(dx**2 + dy**2), 1.0)

    log_d = np.log10(distances)

    if len(log_d) >= 3:
        coeffs = np.polyfit(log_d, observed_powers_dB, 1)
        n_est = -coeffs[0] / 10.0
        return float(np.clip(n_est, n_min, n_max))

    return 3.5  # fallback


# ---------------------------------------------------------------------------
# Diagnostic visualizations
# ---------------------------------------------------------------------------

def save_likelihood_diagnostics(
    pmf: np.ndarray,
    transmit_power_map: np.ndarray,
    map_data: Dict,
    sensor_locations: np.ndarray,
    observed_powers_dB: np.ndarray,
    val_points: np.ndarray,
    observed_val_dBm: np.ndarray,
    predicted_val_dBm: np.ndarray,
    noise_floor: float,
    metrics: Dict,
    true_tx_locations: Optional[Dict],
    output_dir: Path,
    experiment_name: str,
):
    """
    Generate publication-quality diagnostic plots for a single likelihood
    reconstruction experiment.

    Produces four separate figures:
      1. PMF map (contourf) with sensors and true TX locations
      2. Predicted signal strength map with validation points overlaid
      3. Observed vs predicted scatter plot
      4. Spatial reconstruction error at validation points
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    height, width = map_data['shape']
    UTM_lat = map_data['UTM_lat']
    UTM_long = map_data['UTM_long']

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Publication defaults
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
    })

    x = np.linspace(0, width, width, endpoint=False)
    y = np.linspace(0, height, height, endpoint=False)
    X, Y = np.meshgrid(x, y)

    def _set_utm_ticks(ax):
        interval = max(1, len(UTM_lat) // 5)
        xt = list(range(0, len(UTM_lat), interval))
        xl = [f'{UTM_lat[i]:.1f}' for i in xt]
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=12)
        interval = max(1, len(UTM_long) // 5)
        yt = list(range(0, len(UTM_long), interval))
        yl = [f'{UTM_long[i]:.1f}' for i in yt]
        ax.set_yticks(yt)
        ax.set_yticklabels(yl, fontsize=12, rotation=90)
        ax.set_xlabel('UTM$_E$ [m]', fontsize=16, labelpad=8)
        ax.set_ylabel('UTM$_N$ [m]', fontsize=16, labelpad=8)

    def _add_tx_markers(ax):
        if true_tx_locations and len(true_tx_locations) > 0:
            tx_coords = np.array([loc['coordinates'] for loc in true_tx_locations.values()])
            ax.scatter(tx_coords[:, 0], tx_coords[:, 1],
                       marker='x', s=200, c='blue', linewidth=3,
                       label='Transmitter Locations', zorder=10)

    # --- Figure 1: PMF Map ---
    fig1, ax1 = plt.subplots(figsize=(13, 8))
    pmf_2d = pmf.reshape(height, width)
    cf = ax1.contourf(X, Y, pmf_2d, 100, cmap='hot')
    cbar = plt.colorbar(cf, ax=ax1, label='Probability Mass')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Probability Mass', size=16)

    sc = ax1.scatter(sensor_locations[:, 0], sensor_locations[:, 1],
                     c=observed_powers_dB, s=150, edgecolor='green',
                     linewidth=2, cmap='hot', label='Monitoring Locations',
                     zorder=6)
    scatter_cbar = plt.colorbar(sc, ax=ax1, label='Observed Signal Strength [dBm]',
                                location='left')
    scatter_cbar.ax.tick_params(labelsize=14)
    scatter_cbar.set_label('Observed Signal Strength [dBm]', size=16)

    _add_tx_markers(ax1)
    # Mark PMF peak
    peak_idx = np.argmax(pmf)
    peak_row, peak_col = peak_idx // width, peak_idx % width
    ax1.scatter(peak_col, peak_row, marker='*', s=400, c='magenta',
                edgecolor='white', linewidth=1.5, zorder=12,
                label=f'PMF Peak')
    ax1.legend(fontsize=11)
    _set_utm_ticks(ax1)
    ax1.set_title(f'2D PMF of Transmitter Location — {experiment_name}',
                  fontsize=18)
    # Adjust colorbar positions
    cbar.ax.set_position([0.77, 0.1, 0.04, 0.8])
    scatter_cbar.ax.set_position([0.18, 0.1, 0.04, 0.8])

    fig1.savefig(plots_dir / f'{experiment_name}_pmf.png', dpi=200,
                 bbox_inches='tight')
    plt.close(fig1)

    # --- Figure 2: Predicted Signal Strength with Validation Overlay ---
    fig2, ax2 = plt.subplots(figsize=(13, 8))

    # Build marginalized signal-strength map for visualization
    # For the full grid: predicted_linear = A_model^T @ (tx_linear * pmf)
    # We already have transmit_power_map and pmf; compute per-pixel prediction
    t_hat = transmit_power_map.ravel()
    tx_linear = 10.0 ** (t_hat / 10.0)
    nz = pmf > 0
    weighted = tx_linear[nz] * pmf[nz]
    # Sum weighted TX powers to get a scalar "total predicted power at each
    # grid cell" is not meaningful without a propagation matrix for the grid
    # itself. Instead, show the transmit power map filtered by PMF support.
    # Use the validation-point predictions directly on the spatial plot.

    # Validation points colored by prediction error (diverging)
    from matplotlib.colors import TwoSlopeNorm
    val_error_dB = predicted_val_dBm - observed_val_dBm
    abs_max_err = max(abs(val_error_dB.min()), abs(val_error_dB.max()), 1.0)
    err_norm = TwoSlopeNorm(vmin=-abs_max_err, vcenter=0, vmax=abs_max_err)

    sc_err = ax2.scatter(val_points[:, 0], val_points[:, 1],
                         c=val_error_dB, s=14, alpha=1.0,
                         cmap='RdBu_r', norm=err_norm,
                         edgecolor='none', zorder=3,
                         label='Validation Error')
    cbar2 = plt.colorbar(sc_err, ax=ax2, label='Prediction Error [dB]')
    cbar2.ax.tick_params(labelsize=14)
    cbar2.set_label('Prediction Error (Pred − Obs) [dB]', size=16)

    # Monitoring locations colored by observed power
    sc_obs = ax2.scatter(sensor_locations[:, 0], sensor_locations[:, 1],
                         c=observed_powers_dB, s=150, edgecolor='green',
                         linewidth=2, cmap='hot',
                         label='Monitoring Locations', zorder=6)
    obs_cbar = plt.colorbar(sc_obs, ax=ax2,
                            label='Observed Signal Strength [dBm]',
                            location='left')
    obs_cbar.ax.tick_params(labelsize=14)
    obs_cbar.set_label('Observed Signal Strength [dBm]', size=16)

    # Show truncated PMF support as estimated TX region
    nz_indices = np.where(pmf > 0)[0]
    if len(nz_indices) > 0:
        pmf_rows = nz_indices // width
        pmf_cols = nz_indices % width
        pmf_vals = pmf[nz_indices]
        # Scale marker size by relative PMF weight (min 15, max 120)
        pmf_norm = pmf_vals / pmf_vals.max()
        sizes = 15 + 105 * pmf_norm
        ax2.scatter(pmf_cols, pmf_rows, c='cyan', s=sizes, alpha=0.2,
                    marker='s', edgecolor='none', zorder=1,
                    label=f'Estimated TX Region ({len(nz_indices)} cells)')
    # Mark PMF peak explicitly
    peak_idx = np.argmax(pmf)
    peak_row, peak_col = peak_idx // width, peak_idx % width
    ax2.scatter(peak_col, peak_row, marker='*', s=350, c='cyan',
                edgecolor='black', linewidth=1.5, zorder=11,
                label='PMF Peak')

    _add_tx_markers(ax2)
    ax2.legend(fontsize=11)
    _set_utm_ticks(ax2)
    ax2.set_title(
        f'Reconstruction Error & Estimated TX — {experiment_name}\n'
        f'RMSE={metrics["rmse"]:.1f} dB | MAE={metrics["mae"]:.1f} dB | '
        f'Bias={metrics["bias"]:.1f} dB',
        fontsize=16)
    cbar2.ax.set_position([0.77, 0.1, 0.04, 0.8])
    obs_cbar.ax.set_position([0.18, 0.1, 0.04, 0.8])

    fig2.savefig(plots_dir / f'{experiment_name}_signal.png', dpi=200,
                 bbox_inches='tight')
    plt.close(fig2)

    # --- Figure 3: Observed vs Predicted Scatter ---
    fig3, ax3 = plt.subplots(figsize=(9, 9))
    ax3.scatter(observed_val_dBm, predicted_val_dBm, s=10, alpha=0.4,
                edgecolor='none', c='steelblue')

    lo = min(observed_val_dBm.min(), predicted_val_dBm.min()) - 5
    hi = max(observed_val_dBm.max(), predicted_val_dBm.max()) + 5
    ax3.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='1:1')
    ax3.axhline(noise_floor, color='gray', linestyle=':', linewidth=1,
                label=f'Noise floor ({noise_floor:.0f} dBm)')

    ax3.set_xlabel('Observed RSS [dBm]', fontsize=16)
    ax3.set_ylabel('Predicted RSS [dBm]', fontsize=16)
    ax3.set_title(
        f'Observed vs Predicted — {experiment_name}\n'
        f'RMSE={metrics["rmse"]:.1f} dB | MAE={metrics["mae"]:.1f} dB | '
        f'Bias={metrics["bias"]:.1f} dB | N={metrics["n_samples"]}',
        fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_xlim(lo, hi)
    ax3.set_ylim(lo, hi)

    fig3.tight_layout()
    fig3.savefig(plots_dir / f'{experiment_name}_scatter.png', dpi=200,
                 bbox_inches='tight')
    plt.close(fig3)



# ---------------------------------------------------------------------------
# Single-directory experiment
# ---------------------------------------------------------------------------

def run_likelihood_experiment(
    data_info: Dict,
    config: Dict,
    map_data: Dict,
    model_type: str = 'log_distance',
    sigma: float = 4.5,
    delta_c: float = 50.0,
    pmf_threshold: float = 1e-6,
    fit_exponent: bool = False,
    cov_type: str = 'exponential',
    verbose: bool = False,
    save_plots: bool = False,
    plot_output_dir: Optional[Path] = None,
    all_tx_locations: Optional[Dict] = None,
    np_exponent_override: Optional[float] = None,
) -> Optional[Dict]:
    """
    Run likelihood-based reconstruction for one data directory.

    Returns a dict of reconstruction metrics comparable to the GLRT ablation
    study, or None on hard failure.
    """
    transmitters = data_info['transmitters']
    tx_underscore = "_".join(transmitters)
    data_dir = data_info['path']
    seed = data_info['seed']
    num_locations = data_info.get('num_locations')
    scale = config['spatial']['proxel_size']
    np_exponent = np_exponent_override if np_exponent_override is not None else config['localization']['path_loss_exponent']

    result_template = {
        'dir_name': data_info['name'],
        'tx_count': len(transmitters),
        'transmitters': ','.join(transmitters),
        'seed': seed,
        'num_locations': num_locations,
        'model_type': model_type,
        'sigma': sigma,
        'delta_c': delta_c,
        'pmf_threshold': pmf_threshold,
        'n_pmf_active': 0,
        'recon_rmse': np.nan,
        'recon_mae': np.nan,
        'recon_bias': np.nan,
        'recon_max_error': np.nan,
        'recon_n_val_points': 0,
        'recon_noise_floor': np.nan,
        'recon_status': 'unknown',
        'runtime_s': 0.0,
        'np_exponent_used': np.nan,
        'obs_min_dbm': np.nan,
        'obs_mean_dbm': np.nan,
        'obs_max_dbm': np.nan,
        'pmf_entropy': np.nan,
        'pmf_entropy_norm': np.nan,
        'pmf_peak_mass': np.nan,
        'pmf_top5_mass': np.nan,
        'pmf_effective_support': np.nan,
        'pmf_support_size': 0,
        'pmf_support_fraction': np.nan,
        'peak_resid_rmse': np.nan,
        'peak_resid_mae': np.nan,
        'peak_resid_bias': np.nan,
        'peak_resid_std': np.nan,
        'peak_resid_max_abs': np.nan,
    }

    try:
        # --- Config & data paths ---
        config_id = tx_underscore
        if num_locations is not None:
            config_id = f"{config_id}_nloc{num_locations}"
        if seed is not None:
            config_id = f"{config_id}_seed_{seed}"
        config_path = f'config/monitoring_locations_{config_id}.yaml'

        if not Path(config_path).exists():
            if verbose:
                print(f"  Config not found: {config_path}")
            result_template['recon_status'] = 'no_config'
            return result_template

        locations_config = load_monitoring_locations(config_path=config_path, map_data=map_data)
        sensor_locations = get_sensor_locations_array(locations_config)

        powers_file = data_dir / f"{tx_underscore}_avg_powers.npy"
        if not powers_file.exists():
            if verbose:
                print(f"  Powers not found: {powers_file}")
            result_template['recon_status'] = 'no_powers'
            return result_template

        observed_powers_dB = np.load(powers_file)
        result_template['obs_min_dbm'] = float(np.min(observed_powers_dB))
        result_template['obs_mean_dbm'] = float(np.mean(observed_powers_dB))
        result_template['obs_max_dbm'] = float(np.max(observed_powers_dB))

        start_time = time.time()

        # --- Model config path ---
        if model_type == 'tirem':
            model_config_path = 'config/tirem_parameters.yaml'
        elif model_type == 'raytracing':
            model_config_path = 'config/sionna_parameters.yaml'
        else:
            model_config_path = None

        # --- 1) Propagation matrix (sensors x grid) ---
        A_model = compute_propagation_matrix(
            sensor_locations=sensor_locations,
            map_shape=map_data['shape'],
            scale=scale,
            model_type=model_type,
            config_path=model_config_path,
            np_exponent=np_exponent,
            n_jobs=-1,
            verbose=False,
        )

        # --- 2) Transmit power map (closed-form with propagation matrix) ---
        transmit_power_map = estimate_transmit_power_map(
            map_shape=map_data['shape'],
            sensor_locations=sensor_locations,
            observed_powers=observed_powers_dB,
            scale=scale,
            np_exponent=np_exponent,
            n_jobs=1,
            verbose=False,
            propagation_matrix=A_model,
        )

        # --- 3) Covariance matrix ---
        if cov_type == 'identity':
            M = len(sensor_locations)
            cov_matrix = (sigma ** 2) * np.eye(M)
        else:
            cov_matrix = build_covariance_matrix(
                sensor_locations=sensor_locations,
                sigma=sigma,
                delta_c=delta_c,
                scale=scale,
            )

        # --- 4) PMF (vectorized) ---
        pmf = compute_pmf_vectorized(
            transmit_power_map=transmit_power_map,
            propagation_matrix=A_model,
            observed_powers=observed_powers_dB,
            cov_matrix=cov_matrix,
            threshold=pmf_threshold,
        )

        # --- 4b) Optionally fit exponent from sensor data and recompute ---
        if fit_exponent and model_type == 'log_distance':
            n_fitted = estimate_path_loss_exponent(
                sensor_locations=sensor_locations,
                observed_powers_dB=observed_powers_dB,
                map_shape=map_data['shape'],
                pmf=pmf,
                scale=scale,
            )
            if verbose:
                print(f"  Fitted exponent: {n_fitted:.2f} (config: {np_exponent:.2f})")
            np_exponent = n_fitted

            # Recompute propagation matrix, TX power map, and PMF with fitted exponent
            A_model = compute_propagation_matrix(
                sensor_locations=sensor_locations,
                map_shape=map_data['shape'],
                scale=scale,
                model_type=model_type,
                config_path=model_config_path,
                np_exponent=np_exponent,
                n_jobs=-1,
                verbose=False,
            )
            transmit_power_map = estimate_transmit_power_map(
                map_shape=map_data['shape'],
                sensor_locations=sensor_locations,
                observed_powers=observed_powers_dB,
                scale=scale,
                np_exponent=np_exponent,
                n_jobs=1,
                verbose=False,
                propagation_matrix=A_model,
            )
            pmf = compute_pmf_vectorized(
                transmit_power_map=transmit_power_map,
                propagation_matrix=A_model,
                observed_powers=observed_powers_dB,
                cov_matrix=cov_matrix,
                threshold=pmf_threshold,
            )

        result_template['np_exponent_used'] = np_exponent
        result_template['n_pmf_active'] = int(np.sum(pmf > 0))
        result_template.update(
            compute_pmf_sharpness_metrics(pmf, threshold=pmf_threshold)
        )

        peak_residual = compute_peak_hypothesis_residual(
            transmit_power_map=transmit_power_map,
            propagation_matrix=A_model,
            observed_powers=observed_powers_dB,
            pmf=pmf,
        )
        result_template.update(summarize_residual_vector(peak_residual))

        # --- 5) Validation ---
        val_config, val_data_dir = get_validation_paths(transmitters, _PROJECT_ROOT)
        if val_config is None:
            result_template['recon_status'] = 'no_validation_data'
            result_template['runtime_s'] = time.time() - start_time
            return result_template

        # Suppress validator print output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            validator = ReconstructionValidator(
                map_data=map_data,
                validation_config_path=str(val_config),
                validation_data_dir=str(val_data_dir),
            )
            tx_id = normalize_tx_id(transmitters)
            validator.load_observed_data(f"validation_{tx_id}")
            validator.filter_out_of_bounds(verbose=False)
        finally:
            sys.stdout = old_stdout

        # Validation propagation matrix — use validator's cache-aware loader
        # for tirem/raytracing (loads pre-computed 698/1221-pt matrices),
        # fall back to direct computation for log_distance (needs np_exponent).
        if model_type in ('tirem', 'raytracing'):
            validator.get_propagation_matrix(
                model_type=model_type,
                model_config_path=model_config_path,
                scale=scale,
                cache_dir=str(_PROJECT_ROOT / 'data' / 'cache'),
                verbose=False,
            )
            A_val = validator.prop_matrix
        else:
            A_val = compute_propagation_matrix(
                sensor_locations=validator.val_points,
                map_shape=map_data['shape'],
                scale=scale,
                model_type=model_type,
                config_path=model_config_path,
                np_exponent=np_exponent,
                n_jobs=-1,
                verbose=False,
            )

        # Marginalized prediction
        predicted_dBm = predict_at_validation_points(
            transmit_power_map=transmit_power_map,
            pmf=pmf,
            A_val=A_val,
        )

        # Noise floor clamping (same as GLRT pipeline)
        nloc = num_locations if num_locations else len(observed_powers_dB)
        noise_floor = compute_noise_floor(observed_powers_dB, nloc)
        predicted_dBm = np.maximum(predicted_dBm, noise_floor)

        # Metrics
        metrics = validator.compute_metrics(predicted_dBm, verbose=False)

        # Diagnostic plots
        if save_plots and plot_output_dir is not None:
            tx_locs = None
            if all_tx_locations:
                tx_locs = {n: all_tx_locations[n] for n in transmitters if n in all_tx_locations}
            try:
                save_likelihood_diagnostics(
                    pmf=pmf,
                    transmit_power_map=transmit_power_map,
                    map_data=map_data,
                    sensor_locations=sensor_locations,
                    observed_powers_dB=observed_powers_dB,
                    val_points=validator.val_points,
                    observed_val_dBm=validator.observed_powers_dBm,
                    predicted_val_dBm=predicted_dBm,
                    noise_floor=noise_floor,
                    metrics=metrics,
                    true_tx_locations=tx_locs,
                    output_dir=plot_output_dir,
                    experiment_name=data_info['name'],
                )
            except Exception as plot_err:
                if verbose:
                    print(f"  Plot failed: {plot_err}")

        elapsed = time.time() - start_time

        result_template.update({
            'n_pmf_active': int(np.sum(pmf > 0)),
            'recon_rmse': metrics['rmse'],
            'recon_mae': metrics['mae'],
            'recon_bias': metrics['bias'],
            'recon_max_error': max(abs(metrics['min_error']), abs(metrics['max_error'])),
            'recon_n_val_points': metrics['n_samples'],
            'recon_noise_floor': noise_floor,
            'recon_status': 'success',
            'runtime_s': elapsed,
        })
        return result_template

    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        result_template['recon_status'] = f'error:{str(e)[:60]}'
        return result_template


# ---------------------------------------------------------------------------
# Worker wrapper (for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker_init():
    """Initialize worker process."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


def _worker(args_tuple):
    """Unpack arguments and run a single experiment."""
    (data_info_ser, config, map_data, model_type, sigma, delta_c,
     pmf_threshold, verbose, save_plots, plot_output_dir_str,
     all_tx_locations, fit_exponent, cov_type, np_exponent_override) = args_tuple

    # Reconstruct non-serializable objects
    data_info = data_info_ser.copy()
    data_info['path'] = Path(data_info['path_str'])
    plot_output_dir = Path(plot_output_dir_str) if plot_output_dir_str else None

    return run_likelihood_experiment(
        data_info=data_info,
        config=config,
        map_data=map_data,
        model_type=model_type,
        sigma=sigma,
        delta_c=delta_c,
        pmf_threshold=pmf_threshold,
        fit_exponent=fit_exponent,
        cov_type=cov_type,
        verbose=verbose,
        save_plots=save_plots,
        plot_output_dir=plot_output_dir,
        all_tx_locations=all_tx_locations,
        np_exponent_override=np_exponent_override,
    )


# ---------------------------------------------------------------------------
# Parameter Sensitivity Studies
# ---------------------------------------------------------------------------

def _run_sweep_single(
    data_info: Dict,
    config: Dict,
    map_data: Dict,
    model_type: str,
    sigma: float,
    delta_c: float,
    pmf_threshold: float,
    fit_exponent: bool,
    cov_type: str,
    np_exponent_override: Optional[float],
) -> Optional[Dict]:
    """Run a single experiment for the parameter sweep (sequential helper)."""
    return run_likelihood_experiment(
        data_info=data_info,
        config=config,
        map_data=map_data,
        model_type=model_type,
        sigma=sigma,
        delta_c=delta_c,
        pmf_threshold=pmf_threshold,
        fit_exponent=fit_exponent,
        cov_type=cov_type,
        verbose=False,
        save_plots=False,
        np_exponent_override=np_exponent_override,
    )


def run_parameter_study(
    args,
    config: Dict,
    map_data: Dict,
    all_dirs: list,
    output_dir: Path,
    all_tx_locations: Optional[Dict],
):
    """
    Run a parameter sensitivity study, sweeping over covariance parameters
    (sigma, delta_c) or path loss exponent (np).

    Results are collected per parameter combination, summarized, and plotted.
    """
    study_type = args.study
    total_dirs = len(all_dirs)

    n_workers = args.workers
    if n_workers == -1:
        n_workers = max(1, os.cpu_count() - 1)

    # Serialize data_info dicts (Path objects aren't picklable)
    all_dirs_ser = []
    for d in all_dirs:
        all_dirs_ser.append({
            'name': d['name'],
            'transmitters': d['transmitters'],
            'num_locations': d.get('num_locations'),
            'seed': d['seed'],
            'path_str': str(d['path']),
        })

    # --- Build flat list of task argument tuples ---
    if study_type == 'covariance':
        sigma_values = [float(x.strip()) for x in args.sigma_values.split(',')]
        delta_c_values = [float(x.strip()) for x in args.delta_c_values.split(',')]
        param_combos = [(s, dc) for s in sigma_values for dc in delta_c_values]
        total_runs = len(param_combos) * total_dirs

        print(f"\n{'='*70}")
        print(f"COVARIANCE PARAMETER SENSITIVITY STUDY")
        print(f"  sigma values:   {sigma_values}")
        print(f"  delta_c values: {delta_c_values}")
        print(f"  Parameter combos: {len(param_combos)}")
        print(f"  Directories:      {total_dirs}")
        print(f"  Total runs:       {total_runs}")
        print(f"  Workers:          {n_workers}")
        print(f"{'='*70}\n")

        task_args = []
        for sigma, delta_c in param_combos:
            for d_ser in all_dirs_ser:
                task_args.append(
                    (d_ser, config, map_data, 'log_distance', sigma, delta_c,
                     args.pmf_threshold, False, False, None,
                     None, args.fit_exponent, 'exponential', None)
                )

    elif study_type == 'path_loss':
        np_values = [float(x.strip()) for x in args.np_values.split(',')]
        total_runs = len(np_values) * total_dirs

        print(f"\n{'='*70}")
        print(f"PATH LOSS EXPONENT SENSITIVITY STUDY")
        print(f"  np values:   {np_values}")
        print(f"  Directories: {total_dirs}")
        print(f"  Total runs:  {total_runs}")
        print(f"  Workers:     {n_workers}")
        print(f"{'='*70}\n")

        task_args = []
        for np_val in np_values:
            for d_ser in all_dirs_ser:
                task_args.append(
                    (d_ser, config, map_data, 'log_distance', args.sigma, args.delta_c,
                     args.pmf_threshold, False, False, None,
                     None, False, args.cov_type, np_val)
                )

    else:
        print(f"Unknown study type: {study_type}")
        return

    # --- Resume: load previous checkpoint and filter out completed tasks ---
    checkpoint_csv = output_dir / f'study_{study_type}_checkpoint.csv'
    resumed_results = []

    if args.resume and checkpoint_csv.exists():
        prev_df = pd.read_csv(checkpoint_csv)
        resumed_results = prev_df.to_dict('records')

        # Build set of completed (dir_name, sigma, delta_c, np_exponent) keys
        done_keys = set()
        for _, row in prev_df.iterrows():
            if study_type == 'covariance':
                done_keys.add((row['dir_name'], float(row['sigma']), float(row['delta_c'])))
            elif study_type == 'path_loss':
                np_col = 'np_exponent_param' if 'np_exponent_param' in prev_df.columns else 'np_exponent_used'
                done_keys.add((row['dir_name'], float(row[np_col])))

        # Filter out already-completed tasks
        original_count = len(task_args)
        filtered_args = []
        for ta in task_args:
            d_ser = ta[0]
            if study_type == 'covariance':
                key = (d_ser['name'], ta[4], ta[5])  # (dir_name, sigma, delta_c)
            else:
                key = (d_ser['name'], ta[13])  # (dir_name, np_exponent)
            if key not in done_keys:
                filtered_args.append(ta)
        task_args = filtered_args

        print(f"  Resumed from checkpoint: {len(resumed_results)} completed, "
              f"{len(task_args)} remaining (of {original_count} total)")
        total_runs = original_count  # keep total for progress display
    else:
        total_runs = len(task_args)

    # --- Execute tasks (sequential or parallel) ---
    all_results = list(resumed_results)
    start_time = time.time()
    CHECKPOINT_INTERVAL = 20
    tasks_remaining = len(task_args)

    def _flush_study_checkpoint():
        if all_results:
            pd.DataFrame(all_results).to_csv(checkpoint_csv, index=False)

    if tasks_remaining == 0:
        print("  All tasks already completed — skipping to summary/plots.")
    elif n_workers <= 1:
        for i, ta in enumerate(task_args):
            result = _worker(ta)
            if result is not None:
                if study_type == 'path_loss' and ta[13] is not None:
                    result['np_exponent_param'] = ta[13]
                all_results.append(result)

            completed = len(all_results)
            if (i + 1) % 20 == 0 or (i + 1) <= 3:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (tasks_remaining - i - 1) / rate / 60 if rate > 0 else 0
                print(f"  [{completed}/{total_runs}] {elapsed/60:.1f}min elapsed, "
                      f"~{remaining:.1f}min remaining")

            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                _flush_study_checkpoint()

        _flush_study_checkpoint()
    else:
        print(f"  Using {n_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_worker_init) as executor:
            futures = {executor.submit(_worker, ta): ta for ta in task_args}
            done_count = 0
            for future in as_completed(futures):
                ta = futures[future]
                done_count += 1
                try:
                    result = future.result()
                except Exception as e:
                    result = None
                    print(f"  Worker exception: {e}")
                if result is not None:
                    if study_type == 'path_loss' and ta[13] is not None:
                        result['np_exponent_param'] = ta[13]
                    all_results.append(result)

                if done_count % 20 == 0 or done_count <= 5:
                    elapsed = time.time() - start_time
                    rate = done_count / elapsed if elapsed > 0 else 0
                    remaining = (tasks_remaining - done_count) / rate / 60 if rate > 0 else 0
                    print(f"  [{len(all_results)}/{total_runs}] {elapsed/60:.1f}min elapsed, "
                          f"~{remaining:.1f}min remaining")

                if done_count % CHECKPOINT_INTERVAL == 0:
                    _flush_study_checkpoint()

        _flush_study_checkpoint()

    raw_df = pd.DataFrame(all_results)

    # --- Save final results ---
    elapsed_total = time.time() - start_time
    raw_csv = output_dir / f'study_{study_type}_raw_results.csv'
    raw_df.to_csv(raw_csv, index=False)
    print(f"\nRaw results saved: {raw_csv} ({len(raw_df)} rows)")

    # --- Generate summary & plots ---
    success_df = raw_df[raw_df['recon_status'] == 'success'].copy()
    if success_df.empty:
        print("No successful results to summarize.")
        return

    if study_type == 'covariance':
        _summarize_covariance_study(success_df, output_dir)
        _plot_covariance_study(success_df, output_dir)
    elif study_type == 'path_loss':
        _summarize_path_loss_study(success_df, output_dir)
        _plot_path_loss_study(success_df, output_dir)

    print(f"\n{'='*70}")
    print(f"Parameter study '{study_type}' complete!")
    print(f"  Results: {output_dir}")
    print(f"  Runtime: {elapsed_total/60:.1f} minutes")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Covariance study summary & plots
# ---------------------------------------------------------------------------

def _summarize_covariance_study(df: pd.DataFrame, output_dir: Path):
    """Print and save summary table for the covariance parameter sweep."""
    summary_rows = []
    for (sigma, delta_c), group in df.groupby(['sigma', 'delta_c']):
        row = {
            'sigma': sigma,
            'delta_c': delta_c,
            'n_dirs': len(group),
        }
        for metric in ['recon_rmse', 'recon_mae', 'recon_bias', 'recon_max_error']:
            vals = group[metric].dropna()
            row[f'{metric}_mean'] = vals.mean() if len(vals) > 0 else np.nan
            row[f'{metric}_std'] = vals.std() if len(vals) > 0 else np.nan
            row[f'{metric}_median'] = vals.median() if len(vals) > 0 else np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / 'study_covariance_summary.csv'
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n{'='*70}")
    print("COVARIANCE STUDY SUMMARY")
    print(f"{'='*70}")
    print(f"  {'sigma':<8s} {'delta_c':<10s} {'N':<6s} {'RMSE (dB)':<16s} {'MAE (dB)':<16s} {'Bias (dB)':<16s}")
    print(f"  {'-'*8} {'-'*10} {'-'*6} {'-'*16} {'-'*16} {'-'*16}")
    for _, row in summary_df.iterrows():
        rmse = f"{row['recon_rmse_mean']:.2f}+/-{row['recon_rmse_std']:.2f}"
        mae = f"{row['recon_mae_mean']:.2f}+/-{row['recon_mae_std']:.2f}"
        bias = f"{row['recon_bias_mean']:.2f}+/-{row['recon_bias_std']:.2f}"
        print(f"  {row['sigma']:<8.1f} {row['delta_c']:<10.0f} {int(row['n_dirs']):<6d} "
              f"{rmse:<16s} {mae:<16s} {bias:<16s}")

    print(f"\n  Summary saved: {summary_csv}")


def _plot_covariance_study(df: pd.DataFrame, output_dir: Path):
    """Generate heatmaps and grouped bar charts for the covariance study."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

    sigma_values = sorted(df['sigma'].unique())
    delta_c_values = sorted(df['delta_c'].unique())

    def _save(fig, name):
        fig.savefig(plots_dir / f'{name}.pdf', format='pdf')
        fig.savefig(plots_dir / f'{name}.png', format='png')
        plt.close(fig)

    # --- Heatmaps of mean RMSE and MAE ---
    for metric_col, metric_label in [('recon_rmse', 'RMSE'), ('recon_mae', 'MAE')]:
        grid = np.full((len(sigma_values), len(delta_c_values)), np.nan)
        for i, sigma in enumerate(sigma_values):
            for j, delta_c in enumerate(delta_c_values):
                vals = df[(df['sigma'] == sigma) & (df['delta_c'] == delta_c)][metric_col].dropna()
                if len(vals) > 0:
                    grid[i, j] = vals.mean()

        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(grid, cmap='RdYlGn_r', aspect='auto', origin='lower')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'Mean {metric_label} (dB)', fontsize=14)

        ax.set_xticks(range(len(delta_c_values)))
        ax.set_xticklabels([f'{d:.0f}' for d in delta_c_values])
        ax.set_yticks(range(len(sigma_values)))
        ax.set_yticklabels([f'{s:.1f}' for s in sigma_values])
        ax.set_xlabel(r'Decorrelation Distance $\delta_c$ [m]')
        ax.set_ylabel(r'Shadow Fading Std. Dev. $\sigma$ [dB]')
        ax.set_title(f'Mean {metric_label} — Covariance Parameter Sensitivity\n'
                     f'Log-Distance Model  |  N={len(df)} per combo (successful)')

        # Annotate cells with values
        for i in range(len(sigma_values)):
            for j in range(len(delta_c_values)):
                if np.isfinite(grid[i, j]):
                    text_color = 'white' if grid[i, j] > np.nanmean(grid) else 'black'
                    ax.text(j, i, f'{grid[i, j]:.1f}', ha='center', va='center',
                            fontsize=13, fontweight='bold', color=text_color)

        fig.tight_layout()
        _save(fig, f'study_covariance_heatmap_{metric_col}')

    # --- Grouped bar chart: RMSE by sigma, grouped by delta_c ---
    COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9']

    fig, ax = plt.subplots(figsize=(8, 5))
    n_sigma = len(sigma_values)
    n_delta = len(delta_c_values)
    bar_width = 0.8 / n_delta
    x = np.arange(n_sigma)

    for j, delta_c in enumerate(delta_c_values):
        means = []
        sems = []
        for sigma in sigma_values:
            vals = df[(df['sigma'] == sigma) & (df['delta_c'] == delta_c)]['recon_rmse'].dropna()
            means.append(vals.mean() if len(vals) > 0 else np.nan)
            sems.append(vals.sem() if len(vals) > 1 else 0)

        offset = (j - n_delta / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=sems,
                      color=COLORS[j % len(COLORS)], capsize=3,
                      edgecolor='white', linewidth=0.5,
                      label=f'$\\delta_c$={delta_c:.0f} m')

        for k, (m, s) in enumerate(zip(means, sems)):
            if np.isfinite(m):
                ax.text(x[k] + offset, m + s + 0.2, f'{m:.1f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{s:.1f}' for s in sigma_values])
    ax.set_xlabel(r'Shadow Fading Std. Dev. $\sigma$ [dB]')
    ax.set_ylabel('Mean RMSE (dB)')
    ax.set_title('Reconstruction RMSE — Covariance Parameter Sensitivity')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'study_covariance_grouped_bar_rmse')

    # --- Box plots: one per (sigma, delta_c) combo ---
    fig, ax = plt.subplots(figsize=(max(8, len(sigma_values) * len(delta_c_values) * 0.8), 5))
    combo_data = []
    combo_labels = []
    for sigma in sigma_values:
        for delta_c in delta_c_values:
            vals = df[(df['sigma'] == sigma) & (df['delta_c'] == delta_c)]['recon_rmse'].dropna()
            if len(vals) > 0:
                combo_data.append(vals.values)
                combo_labels.append(f'$\\sigma$={sigma:.1f}\n$\\delta_c$={delta_c:.0f}')

    if combo_data:
        bp = ax.boxplot(combo_data, labels=combo_labels, patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

    ax.set_ylabel('RMSE (dB)')
    ax.set_title('RMSE Distribution — Covariance Parameter Combinations')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'study_covariance_boxplot_rmse')

    print(f"  Covariance study plots saved to: {plots_dir}")


# ---------------------------------------------------------------------------
# Path loss exponent study summary & plots
# ---------------------------------------------------------------------------

def _summarize_path_loss_study(df: pd.DataFrame, output_dir: Path):
    """Print and save summary table for the path loss exponent sweep."""
    np_col = 'np_exponent_param' if 'np_exponent_param' in df.columns else 'np_exponent_used'

    summary_rows = []
    for np_val, group in df.groupby(np_col):
        row = {
            'np_exponent': np_val,
            'n_dirs': len(group),
        }
        for metric in ['recon_rmse', 'recon_mae', 'recon_bias', 'recon_max_error']:
            vals = group[metric].dropna()
            row[f'{metric}_mean'] = vals.mean() if len(vals) > 0 else np.nan
            row[f'{metric}_std'] = vals.std() if len(vals) > 0 else np.nan
            row[f'{metric}_median'] = vals.median() if len(vals) > 0 else np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / 'study_path_loss_summary.csv'
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n{'='*70}")
    print("PATH LOSS EXPONENT STUDY SUMMARY")
    print(f"{'='*70}")
    print(f"  {'n_p':<8s} {'N':<6s} {'RMSE (dB)':<16s} {'MAE (dB)':<16s} {'Bias (dB)':<16s}")
    print(f"  {'-'*8} {'-'*6} {'-'*16} {'-'*16} {'-'*16}")
    for _, row in summary_df.iterrows():
        rmse = f"{row['recon_rmse_mean']:.2f}+/-{row['recon_rmse_std']:.2f}"
        mae = f"{row['recon_mae_mean']:.2f}+/-{row['recon_mae_std']:.2f}"
        bias = f"{row['recon_bias_mean']:.2f}+/-{row['recon_bias_std']:.2f}"
        print(f"  {row['np_exponent']:<8.1f} {int(row['n_dirs']):<6d} "
              f"{rmse:<16s} {mae:<16s} {bias:<16s}")

    print(f"\n  Summary saved: {summary_csv}")


def _plot_path_loss_study(df: pd.DataFrame, output_dir: Path):
    """Generate bar charts and box plots for the path loss exponent study."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    np_col = 'np_exponent_param' if 'np_exponent_param' in df.columns else 'np_exponent_used'
    np_values = sorted(df[np_col].unique())

    COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9']

    def _save(fig, name):
        fig.savefig(plots_dir / f'{name}.pdf', format='pdf')
        fig.savefig(plots_dir / f'{name}.png', format='png')
        plt.close(fig)

    # --- Bar chart: RMSE, MAE, Bias by np ---
    metrics = [
        ('recon_rmse', 'RMSE (dB)'),
        ('recon_mae', 'MAE (dB)'),
        ('recon_bias', 'Bias (dB)'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.5))
    x = np.arange(len(np_values))
    bar_width = 0.6

    for ax_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]
        means = []
        sems = []
        for np_val in np_values:
            vals = df[df[np_col] == np_val][metric_col].dropna()
            means.append(vals.mean() if len(vals) > 0 else np.nan)
            sems.append(vals.sem() if len(vals) > 1 else 0)

        bars = ax.bar(x, means, bar_width, yerr=sems,
                      color=[COLORS[i % len(COLORS)] for i in range(len(np_values))],
                      capsize=4, edgecolor='white', linewidth=0.5)

        for i, (m, s) in enumerate(zip(means, sems)):
            if np.isfinite(m):
                ax.text(i, m + s + 0.3, f'{m:.1f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:.1f}' for v in np_values])
        ax.set_xlabel(r'Path Loss Exponent $n_p$')
        ax.set_ylabel(metric_label)

    fig.suptitle('Reconstruction Performance — Path Loss Exponent Sensitivity\n'
                 'Log-Distance Model',
                 fontsize=14, fontweight='bold', y=1.05)
    fig.tight_layout()
    _save(fig, 'study_path_loss_bar_metrics')

    # --- Box plots: RMSE distribution per np ---
    fig, ax = plt.subplots(figsize=(7, 5))
    box_data = []
    box_labels = []
    for np_val in np_values:
        vals = df[df[np_col] == np_val]['recon_rmse'].dropna()
        if len(vals) > 0:
            box_data.append(vals.values)
            box_labels.append(f'$n_p$={np_val:.1f}')

    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                        widths=0.5, showfliers=True,
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

    ax.set_ylabel('RMSE (dB)')
    ax.set_title('RMSE Distribution — Path Loss Exponent Sensitivity')
    fig.tight_layout()
    _save(fig, 'study_path_loss_boxplot_rmse')

    # --- Per TX-count breakdown (grouped bar) ---
    tx_counts = sorted(df['tx_count'].unique())
    if len(tx_counts) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        n_np = len(np_values)
        n_tc = len(tx_counts)
        bar_width = 0.8 / n_np
        x = np.arange(n_tc)

        for j, np_val in enumerate(np_values):
            means = []
            sems = []
            for tc in tx_counts:
                vals = df[(df[np_col] == np_val) & (df['tx_count'] == tc)]['recon_rmse'].dropna()
                means.append(vals.mean() if len(vals) > 0 else np.nan)
                sems.append(vals.sem() if len(vals) > 1 else 0)

            offset = (j - n_np / 2 + 0.5) * bar_width
            ax.bar(x + offset, means, bar_width, yerr=sems,
                   color=COLORS[j % len(COLORS)], capsize=3,
                   edgecolor='white', linewidth=0.5,
                   label=f'$n_p$={np_val:.1f}')

        ax.set_xticks(x)
        ax.set_xticklabels([str(tc) for tc in tx_counts])
        ax.set_xlabel('TX Count')
        ax.set_ylabel('Mean RMSE (dB)')
        ax.set_title('RMSE by TX Count — Path Loss Exponent Sensitivity')
        ax.legend(fontsize=11)
        fig.tight_layout()
        _save(fig, 'study_path_loss_by_tx_count')

    print(f"  Path loss study plots saved to: {plots_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run likelihood-based reconstruction for comparison with GLRT ablation study."
    )
    parser.add_argument(
        '--model-type', type=str, default='log_distance',
        choices=['log_distance', 'tirem', 'raytracing'],
        help='Propagation model (default: log_distance)',
    )
    parser.add_argument(
        '--sigma', type=float, default=4.5,
        help='Shadow fading std deviation in dB (default: 4.5)',
    )
    parser.add_argument(
        '--delta-c', type=float, default=50.0,
        help='Decorrelation distance in meters (default: 50)',
    )
    parser.add_argument(
        '--pmf-threshold', type=float, default=1e-6,
        help='PMF probability threshold for marginalization (default: 1e-6)',
    )
    parser.add_argument(
        '--nloc', type=int, default=None,
        help='Filter directories by number of monitoring locations',
    )
    parser.add_argument(
        '--tx-counts', type=str, default=None,
        help='Comma-separated TX counts to include (e.g., "1,2,3")',
    )
    parser.add_argument(
        '--max-dirs', type=int, default=None,
        help='Max directories per TX count',
    )
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Parallel workers (default: 1, -1 = all CPUs minus 1)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (auto-generated if not specified)',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed output per experiment',
    )
    parser.add_argument(
        '--save-plots', action='store_true',
        help='Save diagnostic plots per directory (PMF, scatter, error map)',
    )
    parser.add_argument(
        '--fit-exponent', action='store_true',
        help='Empirically fit path loss exponent from sensor data (log_distance only)',
    )
    parser.add_argument(
        '--cov-type', type=str, default='exponential',
        choices=['exponential', 'identity'],
        help='Covariance matrix type: exponential (spatial decay) or identity (independent sensors)',
    )
    parser.add_argument(
        '--study', type=str, default=None,
        choices=['covariance', 'path_loss'],
        help='Parameter sensitivity study: "covariance" sweeps sigma/delta_c, '
             '"path_loss" sweeps path loss exponent',
    )
    parser.add_argument(
        '--sigma-values', type=str, default='2.0,4.5,8.0',
        help='Comma-separated sigma values for covariance study (default: 2.0,4.5,8.0)',
    )
    parser.add_argument(
        '--delta-c-values', type=str, default='20,50,100',
        help='Comma-separated delta_c values for covariance study (default: 20,50,100)',
    )
    parser.add_argument(
        '--np-values', type=str, default='3.0,3.5,4.0,4.5',
        help='Comma-separated path loss exponent values for path_loss study (default: 3.0,3.5,4.0,4.5)',
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume an incomplete study from a previous output directory '
             '(reads existing checkpoint CSV and skips completed experiments)',
    )
    args = parser.parse_args()

    # --- Output directory ---
    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.exists():
            print(f"ERROR: Resume directory does not exist: {output_dir}")
            sys.exit(1)
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.study:
            parts = [f'likelihood_study_{args.study}_{timestamp}']
            parts.append(f'model_{args.model_type}')
            if args.nloc is not None:
                parts.append(f'nloc_{args.nloc}')
        else:
            parts = [f'likelihood_reconstruction_{timestamp}']
            parts.append(f'model_{args.model_type}')
            if args.nloc is not None:
                parts.append(f'nloc_{args.nloc}')
            parts.append(f'sigma_{args.sigma}_dc_{args.delta_c}')
            if args.cov_type != 'exponential':
                parts.append(f'cov_{args.cov_type}')
        output_dir = Path('results') / '_'.join(parts)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load shared data ---
    print("=" * 70)
    print("LIKELIHOOD RECONSTRUCTION EVALUATION")
    print("=" * 70)

    print("\nLoading configuration...")
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Loading SLC map...")
    map_data = load_slc_map(
        map_folder_dir="./",
        downsample_factor=config['spatial']['downsample_factor'],
    )
    print(f"  Map shape: {map_data['shape']}")

    all_tx_locations = None
    if args.save_plots:
        print("Loading transmitter locations (for plot overlays)...")
        all_tx_locations = load_transmitter_locations(
            config_path='config/transmitter_locations.yaml',
            map_data=map_data,
        )
        print(f"  Transmitters: {list(all_tx_locations.keys())}")

    print("\nDiscovering data directories...")
    base_dir = Path('data/processed')
    grouped_dirs = discover_data_directories(base_dir)

    # Apply filters
    if args.tx_counts:
        tc_filter = [int(x.strip()) for x in args.tx_counts.split(',')]
        grouped_dirs = {k: v for k, v in grouped_dirs.items() if k in tc_filter}
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
        if args.max_dirs:
            dirs = dirs[:args.max_dirs]
        all_dirs.extend(dirs)

    total_dirs = len(all_dirs)
    print(f"  Total directories: {total_dirs}")
    print(f"  TX counts: {sorted(grouped_dirs.keys())}")

    print(f"\nParameters:")
    print(f"  Model:         {args.model_type}")
    print(f"  Sigma:         {args.sigma} dB")
    print(f"  Delta_c:       {args.delta_c} m")
    print(f"  PMF threshold: {args.pmf_threshold}")
    print(f"  Fit exponent:  {args.fit_exponent}")
    print(f"  Cov type:      {args.cov_type}")
    print(f"  Workers:       {args.workers}")
    print(f"  Output:        {output_dir}")

    # --- Save run config ---
    run_config = {
        'model_type': args.model_type,
        'sigma': args.sigma,
        'delta_c': args.delta_c,
        'pmf_threshold': args.pmf_threshold,
        'nloc': args.nloc,
        'tx_counts': args.tx_counts,
        'max_dirs': args.max_dirs,
        'fit_exponent': args.fit_exponent,
        'cov_type': args.cov_type,
        'total_dirs': total_dirs,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)

    # --- Parameter sensitivity study dispatch ---
    if args.study is not None:
        run_parameter_study(
            args=args,
            config=config,
            map_data=map_data,
            all_dirs=all_dirs,
            output_dir=output_dir,
            all_tx_locations=all_tx_locations,
        )
        return

    # --- Run experiments ---
    print(f"\n{'='*70}")
    print(f"Running {total_dirs} experiments...")
    print(f"{'='*70}\n")

    n_workers = args.workers
    if n_workers == -1:
        n_workers = max(1, os.cpu_count() - 1)
    n_workers = min(n_workers, total_dirs)

    start_time = time.time()
    all_results = []
    checkpoint_csv = output_dir / 'likelihood_raw_results_checkpoint.csv'
    last_checkpoint_count = 0
    CHECKPOINT_INTERVAL = 20

    def _flush_checkpoint(results_list, completed_count, total_count):
        """Write incremental checkpoint of raw results to disk."""
        nonlocal last_checkpoint_count
        if len(results_list) == last_checkpoint_count:
            return
        df = pd.DataFrame(results_list)
        df.to_csv(checkpoint_csv, index=False)
        last_checkpoint_count = len(results_list)
        elapsed = time.time() - start_time
        print(f"  >> Checkpoint saved: {len(results_list)} results "
              f"({completed_count}/{total_count} tasks, {elapsed/60:.1f}min)")

    def _log_result(result, dir_name, completed_count, total_count):
        status = result['recon_status'] if result else 'failed'
        rmse_str = (f"{result['recon_rmse']:.1f}"
                    if result and np.isfinite(result.get('recon_rmse', np.nan))
                    else 'N/A')
        print(f"  [{completed_count:4d}/{total_count}] {dir_name:<50s}  "
              f"RMSE={rmse_str:>6s} dB  ({status})")

    # Serialize data_info dicts (Path objects aren't picklable)
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

    if n_workers <= 1:
        # Sequential
        for i, data_info in enumerate(all_dirs):
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total_dirs - i - 1) / rate / 60 if rate > 0 else 0
                print(f"  --- {elapsed/60:.1f}min elapsed, "
                      f"~{remaining:.1f}min remaining ---")

            result = run_likelihood_experiment(
                data_info=data_info,
                config=config,
                map_data=map_data,
                model_type=args.model_type,
                sigma=args.sigma,
                delta_c=args.delta_c,
                pmf_threshold=args.pmf_threshold,
                fit_exponent=args.fit_exponent,
                cov_type=args.cov_type,
                verbose=args.verbose,
                save_plots=args.save_plots,
                plot_output_dir=output_dir,
                all_tx_locations=all_tx_locations,
            )
            if result is not None:
                all_results.append(result)
            _log_result(result, data_info['name'], i + 1, total_dirs)

            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                _flush_checkpoint(all_results, i + 1, total_dirs)
    else:
        # Parallel
        print(f"  Using {n_workers} parallel workers...")
        task_args = [
            (d_ser, config, map_data, args.model_type, args.sigma, args.delta_c,
             args.pmf_threshold, args.verbose, args.save_plots, str(output_dir),
             all_tx_locations, args.fit_exponent, args.cov_type, None)
            for d_ser in all_dirs_ser
        ]
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_worker_init) as executor:
            futures = {executor.submit(_worker, ta): ta[0]['name']
                       for ta in task_args}
            completed = 0
            for future in as_completed(futures):
                dir_name = futures[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as e:
                    result = None
                    print(f"  Worker exception for {dir_name}: {e}")
                if result is not None:
                    all_results.append(result)
                _log_result(result, dir_name, completed, total_dirs)

                if completed % 10 == 0 or completed <= 5:
                    elapsed = time.time() - start_time
                    print(f"  --- {completed}/{total_dirs} tasks, "
                          f"{elapsed/60:.1f}min elapsed, "
                          f"{len(all_results)} results collected ---")

                if completed % CHECKPOINT_INTERVAL == 0:
                    _flush_checkpoint(all_results, completed, total_dirs)

    # Final checkpoint
    _flush_checkpoint(all_results, total_dirs, total_dirs)

    elapsed_total = time.time() - start_time
    print(f"\nAll experiments completed in {elapsed_total/60:.1f} minutes")
    print(f"Total results: {len(all_results)}")

    if not all_results:
        print("No results collected.")
        return

    # --- Save results ---
    print(f"\n{'='*70}")
    print("Saving results...")
    print(f"{'='*70}")

    raw_df = pd.DataFrame(all_results)
    raw_csv = output_dir / 'likelihood_raw_results.csv'
    raw_df.to_csv(raw_csv, index=False)
    print(f"  Raw results: {raw_csv} ({len(raw_df)} rows)")

    # --- Summary statistics per TX count ---
    success_df = raw_df[raw_df['recon_status'] == 'success'].copy()

    if not success_df.empty:
        summary_rows = []
        for tx_count, group in success_df.groupby('tx_count'):
            row = {
                'tx_count': tx_count,
                'n_dirs': len(group),
            }
            for metric in ['recon_rmse', 'recon_mae', 'recon_bias', 'recon_max_error']:
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

        # Overall summary
        row = {'tx_count': 'all', 'n_dirs': len(success_df)}
        for metric in ['recon_rmse', 'recon_mae', 'recon_bias', 'recon_max_error']:
            vals = success_df[metric].dropna()
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
        summary_csv = output_dir / 'likelihood_summary.csv'
        summary_df.to_csv(summary_csv, index=False)
        print(f"  Summary: {summary_csv}")

        # Print summary table
        print(f"\n{'='*70}")
        print("LIKELIHOOD RECONSTRUCTION SUMMARY")
        print(f"  Model: {args.model_type}  |  sigma={args.sigma}  |  delta_c={args.delta_c}")
        print(f"{'='*70}")
        print(f"  {'TX Count':<12s} {'N dirs':<8s} {'RMSE (dB)':<14s} {'MAE (dB)':<14s} {'Bias (dB)':<14s}")
        print(f"  {'-'*12} {'-'*8} {'-'*14} {'-'*14} {'-'*14}")
        for _, row in summary_df.iterrows():
            tc = str(row['tx_count'])
            n = int(row['n_dirs']) if row['tx_count'] != 'all' else int(row['n_dirs'])
            rmse = f"{row['recon_rmse_mean']:.1f}+/-{row['recon_rmse_std']:.1f}" if np.isfinite(row['recon_rmse_mean']) else 'N/A'
            mae = f"{row['recon_mae_mean']:.1f}+/-{row['recon_mae_std']:.1f}" if np.isfinite(row['recon_mae_mean']) else 'N/A'
            bias = f"{row['recon_bias_mean']:.1f}+/-{row['recon_bias_std']:.1f}" if np.isfinite(row['recon_bias_mean']) else 'N/A'
            print(f"  {tc:<12s} {n:<8d} {rmse:<14s} {mae:<14s} {bias:<14s}")

    # Status breakdown
    status_counts = raw_df['recon_status'].value_counts()
    print(f"\n  Status breakdown:")
    for status, count in status_counts.items():
        print(f"    {status}: {count}")

    # --- Generate plots ---
    print(f"\n{'='*70}")
    print("Generating plots...")
    print(f"{'='*70}")
    try:
        generate_likelihood_plots(raw_df, output_dir, args.model_type)
    except Exception as e:
        print(f"  Plot generation failed: {e}")

    print(f"\n{'='*70}")
    print(f"Likelihood reconstruction complete!")
    print(f"  Results: {output_dir}")
    print(f"  Runtime: {elapsed_total/60:.1f} minutes")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_likelihood_plots(raw_df: pd.DataFrame, output_dir: Path,
                              model_type: str):
    """
    Generate publication-quality bar plots summarizing likelihood reconstruction
    performance, grouped by TX count.

    Produces:
      1. Reconstruction metrics by TX count (RMSE, MAE, Bias, Max Error)
      2. Box plots of RMSE and MAE distributions by TX count
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    success_df = raw_df[raw_df['recon_status'] == 'success'].copy()
    if success_df.empty:
        print("  No successful results to plot.")
        return

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Publication style (matching ablation_plotting.py)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'errorbar.capsize': 3,
    })

    COLORS = {
        'blue':   '#0072B2',
        'orange': '#E69F00',
        'green':  '#009E73',
        'red':    '#D55E00',
        'purple': '#CC79A7',
        'cyan':   '#56B4E9',
    }
    color_list = list(COLORS.values())

    tx_counts = sorted(success_df['tx_count'].unique())

    def _save(fig, name):
        fig.savefig(plots_dir / f'{name}.pdf', format='pdf')
        fig.savefig(plots_dir / f'{name}.png', format='png')
        plt.close(fig)

    # --- Figure 1: Bar chart of reconstruction metrics by TX count ---
    metrics = [
        ('recon_rmse', 'RMSE (dB)'),
        ('recon_mae', 'MAE (dB)'),
        ('recon_bias', 'Bias (dB)'),
        ('recon_max_error', 'Max Error (dB)'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(7.5, 3.0))
    x = np.arange(len(tx_counts))
    bar_width = 0.6

    for ax_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]
        means = []
        sems = []
        for tc in tx_counts:
            vals = success_df[success_df['tx_count'] == tc][metric_col].dropna()
            means.append(vals.mean() if len(vals) > 0 else np.nan)
            sems.append(vals.sem() if len(vals) > 1 else 0)

        bars = ax.bar(x, means, bar_width, yerr=sems,
                      color=[color_list[i % len(color_list)] for i in range(len(tx_counts))],
                      capsize=3, edgecolor='white', linewidth=0.5)

        # Value labels on bars
        for i, (m, s) in enumerate(zip(means, sems)):
            if np.isfinite(m):
                ax.text(i, m + s + 0.3, f'{m:.1f}', ha='center', va='bottom',
                        fontsize=7, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([str(tc) for tc in tx_counts])
        ax.set_xlabel('TX Count')
        ax.set_ylabel(metric_label)

    fig.suptitle(f'Likelihood Reconstruction Performance by TX Count\n'
                 f'Model: {model_type}',
                 fontsize=12, fontweight='bold', y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'likelihood_metrics_by_tx_count')

    # --- Figure 2: Box plots of RMSE and MAE by TX count ---
    box_metrics = [
        ('recon_rmse', 'RMSE (dB)'),
        ('recon_mae', 'MAE (dB)'),
        ('recon_bias', 'Bias (dB)'),
    ]

    fig, axes = plt.subplots(1, len(box_metrics), figsize=(12, 5))

    for ax_idx, (metric_col, metric_label) in enumerate(box_metrics):
        ax = axes[ax_idx]
        data_by_tc = []
        labels = []
        for tc in tx_counts:
            vals = success_df[success_df['tx_count'] == tc][metric_col].dropna()
            if len(vals) > 0:
                data_by_tc.append(vals.values)
                labels.append(str(tc))

        if data_by_tc:
            bp = ax.boxplot(data_by_tc, labels=labels, patch_artist=True,
                            widths=0.5, showfliers=True,
                            flierprops=dict(marker='o', markersize=3, alpha=0.5))
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(color_list[i % len(color_list)])
                patch.set_alpha(0.7)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(1.5)

        ax.set_xlabel('TX Count')
        ax.set_ylabel(metric_label)

    fig.suptitle(f'Likelihood Reconstruction Distribution by TX Count\n'
                 f'Model: {model_type}  |  N={len(success_df)} directories',
                 fontsize=20, fontweight='normal')
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    _save(fig, 'likelihood_boxplots_by_tx_count')

    # --- Figure 3: Per-directory RMSE scatter (sorted) ---
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    sorted_df = success_df.sort_values('recon_rmse').reset_index(drop=True)
    colors = [color_list[int(tc) - 1] if int(tc) <= len(color_list)
              else color_list[-1] for tc in sorted_df['tx_count']]
    ax.bar(range(len(sorted_df)), sorted_df['recon_rmse'],
           color=colors, edgecolor='none', width=1.0)

    # Add legend for TX counts
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_list[i], label=f'TX={tc}')
                       for i, tc in enumerate(tx_counts)
                       if i < len(color_list)]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

    # Horizontal reference lines
    overall_mean = sorted_df['recon_rmse'].mean()
    overall_median = sorted_df['recon_rmse'].median()
    ax.axhline(overall_mean, color='black', linestyle='--', linewidth=0.8,
               label=f'Mean={overall_mean:.1f}')
    ax.axhline(overall_median, color='gray', linestyle=':', linewidth=0.8,
               label=f'Median={overall_median:.1f}')
    ax.text(len(sorted_df) * 0.98, overall_mean + 0.3, f'Mean={overall_mean:.1f}',
            ha='right', va='bottom', fontsize=8)
    ax.text(len(sorted_df) * 0.98, overall_median - 0.3, f'Median={overall_median:.1f}',
            ha='right', va='top', fontsize=8, color='gray')

    ax.set_xlabel('Directory (sorted by RMSE)')
    ax.set_ylabel('Reconstruction RMSE (dB)')
    ax.set_title(f'Per-Directory Reconstruction RMSE  |  Model: {model_type}',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, len(sorted_df) - 0.5)
    fig.tight_layout()
    _save(fig, 'likelihood_rmse_per_directory')

    # --- Figure 4: Aggregate box plot across all directories ---
    agg_metrics = [
        ('recon_rmse', 'RMSE (dB)'),
        ('recon_mae', 'MAE (dB)'),
        ('recon_bias', 'Bias (dB)'),
        ('recon_max_error', 'Max Error (dB)'),
    ]
    agg_colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red']]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    data_list = []
    labels = []
    for metric_col, metric_label in agg_metrics:
        vals = success_df[metric_col].dropna().values
        data_list.append(vals)
        labels.append(metric_label)

    bp = ax.boxplot(data_list, labels=labels, patch_artist=True,
                    widths=0.5, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(agg_colors[i])
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
    ax.tick_params(axis='y', which='minor', length=2)
    ax.set_ylabel('Value (dB)')
    ax.set_title(f'Aggregate Reconstruction Performance\n'
                 f'Model: {model_type}  |  N={len(success_df)} directories',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'likelihood_aggregate_performance')

    print(f"  Plots saved to: {plots_dir}")


if __name__ == '__main__':
    main()
