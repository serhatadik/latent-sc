"""Publication-quality plots for ablation study results.

Generates figures suitable for direct insertion into academic papers:
- Serif fonts (Computer Modern / Times-like), 300 DPI
- Color-blind friendly palette (Okabe-Ito)
- PDF + PNG dual output
- Proper axis labels, legends, significance markers
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Ablation Factor Definitions (shared with run_ablation_study.py)
# ---------------------------------------------------------------------------
ABLATION_FACTORS = {
    'whitening': {
        'display_name': 'Whitening Method',
        'type': 'comparison',
        'variants': [
            {'name': 'Homoscedastic',    'whitening_method': 'spatial_corr_exp_decay', 'whitening_config_name': 'homoscedastic'},
            {'name': 'Hetero-Diagonal',  'whitening_method': 'hetero_diag',            'whitening_config_name': 'hetero_diag'},
            {'name': 'Hetero-Spatial',   'whitening_method': 'hetero_spatial',          'whitening_config_name': 'hetero_spatial'},
            {'name': 'Hetero-Geo-Aware', 'whitening_method': 'hetero_geo_aware',        'whitening_config_name': 'hetero_geo_aware',
             'feature_rho': [0.5, 10.0, 1e6, 150.0]},
        ],
        'baseline_variant': 'Hetero-Spatial',
    },
    'beam_search': {
        'display_name': 'Beam Search',
        'type': 'binary',
        'variants': [
            {'name': 'Greedy (B=1)', 'beam_width': 1},
            {'name': 'Beam (B=3)',   'beam_width': 3},
        ],
        'baseline_variant': 'Beam (B=3)',
    },
    'pool_refinement': {
        'display_name': 'Pool Refinement & Dedup',
        'type': 'binary',
        'variants': [
            {'name': 'Disabled', 'pool_refinement': False, 'dedupe_distance_m': 0},
            {'name': 'Enabled',  'pool_refinement': True,  'dedupe_distance_m': 60.0},
        ],
        'baseline_variant': 'Enabled',
    },
    'physics_filters': {
        'display_name': 'Physics-Based Filters',
        'type': 'binary',
        'variants': [
            {'name': 'Disabled', 'max_tx_power_dbm': 999.0, 'veto_margin_db': 999.0, 'ceiling_penalty_weight': 0.0},
            {'name': 'Enabled',  'max_tx_power_dbm': 40.0,  'veto_margin_db': 5.0,   'ceiling_penalty_weight': 0.1},
        ],
        'baseline_variant': 'Enabled',
    },
    'hard_filtering': {
        'display_name': 'Hard Candidate Filtering',
        'type': 'binary',
        'variants': [
            {'name': 'Disabled', 'rmse_threshold': 9999.0, 'max_error_threshold': 9999.0},
            {'name': 'Enabled',  'rmse_threshold': 20.0,   'max_error_threshold': 38.0},
        ],
        'baseline_variant': 'Enabled',
    },
    'bic_selection': {
        'display_name': 'BIC TX Selection',
        'type': 'binary',
        'variants': [
            {'name': 'Disabled (Raw GLRT)', 'skip_bic_selection': True},
            {'name': 'Enabled (BIC)',       'skip_bic_selection': False},
        ],
        'baseline_variant': 'Enabled (BIC)',
    },
    'localization_model': {
        'display_name': 'Localization Model',
        'type': 'comparison',
        'variants': [
            {'name': 'Log-Distance', 'model_type': 'log_distance'},
            {'name': 'TIREM',        'model_type': 'tirem'},
            {'name': 'Ray Tracing',  'model_type': 'raytracing'},
        ],
        'baseline_variant': 'TIREM',
    },
    'reconstruction_model': {
        'display_name': 'Reconstruction Model',
        'type': 'comparison',
        'variants': [
            {'name': 'Log-Distance', 'recon_model_type': 'log_distance'},
            {'name': 'TIREM',        'recon_model_type': 'tirem'},
            {'name': 'Ray Tracing',  'recon_model_type': 'raytracing'},
        ],
        'baseline_variant': 'TIREM',
    },
}


# ---------------------------------------------------------------------------
# Color-blind friendly palette (Okabe-Ito)
# ---------------------------------------------------------------------------
COLORS = {
    'blue':    '#0072B2',
    'orange':  '#E69F00',
    'green':   '#009E73',
    'red':     '#D55E00',
    'purple':  '#CC79A7',
    'cyan':    '#56B4E9',
    'yellow':  '#F0E442',
    'black':   '#000000',
}
COLOR_LIST = list(COLORS.values())

# Paired colors for enabled/disabled
ENABLED_COLOR  = COLORS['blue']
DISABLED_COLOR = COLORS['orange']


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
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
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'errorbar.capsize': 3,
    })


def _save_figure(fig, output_dir: Path, name: str):
    """Save figure as both PDF and PNG."""
    fig.savefig(output_dir / f'{name}.pdf', format='pdf')
    fig.savefig(output_dir / f'{name}.png', format='png')
    plt.close(fig)


def _significance_marker(p_value: float) -> str:
    """Return significance marker string for a p-value."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return 'n.s.'


def _paired_wilcoxon(baseline_values: np.ndarray, variant_values: np.ndarray) -> Tuple[float, str]:
    """Compute paired Wilcoxon signed-rank test, return (p_value, marker)."""
    # Need at least 6 paired observations
    mask = np.isfinite(baseline_values) & np.isfinite(variant_values)
    b = baseline_values[mask]
    v = variant_values[mask]
    if len(b) < 6:
        return 1.0, 'n.s.'
    diffs = b - v
    if np.all(diffs == 0):
        return 1.0, 'n.s.'
    try:
        stat, p = stats.wilcoxon(b, v, alternative='two-sided')
        return p, _significance_marker(p)
    except Exception:
        return 1.0, 'n.s.'


def _add_significance_bracket(ax, x1, x2, y, marker, height_frac=0.02):
    """Draw a significance bracket between two bar positions."""
    if marker == 'n.s.':
        return
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * height_frac
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c='k')
    ax.text((x1 + x2) / 2, y + h, marker, ha='center', va='bottom', fontsize=8)


# ---------------------------------------------------------------------------
# Plot 1: Binary Ablation Summary
# ---------------------------------------------------------------------------
def plot_binary_ablation_summary(
    df: pd.DataFrame,
    binary_factors: Dict[str, dict],
    output_dir: Path,
):
    """
    Grouped bar chart for ON/OFF ablation factors.

    Parameters
    ----------
    df : DataFrame
        Ablation results with columns: factor, variant, dir_name,
        combo_ale, combo_pd, combo_precision, recon_rmse, recon_mae, recon_max_error
    binary_factors : dict
        Factor definitions (keys = factor names, values = factor config dicts)
    output_dir : Path
        Output directory for figures
    """
    setup_publication_style()

    # Metrics to plot
    loc_metrics = [
        ('combo_ale', 'ALE (m)', True),       # lower is better
        ('combo_pd', 'Pd', False),             # higher is better
        ('combo_precision', 'Precision', False),
    ]
    recon_metrics = [
        ('recon_rmse', 'Recon. RMSE (dB)', True),
        ('recon_mae', 'Recon. MAE (dB)', True),
        ('recon_max_error', 'Recon. Max Error (dB)', True),
    ]
    all_metrics = loc_metrics + recon_metrics

    factor_names = [f for f in binary_factors if f in df['factor'].unique()]
    if not factor_names:
        return

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.5))
    axes = axes.flatten()

    bar_width = 0.35
    x = np.arange(len(factor_names))

    for ax_idx, (metric_col, metric_label, lower_better) in enumerate(all_metrics):
        ax = axes[ax_idx]

        # Collect per-factor stats, then filter to factors where both
        # enabled and disabled have finite means for this metric.
        per_factor = []  # (factor_name, en_mean, en_sem, dis_mean, dis_sem)
        for factor_name in factor_names:
            finfo = binary_factors[factor_name]
            baseline_name = finfo['baseline_variant']
            other_names = [v['name'] for v in finfo['variants'] if v['name'] != baseline_name]
            ablated_name = other_names[0] if other_names else baseline_name

            fdf = df[df['factor'] == factor_name]

            baseline_vals = fdf[fdf['variant'] == baseline_name][metric_col].dropna()
            ablated_vals = fdf[fdf['variant'] == ablated_name][metric_col].dropna()

            en_mean = baseline_vals.mean() if len(baseline_vals) > 0 else np.nan
            en_sem = baseline_vals.sem() if len(baseline_vals) > 1 else 0
            dis_mean = ablated_vals.mean() if len(ablated_vals) > 0 else np.nan
            dis_sem = ablated_vals.sem() if len(ablated_vals) > 1 else 0

            # Skip factors where either variant has N/A for this metric
            if not np.isfinite(en_mean) or not np.isfinite(dis_mean):
                continue
            per_factor.append((factor_name, en_mean, en_sem, dis_mean, dis_sem))

        if not per_factor:
            ax.set_visible(False)
            continue

        valid_fnames = [pf[0] for pf in per_factor]
        enabled_means = [pf[1] for pf in per_factor]
        enabled_sems = [pf[2] for pf in per_factor]
        disabled_means = [pf[3] for pf in per_factor]
        disabled_sems = [pf[4] for pf in per_factor]

        x_valid = np.arange(len(valid_fnames))

        ax.bar(x_valid - bar_width / 2, enabled_means, bar_width, yerr=enabled_sems,
               color=ENABLED_COLOR, label='Enabled', capsize=3, edgecolor='white', linewidth=0.5)
        ax.bar(x_valid + bar_width / 2, disabled_means, bar_width, yerr=disabled_sems,
               color=DISABLED_COLOR, label='Disabled', capsize=3, edgecolor='white', linewidth=0.5)

        # Add significance markers
        for i, factor_name in enumerate(valid_fnames):
            finfo = binary_factors[factor_name]
            baseline_name = finfo['baseline_variant']
            other_names = [v['name'] for v in finfo['variants'] if v['name'] != baseline_name]
            ablated_name = other_names[0] if other_names else baseline_name

            fdf = df[df['factor'] == factor_name]
            # Get paired values per directory
            baseline_per_dir = fdf[fdf['variant'] == baseline_name].set_index('dir_name')[metric_col]
            ablated_per_dir = fdf[fdf['variant'] == ablated_name].set_index('dir_name')[metric_col]
            common_dirs = baseline_per_dir.index.intersection(ablated_per_dir.index)
            if len(common_dirs) >= 6:
                _, marker = _paired_wilcoxon(
                    baseline_per_dir.loc[common_dirs].values,
                    ablated_per_dir.loc[common_dirs].values,
                )
                y_max = max(enabled_means[i] + enabled_sems[i],
                           disabled_means[i] + disabled_sems[i])
                if np.isfinite(y_max) and marker != 'n.s.':
                    _add_significance_bracket(ax, i - bar_width / 2, i + bar_width / 2, y_max * 1.02, marker)

        display_names = [binary_factors[f]['display_name'] for f in valid_fnames]
        # Shorten long names
        short_names = []
        for dn in display_names:
            dn = dn.replace('Pool Refinement & Dedup', 'Pool Refine.')
            dn = dn.replace('Physics-Based Filters', 'Physics Filt.')
            dn = dn.replace('Hard Candidate Filtering', 'Hard Filt.')
            dn = dn.replace('BIC TX Selection', 'BIC Select.')
            dn = dn.replace('Beam Search', 'Beam Search')
            short_names.append(dn)

        ax.set_xticks(x_valid)
        ax.set_xticklabels(short_names, rotation=25, ha='right')
        ax.set_ylabel(metric_label)

        if ax_idx == 0:
            ax.legend(loc='best', framealpha=0.9)

    fig.suptitle('Ablation Study: Component Contribution', fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_dir, 'ablation_binary_summary')


# ---------------------------------------------------------------------------
# Plot 2: Comparison Variants
# ---------------------------------------------------------------------------
def plot_comparison_variants(
    df: pd.DataFrame,
    factor_name: str,
    factor_config: dict,
    output_dir: Path,
):
    """
    Multi-variant comparison bar chart for a single factor.

    Parameters
    ----------
    df : DataFrame
        Ablation results filtered to this factor
    factor_name : str
        Factor key name
    factor_config : dict
        Factor configuration with 'variants' and 'baseline_variant'
    output_dir : Path
        Output directory
    """
    setup_publication_style()

    fdf = df[df['factor'] == factor_name].copy()
    if fdf.empty:
        return

    metrics = [
        ('combo_ale', 'ALE (m)'),
        ('combo_pd', 'Pd'),
        ('recon_rmse', 'Recon. RMSE (dB)'),
        ('recon_mae', 'Recon. MAE (dB)'),
    ]

    variant_names = [v['name'] for v in factor_config['variants']]
    baseline_name = factor_config.get('baseline_variant', '')
    n_variants = len(variant_names)

    fig, axes = plt.subplots(1, len(metrics), figsize=(7.0, 2.5))

    for ax_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]

        # Collect stats per variant, filtering out those with N/A means
        valid_means = []
        valid_sems = []
        valid_colors = []
        valid_names = []

        for i, vname in enumerate(variant_names):
            vdf = fdf[fdf['variant'] == vname][metric_col].dropna()
            m = vdf.mean() if len(vdf) > 0 else np.nan
            if not np.isfinite(m):
                continue
            valid_means.append(m)
            valid_sems.append(vdf.sem() if len(vdf) > 1 else 0)
            valid_colors.append(COLOR_LIST[i % len(COLOR_LIST)])
            valid_names.append(vname)

        if not valid_means:
            ax.set_visible(False)
            continue

        x = np.arange(len(valid_names))
        bars = ax.bar(x, valid_means, yerr=valid_sems, color=valid_colors, capsize=3,
                      edgecolor='white', linewidth=0.5)

        # Highlight baseline with hatching
        for i, vname in enumerate(valid_names):
            if vname == baseline_name:
                bars[i].set_hatch('///')
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(valid_names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(metric_label)

    display_name = factor_config.get('display_name', factor_name)
    fig.suptitle(f'{display_name} Comparison', fontsize=12, fontweight='bold', y=1.02)

    # Add legend for baseline indicator
    legend_elements = [Patch(facecolor='gray', edgecolor='black', hatch='///', label='Baseline')]
    axes[-1].legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.tight_layout()
    _save_figure(fig, output_dir, f'ablation_{factor_name}')


# ---------------------------------------------------------------------------
# Plot 3: Ablation by TX Count
# ---------------------------------------------------------------------------
def plot_ablation_by_tx_count(
    df: pd.DataFrame,
    factors: Dict[str, dict],
    output_dir: Path,
):
    """
    Heatmap showing performance change (delta %) for each factor x TX count.

    Parameters
    ----------
    df : DataFrame
        Ablation results with factor, variant, tx_count, combo_ale, recon_rmse
    factors : dict
        All factor configs (binary ones used for delta computation)
    output_dir : Path
        Output directory
    """
    setup_publication_style()

    # Only binary factors with clear enabled/disabled
    binary_factors = {k: v for k, v in factors.items() if v.get('type') == 'binary'}
    if not binary_factors:
        return

    tx_counts = sorted(df['tx_count'].dropna().unique())
    if not tx_counts:
        return
    tx_counts = [int(tc) for tc in tx_counts]
    factor_labels = []

    ale_deltas = []
    rmse_deltas = []

    for factor_name, finfo in binary_factors.items():
        baseline_name = finfo['baseline_variant']
        other_names = [v['name'] for v in finfo['variants'] if v['name'] != baseline_name]
        ablated_name = other_names[0] if other_names else baseline_name

        factor_labels.append(finfo['display_name'])

        ale_row = []
        rmse_row = []

        for tc in tx_counts:
            fdf = df[(df['factor'] == factor_name) & (df['tx_count'] == tc)]
            base_ale = fdf[fdf['variant'] == baseline_name]['combo_ale'].mean()
            abl_ale = fdf[fdf['variant'] == ablated_name]['combo_ale'].mean()
            base_rmse = fdf[fdf['variant'] == baseline_name]['recon_rmse'].mean()
            abl_rmse = fdf[fdf['variant'] == ablated_name]['recon_rmse'].mean()

            # Compute percentage change: positive = enabled is better (lower ALE/RMSE)
            if np.isfinite(base_ale) and np.isfinite(abl_ale) and base_ale > 0:
                ale_row.append((abl_ale - base_ale) / base_ale * 100)
            else:
                ale_row.append(np.nan)

            if np.isfinite(base_rmse) and np.isfinite(abl_rmse) and base_rmse > 0:
                rmse_row.append((abl_rmse - base_rmse) / base_rmse * 100)
            else:
                rmse_row.append(np.nan)

        ale_deltas.append(ale_row)
        rmse_deltas.append(rmse_row)

    ale_data = np.array(ale_deltas)
    rmse_data = np.array(rmse_deltas)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, max(3.0, 0.6 * len(factor_labels) + 1.5)))

    vmax = max(np.nanmax(np.abs(ale_data)) if ale_data.size else 10,
               np.nanmax(np.abs(rmse_data)) if rmse_data.size else 10)
    vmax = min(vmax, 100)  # Cap at 100%

    for ax, data, title in [(ax1, ale_data, r'$\Delta$ ALE (%)'),
                            (ax2, rmse_data, r'$\Delta$ Recon. RMSE (%)')]:
        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto',
                       vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(tx_counts)))
        ax.set_xticklabels([str(tc) for tc in tx_counts])
        ax.set_xlabel('TX Count')
        ax.set_yticks(range(len(factor_labels)))
        ax.set_yticklabels(factor_labels, fontsize=8)
        ax.set_title(title)

        # Annotate cells
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    color = 'white' if abs(val) > vmax * 0.6 else 'black'
                    ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                            fontsize=7, color=color)

    fig.colorbar(im, ax=[ax1, ax2], label=r'$\Delta$ when disabled (%)', shrink=0.8)
    fig.suptitle('Performance Impact by TX Count', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    _save_figure(fig, output_dir, 'ablation_by_tx_count')


# ---------------------------------------------------------------------------
# Plot 4: Propagation Model Cross-Matrix
# ---------------------------------------------------------------------------
def plot_propagation_model_matrix(
    df: pd.DataFrame,
    output_dir: Path,
):
    """
    3x3 heatmaps for localization x reconstruction model combinations.

    Parameters
    ----------
    df : DataFrame
        Cross-model results with loc_model, recon_model columns
    output_dir : Path
        Output directory
    """
    setup_publication_style()

    if 'loc_model' not in df.columns or 'recon_model' not in df.columns:
        return

    model_order = ['log_distance', 'tirem', 'raytracing']
    model_labels = {'log_distance': 'Log-Dist.', 'tirem': 'TIREM', 'raytracing': 'Ray Tracing'}

    # Filter to available models
    avail_loc = [m for m in model_order if m in df['loc_model'].unique()]
    avail_recon = [m for m in model_order if m in df['recon_model'].unique()]

    if not avail_loc or not avail_recon:
        return

    metrics = [
        ('combo_ale', 'ALE (m)'),
        ('combo_pd', 'Pd'),
        ('recon_rmse', 'Recon. RMSE (dB)'),
        ('recon_mae', 'Recon. MAE (dB)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    axes = axes.flatten()

    for ax_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]

        matrix = np.full((len(avail_recon), len(avail_loc)), np.nan)
        std_matrix = np.full((len(avail_recon), len(avail_loc)), np.nan)

        for i, rm in enumerate(avail_recon):
            for j, lm in enumerate(avail_loc):
                cell_df = df[(df['loc_model'] == lm) & (df['recon_model'] == rm)]
                vals = cell_df[metric_col].dropna()
                if len(vals) > 0:
                    matrix[i, j] = vals.mean()
                    std_matrix[i, j] = vals.std()

        # Choose colormap
        if metric_col in ('combo_pd',):
            cmap = 'RdYlGn'  # higher is better → green
        else:
            cmap = 'RdYlGn_r'  # lower is better → green

        im = ax.imshow(matrix, cmap=cmap, aspect='auto')

        ax.set_xticks(range(len(avail_loc)))
        ax.set_xticklabels([model_labels.get(m, m) for m in avail_loc])
        ax.set_yticks(range(len(avail_recon)))
        ax.set_yticklabels([model_labels.get(m, m) for m in avail_recon])
        ax.set_xlabel('Localization Model')
        ax.set_ylabel('Reconstruction Model')
        ax.set_title(metric_label)

        # Annotate cells with mean ± std
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                std_val = std_matrix[i, j]
                if np.isfinite(val):
                    txt = f'{val:.1f}'
                    if np.isfinite(std_val) and std_val > 0:
                        txt += f'\n$\\pm${std_val:.1f}'
                    ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                            fontweight='bold')

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Propagation Model Cross-Comparison', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, output_dir, 'ablation_prop_model_matrix')


# ---------------------------------------------------------------------------
# Plot 5: Cumulative Pipeline Build-up
# ---------------------------------------------------------------------------
def plot_cumulative_pipeline(
    df: pd.DataFrame,
    output_dir: Path,
):
    """
    Bottom-up cumulative pipeline: start minimal, progressively add components.

    Parameters
    ----------
    df : DataFrame
        Must contain results for the 'cumulative' factor with stages as variants
    output_dir : Path
        Output directory
    """
    setup_publication_style()

    cdf = df[df['factor'] == 'cumulative'].copy()
    if cdf.empty:
        return

    # Define the stage order (bottom-up)
    stage_order = [
        'Raw GLRT\n(Greedy)',
        '+Physics\nFilters',
        '+Hard\nFiltering',
        '+BIC\nSelection',
        '+Pool\nRefinement',
        '+Beam\nSearch',
    ]

    # Map variant names to stage order
    available_stages = cdf['variant'].unique()
    stages = [s for s in stage_order if s.replace('\n', ' ') in available_stages
              or s in available_stages]

    if not stages:
        # Try matching without newlines
        stage_map = {}
        for s in stage_order:
            plain = s.replace('\n', ' ')
            for av in available_stages:
                if plain == av or s == av:
                    stage_map[s] = av
                    break
        stages = [s for s in stage_order if s in stage_map]
        if not stages:
            return

    metrics = [('combo_ale', 'ALE (m)'), ('recon_rmse', 'Recon. RMSE (dB)')]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for ax_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]

        all_means = []
        all_stds = []

        for stage in stages:
            # Match variant
            plain = stage.replace('\n', ' ')
            vdf = cdf[(cdf['variant'] == plain) | (cdf['variant'] == stage)]
            vals = vdf[metric_col].dropna()
            all_means.append(vals.mean() if len(vals) > 0 else np.nan)
            all_stds.append(vals.std() if len(vals) > 1 else 0)

        # Filter to stages with finite means
        valid = [i for i, m in enumerate(all_means) if np.isfinite(m)]
        if not valid:
            ax.set_visible(False)
            continue

        valid_stages = [stages[i] for i in valid]
        means = np.array([all_means[i] for i in valid])
        stds_arr = np.array([all_stds[i] for i in valid])
        x = np.arange(len(valid_stages))

        ax.plot(x, means, 'o-', color=COLORS['blue'], linewidth=2, markersize=8, zorder=5)
        ax.fill_between(x, means - stds_arr, means + stds_arr,
                        alpha=0.2, color=COLORS['blue'])

        ax.set_xticks(x)
        ax.set_xticklabels(valid_stages, fontsize=7)
        ax.set_ylabel(metric_label)
        ax.set_xlabel('Pipeline Stage')

    fig.suptitle('Cumulative Pipeline Build-up', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_figure(fig, output_dir, 'ablation_cumulative')


# ---------------------------------------------------------------------------
# Plot 6: Metric Distributions (Box/Violin)
# ---------------------------------------------------------------------------
def plot_metric_distributions(
    df: pd.DataFrame,
    factor_name: str,
    factor_config: dict,
    output_dir: Path,
):
    """
    Box plots showing per-directory metric distributions for each variant.

    Parameters
    ----------
    df : DataFrame
        Ablation results
    factor_name : str
        Factor key
    factor_config : dict
        Factor config with variants
    output_dir : Path
        Output directory
    """
    setup_publication_style()

    fdf = df[df['factor'] == factor_name].copy()
    if fdf.empty:
        return

    metrics = [
        ('combo_ale', 'ALE (m)'),
        ('combo_pd', 'Pd'),
        ('recon_rmse', 'Recon. RMSE (dB)'),
        ('recon_mae', 'Recon. MAE (dB)'),
    ]

    variant_names = [v['name'] for v in factor_config['variants']]
    n_variants = len(variant_names)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    axes = axes.flatten()

    for ax_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]

        data_list = []
        labels = []

        for vname in variant_names:
            vals = fdf[fdf['variant'] == vname][metric_col].dropna().values
            if len(vals) > 0:
                data_list.append(vals)
                labels.append(vname)

        if not data_list:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data_list, labels=labels, patch_artist=True,
                        widths=0.6, showfliers=True, flierprops={'markersize': 3})

        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(COLOR_LIST[i % len(COLOR_LIST)])
            patch.set_alpha(0.7)

        ax.set_ylabel(metric_label)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)

    display_name = factor_config.get('display_name', factor_name)
    fig.suptitle(f'{display_name}: Per-Dataset Distributions', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, output_dir, f'ablation_distributions_{factor_name}')


# ---------------------------------------------------------------------------
# Master function: generate all plots
# ---------------------------------------------------------------------------
def generate_all_ablation_plots(
    results_df: pd.DataFrame,
    factors: Dict[str, dict],
    output_dir: Path,
    cross_model_df: Optional[pd.DataFrame] = None,
    cumulative_df: Optional[pd.DataFrame] = None,
):
    """
    Generate all ablation study plots.

    Parameters
    ----------
    results_df : DataFrame
        Main ablation results (columns: factor, variant, dir_name, tx_count, + metrics)
    factors : dict
        Factor definitions
    output_dir : Path
        Output directory for figures
    cross_model_df : DataFrame, optional
        Cross-model matrix results (columns: loc_model, recon_model, + metrics)
    cumulative_df : DataFrame, optional
        Cumulative pipeline results (columns: factor='cumulative', variant=stage_name)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating ablation study plots in {output_dir} ...")

    # Plot 1: Binary ablation summary
    binary_factors = {k: v for k, v in factors.items()
                      if v.get('type') == 'binary' and k in results_df['factor'].unique()}
    if binary_factors:
        print("  [1/6] Binary ablation summary...")
        plot_binary_ablation_summary(results_df, binary_factors, output_dir)

    # Plot 2: Comparison variants (one per comparison-type factor)
    comparison_factors = {k: v for k, v in factors.items()
                          if v.get('type') == 'comparison' and k in results_df['factor'].unique()}
    for i, (fname, fconfig) in enumerate(comparison_factors.items()):
        print(f"  [2/6] Comparison variants: {fname}...")
        plot_comparison_variants(results_df, fname, fconfig, output_dir)

    # Plot 3: Ablation by TX count
    if binary_factors:
        print("  [3/6] Ablation by TX count heatmap...")
        plot_ablation_by_tx_count(results_df, factors, output_dir)

    # Plot 4: Propagation model cross-matrix
    if cross_model_df is not None and not cross_model_df.empty:
        print("  [4/6] Propagation model cross-matrix...")
        plot_propagation_model_matrix(cross_model_df, output_dir)

    # Plot 5: Cumulative pipeline build-up
    cum_df = cumulative_df if cumulative_df is not None else results_df
    if 'factor' in cum_df.columns and 'cumulative' in cum_df['factor'].unique():
        print("  [5/6] Cumulative pipeline build-up...")
        plot_cumulative_pipeline(cum_df, output_dir)

    # Plot 6: Per-factor distributions
    for fname, fconfig in factors.items():
        if fname in results_df['factor'].unique():
            print(f"  [6/6] Distributions: {fname}...")
            plot_metric_distributions(results_df, fname, fconfig, output_dir)

    print(f"  All plots saved to {output_dir}")
