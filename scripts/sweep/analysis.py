"""Analysis and aggregation functions (pure pandas/numpy, no internal sweep imports)."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def analyze_by_tx_count(results_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Generate analysis summary for each TX count.

    Returns
    -------
    dict
        Dictionary mapping TX count -> summary DataFrame
    """
    summaries = {}

    for tx_count in sorted(results_df['tx_count'].unique()):
        df = results_df[results_df['tx_count'] == tx_count]

        # Group by strategy, selection method, power filtering, threshold, and whitening_config
        grouped = df.groupby(['strategy', 'selection_method', 'power_filtering', 'power_threshold', 'whitening_config'], dropna=False).agg({
            'ale': ['mean', 'std', 'min', 'max', 'count'],
            'pd': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'n_estimated': ['mean', 'std'],
            'n_est_raw': ['mean'],
            'n_est_diff': ['mean'],
            'count_error': ['mean', 'std', 'min', 'max'],
            'tp': 'sum',
            'fp': 'sum',
            'fn': 'sum',
            'runtime_s': 'mean',
            'glrt_final_score': ['mean', 'std', 'min', 'max'],
            'glrt_n_iterations': ['mean', 'std'],
            'glrt_score_reduction': ['mean', 'std'],
        }).reset_index()

        # Flatten column names
        grouped.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col
            for col in grouped.columns
        ]

        # Sort by mean ALE
        grouped = grouped.sort_values('ale_mean')

        summaries[tx_count] = grouped

    return summaries


def analyze_universal(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate universal analysis across all TX counts.

    Returns
    -------
    pd.DataFrame
        Summary dataframe
    """
    # Group by strategy, selection method, power filtering, threshold, and whitening_config
    grouped = results_df.groupby(['strategy', 'selection_method', 'power_filtering', 'power_threshold', 'whitening_config'], dropna=False).agg({
        'ale': ['mean', 'std', 'min', 'max', 'count'],
        'pd': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'n_estimated': ['mean', 'std'],
        'n_est_raw': ['mean'],
        'n_est_diff': ['mean'],
        'count_error': ['mean', 'std', 'min', 'max'],
        'tp': 'sum',
        'fp': 'sum',
        'fn': 'sum',
        'runtime_s': 'mean',
        'glrt_final_score': ['mean', 'std', 'min', 'max'],
        'glrt_n_iterations': ['mean', 'std'],
        'glrt_score_reduction': ['mean', 'std'],
    }).reset_index()

    # Flatten column names
    grouped.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    # Sort by mean ALE
    grouped = grouped.sort_values('ale_mean')

    return grouped


def analyze_by_tx_set(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance grouped by specific transmitter sets."""
    # Group by key columns
    group_cols = ['transmitters', 'strategy', 'selection_method', 'power_filtering', 'power_threshold', 'whitening_config']

    # Calculate aggregation
    summary = results_df.groupby(group_cols).agg(
        ale_mean=('ale', 'mean'),
        ale_std=('ale', 'std'),
        ale_min=('ale', 'min'),
        ale_max=('ale', 'max'),
        ale_count=('ale', 'count'),
        pd_mean=('pd', 'mean'),
        pd_std=('pd', 'std'),
        precision_mean=('precision', 'mean'),
        f1_mean=('f1_score', 'mean'),
        n_est_mean=('n_estimated', 'mean'),
        n_est_raw_mean=('n_est_raw', 'mean'),
        n_est_diff_mean=('n_est_diff', 'mean'),
        count_error_mean=('count_error', 'mean'),
    ).reset_index()

    # Sort by transmitters and then by ALE
    summary = summary.sort_values(['transmitters', 'ale_mean'])

    return summary


def analyze_glrt_score_correlation(results_df: pd.DataFrame) -> Dict:
    """
    Analyze correlation between GLRT scores and localization performance.

    Determines if GLRT scores can predict which configuration will perform best.

    Returns
    -------
    dict
        Correlation analysis results including:
        - Overall correlations (Pearson/Spearman)
        - Per-directory comparison of best ALE config vs. lowest GLRT score config
        - Summary statistics
    """
    from scipy import stats

    analysis = {
        'overall_correlations': {},
        'per_directory_analysis': [],
        'matching_rate': 0.0,
        'summary': '',
    }

    # Check if GLRT score columns exist
    if 'glrt_final_score' not in results_df.columns:
        analysis['summary'] = 'GLRT score columns not found in results.'
        return analysis

    # Filter out rows with missing/invalid GLRT scores OR missing ALE
    valid_df = results_df[
        (results_df['glrt_final_score'].notna()) &
        (results_df['glrt_final_score'] > 0) &
        (results_df['ale'].notna())
    ].copy()

    if len(valid_df) < 10:
        analysis['summary'] = f'Insufficient data for correlation analysis ({len(valid_df)} valid experiments).'
        return analysis

    # Overall correlations
    try:
        # Pearson correlation between final GLRT score and ALE
        pearson_r, pearson_p = stats.pearsonr(valid_df['glrt_final_score'], valid_df['ale'])
        analysis['overall_correlations']['final_score_vs_ale_pearson'] = {
            'r': float(pearson_r),
            'p_value': float(pearson_p),
        }

        # Spearman correlation (rank-based, more robust to outliers)
        spearman_r, spearman_p = stats.spearmanr(valid_df['glrt_final_score'], valid_df['ale'])
        analysis['overall_correlations']['final_score_vs_ale_spearman'] = {
            'r': float(spearman_r),
            'p_value': float(spearman_p),
        }

        # Correlation between score reduction and ALE
        if valid_df['glrt_score_reduction'].notna().any():
            reduction_data = valid_df[valid_df['glrt_score_reduction'].notna()]
            if len(reduction_data) > 10:
                sr_pearson_r, sr_pearson_p = stats.pearsonr(reduction_data['glrt_score_reduction'], reduction_data['ale'])
                analysis['overall_correlations']['score_reduction_vs_ale'] = {
                    'r': float(sr_pearson_r),
                    'p_value': float(sr_pearson_p),
                }

        # Correlation between n_iterations and ALE
        if valid_df['glrt_n_iterations'].notna().any():
            iter_data = valid_df[valid_df['glrt_n_iterations'].notna()]
            if len(iter_data) > 10:
                iter_pearson_r, iter_pearson_p = stats.pearsonr(iter_data['glrt_n_iterations'], iter_data['ale'])
                analysis['overall_correlations']['n_iterations_vs_ale'] = {
                    'r': float(iter_pearson_r),
                    'p_value': float(iter_pearson_p),
                }

    except Exception as e:
        analysis['overall_correlations']['error'] = str(e)

    # Per-directory analysis: does lowest GLRT score predict best ALE?
    matches = 0
    total_dirs = 0

    for dir_name in valid_df['dir_name'].unique():
        dir_df = valid_df[valid_df['dir_name'] == dir_name]

        if len(dir_df) < 2:
            continue

        total_dirs += 1

        # Find config with lowest ALE
        best_ale_idx = dir_df['ale'].idxmin()
        best_ale_config = dir_df.loc[best_ale_idx]

        # Find config with lowest final GLRT score
        lowest_glrt_idx = dir_df['glrt_final_score'].idxmin()
        lowest_glrt_config = dir_df.loc[lowest_glrt_idx]

        # Check if they match (same strategy, selection method, etc.)
        config_match = (
            best_ale_config['strategy'] == lowest_glrt_config['strategy'] and
            best_ale_config['selection_method'] == lowest_glrt_config['selection_method'] and
            best_ale_config['power_filtering'] == lowest_glrt_config['power_filtering']
        )

        if config_match:
            matches += 1

        analysis['per_directory_analysis'].append({
            'dir_name': dir_name,
            'best_ale': float(best_ale_config['ale']),
            'best_ale_config': f"{best_ale_config['strategy']}_{best_ale_config['selection_method']}",
            'best_ale_glrt_score': float(best_ale_config['glrt_final_score']),
            'lowest_glrt_score': float(lowest_glrt_config['glrt_final_score']),
            'lowest_glrt_config': f"{lowest_glrt_config['strategy']}_{lowest_glrt_config['selection_method']}",
            'lowest_glrt_ale': float(lowest_glrt_config['ale']),
            'configs_match': config_match,
        })

    # Compute matching rate
    if total_dirs > 0:
        analysis['matching_rate'] = matches / total_dirs

    # Generate summary
    pearson_r = analysis['overall_correlations'].get('final_score_vs_ale_pearson', {}).get('r', None)
    spearman_r = analysis['overall_correlations'].get('final_score_vs_ale_spearman', {}).get('r', None)

    if pearson_r is not None:
        correlation_strength = 'weak'
        if abs(pearson_r) > 0.5:
            correlation_strength = 'moderate'
        if abs(pearson_r) > 0.7:
            correlation_strength = 'strong'

        direction = 'positive' if pearson_r > 0 else 'negative'

        analysis['summary'] = (
            f"GLRT final score shows {correlation_strength} {direction} correlation with ALE "
            f"(Pearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}). "
            f"The config with lowest GLRT score matched the best ALE config in "
            f"{analysis['matching_rate']*100:.1f}% of directories ({matches}/{total_dirs})."
        )
    else:
        analysis['summary'] = f"Could not compute correlations. Matching rate: {analysis['matching_rate']*100:.1f}%"

    return analysis


def create_final_results(results_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Create final_results.csv by selecting the best strategy per directory based on lowest BIC.

    For each unique (dir_name, transmitters, seed) combination, select the row with
    the lowest combo_bic score.

    Parameters
    ----------
    results_df : pd.DataFrame
        Full results dataframe with all strategies
    output_dir : Path
        Output directory

    Returns
    -------
    pd.DataFrame
        Final results with one row per directory (best strategy selected by BIC)
    """
    if 'combo_bic' not in results_df.columns:
        print("Warning: combo_bic column not found, cannot create final results")
        return pd.DataFrame()

    # Group by directory identifiers and select row with minimum BIC
    group_cols = ['dir_name', 'transmitters', 'seed']
    available_group_cols = [col for col in group_cols if col in results_df.columns]

    if not available_group_cols:
        print("Warning: No grouping columns found")
        return pd.DataFrame()

    # For each group, get the row with minimum combo_bic
    idx = results_df.groupby(available_group_cols)['combo_bic'].idxmin()
    final_df = results_df.loc[idx].copy()

    # Select relevant columns for final results
    final_columns = [
        'dir_name', 'transmitters', 'seed', 'tx_count',
        'strategy', 'whitening_config', 'sigma_noise',
        'combo_n_tx', 'combo_ale', 'combo_pd', 'combo_precision',
        'combo_count_error', 'combo_rmse', 'combo_bic',
        # Also include GLRT metrics for reference
        'ale', 'pd', 'precision', 'n_estimated',
    ]

    available_final_cols = [col for col in final_columns if col in final_df.columns]
    final_df = final_df[available_final_cols]

    # Sort by dir_name, transmitters, seed
    sort_cols = [col for col in ['dir_name', 'transmitters', 'seed'] if col in final_df.columns]
    if sort_cols:
        final_df = final_df.sort_values(sort_cols)

    # Save to CSV
    csv_path = output_dir / 'final_results.csv'
    final_df.to_csv(csv_path, index=False)

    print(f"Final results saved to: {csv_path}")
    print(f"  Total directories: {len(final_df)}")
    print(f"  (Selected best strategy per directory based on lowest BIC)")

    return final_df


def cleanup_visualizations_for_best_only(final_df: pd.DataFrame, output_dir: Path):
    """
    Remove visualization directories for non-best strategies, keeping only the best.

    For each directory in final_df, we need to identify which experiment_name
    corresponds to the best strategy and remove others.

    Parameters
    ----------
    final_df : pd.DataFrame
        Final results with best strategy per directory
    output_dir : Path
        Output directory containing glrt_visualizations/
    """
    import shutil

    vis_dir = output_dir / 'glrt_visualizations'
    if not vis_dir.exists():
        print("No visualization directory found, skipping cleanup")
        return

    # Build set of experiment names to keep (from final_df)
    keep_dirs = set()
    for _, row in final_df.iterrows():
        # Reconstruct experiment name components
        transmitters = row.get('transmitters', '')
        tx_underscore = transmitters.replace(',', '_') if transmitters else ''
        strategy = row.get('strategy', '')
        whitening = row.get('whitening_config', '')

        if tx_underscore and strategy and whitening:
            pattern_start = f"{tx_underscore}_{strategy}_{whitening}"
            keep_dirs.add(pattern_start)

    if not keep_dirs:
        print("Could not determine directories to keep, skipping cleanup")
        return

    # List all visualization subdirectories
    removed_count = 0
    kept_count = 0

    for subdir in vis_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Check if this directory matches any of the keep patterns
        should_keep = False
        for pattern in keep_dirs:
            if subdir.name.startswith(pattern):
                should_keep = True
                break

        if should_keep:
            kept_count += 1
        else:
            # Remove this directory
            try:
                shutil.rmtree(subdir)
                removed_count += 1
            except Exception as e:
                print(f"Warning: Could not remove {subdir}: {e}")

    print(f"Visualization cleanup: kept {kept_count} directories, removed {removed_count}")
