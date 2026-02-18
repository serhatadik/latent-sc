"""Visualization plots for sweep results."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict


def generate_plots(
    results_df: pd.DataFrame,
    tx_count_summaries: Dict[int, pd.DataFrame],
    universal_summary: pd.DataFrame,
    output_dir: Path,
):
    """Generate visualization plots."""

    # Plot 1: Summary plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1a: Best ALE per TX count
    ax = axes[0, 0]
    tx_counts = sorted(tx_count_summaries.keys())
    best_ales = [tx_count_summaries[tc].iloc[0]['ale_mean'] for tc in tx_counts]
    ax.bar(tx_counts, best_ales, color='steelblue')
    ax.set_xlabel('TX Count')
    ax.set_ylabel('Best Mean ALE (m)')
    ax.set_title('Best ALE by TX Count')
    ax.set_xticks(tx_counts)

    # Plot 1b: Selection method comparison
    ax = axes[0, 1]
    # Create label combining method and power filtering
    results_df['method_label'] = results_df.apply(
        lambda r: f"{r['selection_method']}{' + PF' if r['power_filtering'] else ''}", axis=1
    )
    method_comparison = results_df.groupby('method_label')['ale'].agg(['mean', 'std']).reset_index()
    # Colors: Light/Dark Green for Cluster, Light/Dark Red for Max
    method_colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b'][:len(method_comparison)]
    bars = ax.bar(method_comparison['method_label'], method_comparison['mean'],
                  yerr=method_comparison['std'], color=method_colors, capsize=5)
    ax.set_xlabel('Selection Configuration')
    ax.set_ylabel('Mean ALE (m)')
    ax.set_title('Selection Method Comparison')
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    # Plot 1c: Whitening Config comparison
    ax = axes[1, 0]
    config_comparison = results_df.groupby('whitening_config')['ale'].agg(['mean', 'std']).reset_index()
    # Use dynamic colors
    cmap = plt.get_cmap('tab10')
    whitening_colors = [cmap(i) for i in np.linspace(0, 1, len(config_comparison))]
    bars = ax.bar(config_comparison['whitening_config'], config_comparison['mean'],
                  yerr=config_comparison['std'], color=whitening_colors, capsize=5)
    ax.set_xlabel('Whitening Config')
    ax.set_ylabel('Mean ALE (m)')
    ax.set_title('Whitening Config Comparison')
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    # Plot 1d: Fixed vs Dynamic boxplot
    ax = axes[1, 1]
    fixed_ales = results_df[results_df['strategy'].str.startswith('fixed')]['ale']
    dynamic_ales = results_df[~results_df['strategy'].str.startswith('fixed')]['ale']
    ax.boxplot([fixed_ales, dynamic_ales], labels=['Fixed sigma', 'Dynamic sigma'])
    ax.set_ylabel('ALE (m)')
    ax.set_title('Fixed vs Dynamic Sigma Noise')

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_plots.png', dpi=150)
    plt.close()

    # Plot 2: Strategy comparison heatmap by TX count
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get top strategies
    top_strategies = universal_summary.head(10)[['strategy', 'selection_method', 'whitening_config']].values.tolist()

    # Build heatmap data
    heatmap_data = []
    for tx_count in sorted(tx_count_summaries.keys()):
        row = []
        for strat, sel, rho in top_strategies:
            df = tx_count_summaries[tx_count]
            match = df[(df['strategy'] == strat) & (df['selection_method'] == sel) & (df['whitening_config'] == rho)]
            if len(match) > 0:
                row.append(match.iloc[0]['ale_mean'])
            else:
                row.append(np.nan)
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)

    im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(top_strategies)))
    ax.set_xticklabels([f"{s[0]}\n({s[1]}/{s[2][:3]})" for s in top_strategies], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(tx_count_summaries)))
    ax.set_yticklabels([f"TX={tc}" for tc in sorted(tx_count_summaries.keys())])

    plt.colorbar(im, label='Mean ALE (m)')
    ax.set_title('Strategy Performance by TX Count (Mean ALE)')

    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap.png', dpi=150)
    plt.close()

    # Plot 3: Power Threshold Sensitivity
    if 'power_threshold' in results_df.columns:
        pf_results = results_df[results_df['power_filtering'] == True].copy()
        if len(pf_results) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Group by threshold and selection method
            sensitivity = pf_results.groupby(['power_threshold', 'selection_method'])['ale'].mean().reset_index()

            for method in sensitivity['selection_method'].unique():
                 subset = sensitivity[sensitivity['selection_method'] == method]
                 subset = subset.sort_values('power_threshold')
                 ax.plot(subset['power_threshold'], subset['ale'], marker='o', label=f"{method} + PF")

            ax.set_xlabel('Power Density Threshold')
            ax.set_ylabel('Mean ALE (m)')
            ax.set_title('Sensitivity to Power Density Threshold (Mean across all strategies)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            plt.tight_layout()
            plt.savefig(output_dir / 'threshold_sensitivity.png', dpi=150)
            plt.close()

    print(f"Plots saved to {output_dir}")
