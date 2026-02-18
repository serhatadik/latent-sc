"""Markdown report generators for BIC, final, and comprehensive analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from .analysis import analyze_glrt_score_correlation


def generate_bic_analysis_report(bic_df: pd.DataFrame, output_dir: Path):
    """
    Generate a markdown analysis report for BIC-based combinatorial selection results.

    Parameters
    ----------
    bic_df : pd.DataFrame
        BIC results dataframe
    output_dir : Path
        Output directory
    """
    report_lines = []
    report_lines.append("# BIC Combinatorial Selection Analysis Report")
    report_lines.append("")
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Overall Summary
    report_lines.append("## Overall Summary")
    report_lines.append("")
    report_lines.append(f"- **Total experiments**: {len(bic_df)}")

    if 'tx_count' in bic_df.columns:
        report_lines.append(f"- **TX count range**: {int(bic_df['tx_count'].min())} - {int(bic_df['tx_count'].max())}")

    report_lines.append("")
    report_lines.append("### Aggregate Metrics")
    report_lines.append("")
    report_lines.append("| Metric | Mean | Std | Min | Max |")
    report_lines.append("|--------|------|-----|-----|-----|")

    for col, name in [('combo_ale', 'ALE (m)'), ('combo_pd', 'Pd'), ('combo_precision', 'Precision'), ('combo_count_error', 'Count Error')]:
        if col in bic_df.columns:
            valid = bic_df[col].dropna()
            if len(valid) > 0:
                if col in ['combo_pd', 'combo_precision']:
                    report_lines.append(f"| {name} | {valid.mean()*100:.1f}% | {valid.std()*100:.1f}% | {valid.min()*100:.1f}% | {valid.max()*100:.1f}% |")
                else:
                    report_lines.append(f"| {name} | {valid.mean():.2f} | {valid.std():.2f} | {valid.min():.2f} | {valid.max():.2f} |")

    # Analysis by Strategy
    if 'strategy' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Analysis by Strategy")
        report_lines.append("")
        report_lines.append("| Strategy | Count | Mean ALE | Mean Pd | Mean Precision | Mean Count Error |")
        report_lines.append("|----------|-------|----------|---------|----------------|------------------|")

        for strategy in sorted(bic_df['strategy'].unique()):
            subset = bic_df[bic_df['strategy'] == strategy]
            count = len(subset)
            ale = subset['combo_ale'].mean() if 'combo_ale' in subset.columns else np.nan
            pd_val = subset['combo_pd'].mean() if 'combo_pd' in subset.columns else np.nan
            prec = subset['combo_precision'].mean() if 'combo_precision' in subset.columns else np.nan
            ce = subset['combo_count_error'].mean() if 'combo_count_error' in subset.columns else np.nan

            ale_str = f"{ale:.2f}" if not np.isnan(ale) else "-"
            pd_str = f"{pd_val*100:.1f}%" if not np.isnan(pd_val) else "-"
            prec_str = f"{prec*100:.1f}%" if not np.isnan(prec) else "-"
            ce_str = f"{ce:.2f}" if not np.isnan(ce) else "-"

            report_lines.append(f"| {strategy} | {count} | {ale_str} | {pd_str} | {prec_str} | {ce_str} |")

    # Analysis by Whitening Config
    if 'whitening_config' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Analysis by Whitening Configuration")
        report_lines.append("")
        report_lines.append("| Whitening Config | Count | Mean ALE | Mean Pd | Mean Precision | Mean Count Error |")
        report_lines.append("|------------------|-------|----------|---------|----------------|------------------|")

        for wc in sorted(bic_df['whitening_config'].unique()):
            subset = bic_df[bic_df['whitening_config'] == wc]
            count = len(subset)
            ale = subset['combo_ale'].mean() if 'combo_ale' in subset.columns else np.nan
            pd_val = subset['combo_pd'].mean() if 'combo_pd' in subset.columns else np.nan
            prec = subset['combo_precision'].mean() if 'combo_precision' in subset.columns else np.nan
            ce = subset['combo_count_error'].mean() if 'combo_count_error' in subset.columns else np.nan

            ale_str = f"{ale:.2f}" if not np.isnan(ale) else "-"
            pd_str = f"{pd_val*100:.1f}%" if not np.isnan(pd_val) else "-"
            prec_str = f"{prec*100:.1f}%" if not np.isnan(prec) else "-"
            ce_str = f"{ce:.2f}" if not np.isnan(ce) else "-"

            report_lines.append(f"| {wc} | {count} | {ale_str} | {pd_str} | {prec_str} | {ce_str} |")

    # Analysis by TX Count
    if 'tx_count' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Analysis by True TX Count")
        report_lines.append("")
        report_lines.append("| TX Count | Experiments | Mean ALE | Mean Pd | Mean Precision | Mean Count Error | Mean Est. TXs |")
        report_lines.append("|----------|-------------|----------|---------|----------------|------------------|---------------|")

        for tx_count in sorted(bic_df['tx_count'].unique()):
            subset = bic_df[bic_df['tx_count'] == tx_count]
            count = len(subset)
            ale = subset['combo_ale'].mean() if 'combo_ale' in subset.columns else np.nan
            pd_val = subset['combo_pd'].mean() if 'combo_pd' in subset.columns else np.nan
            prec = subset['combo_precision'].mean() if 'combo_precision' in subset.columns else np.nan
            ce = subset['combo_count_error'].mean() if 'combo_count_error' in subset.columns else np.nan
            est_tx = subset['combo_n_tx'].mean() if 'combo_n_tx' in subset.columns else np.nan

            ale_str = f"{ale:.2f}" if not np.isnan(ale) else "-"
            pd_str = f"{pd_val*100:.1f}%" if not np.isnan(pd_val) else "-"
            prec_str = f"{prec*100:.1f}%" if not np.isnan(prec) else "-"
            ce_str = f"{ce:.2f}" if not np.isnan(ce) else "-"
            est_str = f"{est_tx:.2f}" if not np.isnan(est_tx) else "-"

            report_lines.append(f"| {int(tx_count)} | {count} | {ale_str} | {pd_str} | {prec_str} | {ce_str} | {est_str} |")

    # Analysis by Strategy + Whitening (Top configurations)
    if 'strategy' in bic_df.columns and 'whitening_config' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Best Configurations (by Mean ALE)")
        report_lines.append("")

        grouped = bic_df.groupby(['strategy', 'whitening_config']).agg({
            'combo_ale': ['mean', 'std', 'count'],
            'combo_pd': 'mean',
            'combo_precision': 'mean',
            'combo_count_error': 'mean'
        }).reset_index()

        # Flatten column names
        grouped.columns = ['strategy', 'whitening_config', 'ale_mean', 'ale_std', 'count', 'pd_mean', 'prec_mean', 'ce_mean']

        # Sort by ALE mean (lower is better)
        grouped = grouped.sort_values('ale_mean')

        report_lines.append("| Rank | Strategy | Whitening | Count | Mean ALE | Std ALE | Mean Pd | Mean Prec | Mean CE |")
        report_lines.append("|------|----------|-----------|-------|----------|---------|---------|-----------|---------|")

        for i, row in grouped.head(10).iterrows():
            rank = grouped.index.get_loc(i) + 1
            ale_str = f"{row['ale_mean']:.2f}" if not np.isnan(row['ale_mean']) else "-"
            ale_std_str = f"{row['ale_std']:.2f}" if not np.isnan(row['ale_std']) else "-"
            pd_str = f"{row['pd_mean']*100:.1f}%" if not np.isnan(row['pd_mean']) else "-"
            prec_str = f"{row['prec_mean']*100:.1f}%" if not np.isnan(row['prec_mean']) else "-"
            ce_str = f"{row['ce_mean']:.2f}" if not np.isnan(row['ce_mean']) else "-"

            report_lines.append(f"| {rank} | {row['strategy']} | {row['whitening_config']} | {int(row['count'])} | {ale_str} | {ale_std_str} | {pd_str} | {prec_str} | {ce_str} |")

    # TX Count Estimation Accuracy
    if 'tx_count' in bic_df.columns and 'combo_n_tx' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## TX Count Estimation Accuracy")
        report_lines.append("")

        # Perfect count rate
        perfect_count = (bic_df['combo_count_error'] == 0).sum()
        total = len(bic_df)
        report_lines.append(f"- **Perfect count rate**: {perfect_count}/{total} ({perfect_count/total*100:.1f}%)")

        # Under/over estimation
        under = (bic_df['combo_n_tx'] < bic_df['tx_count']).sum()
        over = (bic_df['combo_n_tx'] > bic_df['tx_count']).sum()
        exact = (bic_df['combo_n_tx'] == bic_df['tx_count']).sum()
        report_lines.append(f"- **Under-estimation**: {under} ({under/total*100:.1f}%)")
        report_lines.append(f"- **Exact**: {exact} ({exact/total*100:.1f}%)")
        report_lines.append(f"- **Over-estimation**: {over} ({over/total*100:.1f}%)")

        report_lines.append("")
        report_lines.append("### Count Error Distribution")
        report_lines.append("")
        report_lines.append("| Count Error | Occurrences | Percentage |")
        report_lines.append("|-------------|-------------|------------|")

        for ce in sorted(bic_df['combo_count_error'].unique()):
            ce_count = (bic_df['combo_count_error'] == ce).sum()
            report_lines.append(f"| {int(ce)} | {ce_count} | {ce_count/total*100:.1f}% |")

    # Per-Directory Analysis
    if 'dir_name' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Per-Directory Analysis")
        report_lines.append("")

        for dir_name in sorted(bic_df['dir_name'].unique()):
            dir_subset = bic_df[bic_df['dir_name'] == dir_name]
            report_lines.append(f"### {dir_name}")
            report_lines.append("")

            # Summary for this directory
            dir_count = len(dir_subset)
            dir_ale = dir_subset['combo_ale'].mean() if 'combo_ale' in dir_subset.columns else np.nan
            dir_pd = dir_subset['combo_pd'].mean() if 'combo_pd' in dir_subset.columns else np.nan
            dir_prec = dir_subset['combo_precision'].mean() if 'combo_precision' in dir_subset.columns else np.nan
            dir_ce = dir_subset['combo_count_error'].mean() if 'combo_count_error' in dir_subset.columns else np.nan

            report_lines.append(f"- **Experiments**: {dir_count}")
            if not np.isnan(dir_ale):
                report_lines.append(f"- **Mean ALE**: {dir_ale:.2f} m")
            if not np.isnan(dir_pd):
                report_lines.append(f"- **Mean Pd**: {dir_pd*100:.1f}%")
            if not np.isnan(dir_prec):
                report_lines.append(f"- **Mean Precision**: {dir_prec*100:.1f}%")
            if not np.isnan(dir_ce):
                report_lines.append(f"- **Mean Count Error**: {dir_ce:.2f}")

            # TX count estimation for this directory
            if 'tx_count' in dir_subset.columns and 'combo_n_tx' in dir_subset.columns:
                perfect = (dir_subset['combo_count_error'] == 0).sum()
                report_lines.append(f"- **Perfect Count Rate**: {perfect}/{dir_count} ({perfect/dir_count*100:.1f}%)")

            report_lines.append("")

            # Breakdown by TX count within this directory
            if 'tx_count' in dir_subset.columns and len(dir_subset['tx_count'].unique()) > 1:
                report_lines.append("| TX Count | Experiments | Mean ALE | Mean Pd | Mean Prec | Mean CE | Mean Est |")
                report_lines.append("|----------|-------------|----------|---------|-----------|---------|----------|")

                for tx_count in sorted(dir_subset['tx_count'].unique()):
                    tx_subset = dir_subset[dir_subset['tx_count'] == tx_count]
                    tx_count_n = len(tx_subset)
                    tx_ale = tx_subset['combo_ale'].mean() if 'combo_ale' in tx_subset.columns else np.nan
                    tx_pd = tx_subset['combo_pd'].mean() if 'combo_pd' in tx_subset.columns else np.nan
                    tx_prec = tx_subset['combo_precision'].mean() if 'combo_precision' in tx_subset.columns else np.nan
                    tx_ce = tx_subset['combo_count_error'].mean() if 'combo_count_error' in tx_subset.columns else np.nan
                    tx_est = tx_subset['combo_n_tx'].mean() if 'combo_n_tx' in tx_subset.columns else np.nan

                    ale_str = f"{tx_ale:.2f}" if not np.isnan(tx_ale) else "-"
                    pd_str = f"{tx_pd*100:.1f}%" if not np.isnan(tx_pd) else "-"
                    prec_str = f"{tx_prec*100:.1f}%" if not np.isnan(tx_prec) else "-"
                    ce_str = f"{tx_ce:.2f}" if not np.isnan(tx_ce) else "-"
                    est_str = f"{tx_est:.2f}" if not np.isnan(tx_est) else "-"

                    report_lines.append(f"| {int(tx_count)} | {tx_count_n} | {ale_str} | {pd_str} | {prec_str} | {ce_str} | {est_str} |")

                report_lines.append("")

    # Per-Transmitter Set Analysis (by unique transmitters combination)
    if 'transmitters' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Per-Transmitter Set Analysis")
        report_lines.append("")
        report_lines.append("| Transmitters | TX Count | Experiments | Mean ALE | Mean Pd | Mean Prec | Mean CE | Perfect Rate |")
        report_lines.append("|--------------|----------|-------------|----------|---------|-----------|---------|--------------|")

        for tx_set in sorted(bic_df['transmitters'].unique()):
            tx_subset = bic_df[bic_df['transmitters'] == tx_set]
            tx_count_val = tx_subset['tx_count'].iloc[0] if 'tx_count' in tx_subset.columns else 0
            n_exp = len(tx_subset)
            ale = tx_subset['combo_ale'].mean() if 'combo_ale' in tx_subset.columns else np.nan
            pd_val = tx_subset['combo_pd'].mean() if 'combo_pd' in tx_subset.columns else np.nan
            prec = tx_subset['combo_precision'].mean() if 'combo_precision' in tx_subset.columns else np.nan
            ce = tx_subset['combo_count_error'].mean() if 'combo_count_error' in tx_subset.columns else np.nan
            perfect = (tx_subset['combo_count_error'] == 0).sum() if 'combo_count_error' in tx_subset.columns else 0

            ale_str = f"{ale:.2f}" if not np.isnan(ale) else "-"
            pd_str = f"{pd_val*100:.1f}%" if not np.isnan(pd_val) else "-"
            prec_str = f"{prec*100:.1f}%" if not np.isnan(prec) else "-"
            ce_str = f"{ce:.2f}" if not np.isnan(ce) else "-"
            perfect_str = f"{perfect}/{n_exp} ({perfect/n_exp*100:.0f}%)"

            report_lines.append(f"| {tx_set} | {int(tx_count_val)} | {n_exp} | {ale_str} | {pd_str} | {prec_str} | {ce_str} | {perfect_str} |")

    # === Reconstruction Error Analysis ===
    if 'recon_rmse' in bic_df.columns:
        report_lines.append("")
        report_lines.append("## Reconstruction Error Analysis")
        report_lines.append("")

        valid_recon = bic_df[bic_df['recon_status'] == 'success']
        report_lines.append(f"- **Experiments with validation data**: {len(valid_recon)}/{len(bic_df)}")

        if len(valid_recon) > 0:
            report_lines.append(f"- **Mean Reconstruction RMSE**: {valid_recon['recon_rmse'].mean():.2f} dB")
            report_lines.append(f"- **Mean Reconstruction MAE**: {valid_recon['recon_mae'].mean():.2f} dB")
            report_lines.append(f"- **Mean Reconstruction Bias**: {valid_recon['recon_bias'].mean():.2f} dB")
            report_lines.append(f"- **Mean Max Error**: {valid_recon['recon_max_error'].mean():.2f} dB")
            report_lines.append(f"- **Mean Validation Points**: {valid_recon['recon_n_val_points'].mean():.0f}")

            # Breakdown by status
            report_lines.append("")
            report_lines.append("### Reconstruction Status Breakdown")
            report_lines.append("")
            report_lines.append("| Status | Count | Percentage |")
            report_lines.append("|--------|-------|------------|")
            for status in bic_df['recon_status'].unique():
                count = (bic_df['recon_status'] == status).sum()
                pct = count / len(bic_df) * 100
                report_lines.append(f"| {status} | {count} | {pct:.1f}% |")

            # Reconstruction error by TX count
            if 'tx_count' in valid_recon.columns and len(valid_recon) > 0:
                report_lines.append("")
                report_lines.append("### Reconstruction Error by TX Count")
                report_lines.append("")
                report_lines.append("| TX Count | Experiments | Mean RMSE | Mean MAE | Mean Bias | Mean Max Error |")
                report_lines.append("|----------|-------------|-----------|----------|-----------|----------------|")

                for tx_count in sorted(valid_recon['tx_count'].unique()):
                    subset = valid_recon[valid_recon['tx_count'] == tx_count]
                    n_exp = len(subset)
                    rmse = subset['recon_rmse'].mean()
                    mae = subset['recon_mae'].mean()
                    bias = subset['recon_bias'].mean()
                    max_err = subset['recon_max_error'].mean()

                    rmse_str = f"{rmse:.2f}" if not np.isnan(rmse) else "-"
                    mae_str = f"{mae:.2f}" if not np.isnan(mae) else "-"
                    bias_str = f"{bias:.2f}" if not np.isnan(bias) else "-"
                    max_str = f"{max_err:.2f}" if not np.isnan(max_err) else "-"

                    report_lines.append(f"| {int(tx_count)} | {n_exp} | {rmse_str} | {mae_str} | {bias_str} | {max_str} |")

    # Save report
    report_path = output_dir / 'analysis_report_bic.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"BIC analysis report saved to: {report_path}")


def generate_final_analysis_report(final_df: pd.DataFrame, output_dir: Path):
    """
    Generate analysis_report_final.md for the best-strategy-per-directory results.

    Parameters
    ----------
    final_df : pd.DataFrame
        Final results dataframe (one row per directory)
    output_dir : Path
        Output directory
    """
    report_lines = []
    report_lines.append("# Final Results Analysis Report")
    report_lines.append("")
    report_lines.append("This report analyzes results after selecting the **best strategy per directory**")
    report_lines.append("based on lowest BIC score.")
    report_lines.append("")
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Overall Summary
    report_lines.append("## Overall Summary")
    report_lines.append("")
    report_lines.append(f"- **Total directories**: {len(final_df)}")

    if 'tx_count' in final_df.columns:
        report_lines.append(f"- **TX count range**: {int(final_df['tx_count'].min())} - {int(final_df['tx_count'].max())}")

    report_lines.append("")
    report_lines.append("### Aggregate Metrics (Best Strategy per Directory)")
    report_lines.append("")
    report_lines.append("| Metric | Mean | Std | Min | Max |")
    report_lines.append("|--------|------|-----|-----|-----|")

    for col, name in [('combo_ale', 'ALE (m)'), ('combo_pd', 'Pd'), ('combo_precision', 'Precision'), ('combo_count_error', 'Count Error'), ('combo_bic', 'BIC')]:
        if col in final_df.columns:
            valid = final_df[col].dropna()
            if len(valid) > 0:
                if col in ['combo_pd', 'combo_precision']:
                    report_lines.append(f"| {name} | {valid.mean()*100:.1f}% | {valid.std()*100:.1f}% | {valid.min()*100:.1f}% | {valid.max()*100:.1f}% |")
                else:
                    report_lines.append(f"| {name} | {valid.mean():.2f} | {valid.std():.2f} | {valid.min():.2f} | {valid.max():.2f} |")

    # Strategy selection distribution
    if 'strategy' in final_df.columns:
        report_lines.append("")
        report_lines.append("## Strategy Selection Distribution")
        report_lines.append("")
        report_lines.append("Which strategies were selected as best (by BIC) across directories:")
        report_lines.append("")
        report_lines.append("| Strategy | Times Selected | Percentage |")
        report_lines.append("|----------|----------------|------------|")

        strategy_counts = final_df['strategy'].value_counts()
        total = len(final_df)
        for strategy, count in strategy_counts.items():
            report_lines.append(f"| {strategy} | {count} | {count/total*100:.1f}% |")

    # Whitening config selection distribution
    if 'whitening_config' in final_df.columns:
        report_lines.append("")
        report_lines.append("## Whitening Configuration Selection Distribution")
        report_lines.append("")
        report_lines.append("| Whitening Config | Times Selected | Percentage |")
        report_lines.append("|------------------|----------------|------------|")

        wc_counts = final_df['whitening_config'].value_counts()
        for wc, count in wc_counts.items():
            report_lines.append(f"| {wc} | {count} | {count/total*100:.1f}% |")

    # Analysis by TX Count
    if 'tx_count' in final_df.columns:
        report_lines.append("")
        report_lines.append("## Results by True TX Count")
        report_lines.append("")
        report_lines.append("| TX Count | Directories | Mean ALE | Mean Pd | Mean Precision | Mean Count Error | Mean Est. TXs |")
        report_lines.append("|----------|-------------|----------|---------|----------------|------------------|---------------|")

        for tx_count in sorted(final_df['tx_count'].unique()):
            subset = final_df[final_df['tx_count'] == tx_count]
            n_dirs = len(subset)
            ale = subset['combo_ale'].mean() if 'combo_ale' in subset.columns else np.nan
            pd_val = subset['combo_pd'].mean() if 'combo_pd' in subset.columns else np.nan
            prec = subset['combo_precision'].mean() if 'combo_precision' in subset.columns else np.nan
            ce = subset['combo_count_error'].mean() if 'combo_count_error' in subset.columns else np.nan
            est_tx = subset['combo_n_tx'].mean() if 'combo_n_tx' in subset.columns else np.nan

            ale_str = f"{ale:.2f}" if not np.isnan(ale) else "-"
            pd_str = f"{pd_val*100:.1f}%" if not np.isnan(pd_val) else "-"
            prec_str = f"{prec*100:.1f}%" if not np.isnan(prec) else "-"
            ce_str = f"{ce:.2f}" if not np.isnan(ce) else "-"
            est_str = f"{est_tx:.2f}" if not np.isnan(est_tx) else "-"

            report_lines.append(f"| {int(tx_count)} | {n_dirs} | {ale_str} | {pd_str} | {prec_str} | {ce_str} | {est_str} |")

    # TX Count Estimation Accuracy
    if 'tx_count' in final_df.columns and 'combo_n_tx' in final_df.columns:
        report_lines.append("")
        report_lines.append("## TX Count Estimation Accuracy")
        report_lines.append("")

        total = len(final_df)
        if 'combo_count_error' in final_df.columns:
            perfect = (final_df['combo_count_error'] == 0).sum()
            report_lines.append(f"- **Perfect count rate**: {perfect}/{total} ({perfect/total*100:.1f}%)")

        under = (final_df['combo_n_tx'] < final_df['tx_count']).sum()
        over = (final_df['combo_n_tx'] > final_df['tx_count']).sum()
        exact = (final_df['combo_n_tx'] == final_df['tx_count']).sum()
        report_lines.append(f"- **Under-estimation**: {under} ({under/total*100:.1f}%)")
        report_lines.append(f"- **Exact**: {exact} ({exact/total*100:.1f}%)")
        report_lines.append(f"- **Over-estimation**: {over} ({over/total*100:.1f}%)")

    # Per-Directory Details
    if 'dir_name' in final_df.columns:
        report_lines.append("")
        report_lines.append("## Per-Directory Results")
        report_lines.append("")
        report_lines.append("| Directory | TXs | Best Strategy | Whitening | Est TXs | ALE | Pd | Prec | BIC |")
        report_lines.append("|-----------|-----|---------------|-----------|---------|-----|-----|------|-----|")

        for _, row in final_df.iterrows():
            dir_name = row.get('dir_name', '-')
            tx_count = int(row['tx_count']) if 'tx_count' in row and not pd.isna(row['tx_count']) else '-'
            strategy = row.get('strategy', '-')
            whitening = row.get('whitening_config', '-')
            est_tx = int(row['combo_n_tx']) if 'combo_n_tx' in row and not pd.isna(row['combo_n_tx']) else '-'
            ale = f"{row['combo_ale']:.1f}" if 'combo_ale' in row and not pd.isna(row['combo_ale']) else '-'
            pd_val = f"{row['combo_pd']*100:.0f}%" if 'combo_pd' in row and not pd.isna(row['combo_pd']) else '-'
            prec = f"{row['combo_precision']*100:.0f}%" if 'combo_precision' in row and not pd.isna(row['combo_precision']) else '-'
            bic = f"{row['combo_bic']:.1f}" if 'combo_bic' in row and not pd.isna(row['combo_bic']) else '-'

            report_lines.append(f"| {dir_name} | {tx_count} | {strategy} | {whitening} | {est_tx} | {ale} | {pd_val} | {prec} | {bic} |")

    # Save report
    report_path = output_dir / 'analysis_report_final.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Final analysis report saved to: {report_path}")


def generate_analysis_report(
    results_df: pd.DataFrame,
    tx_count_summaries: Dict[int, pd.DataFrame],
    universal_summary: pd.DataFrame,
    output_dir: Path,
    tx_set_summary: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate a comprehensive markdown analysis report.

    Returns
    -------
    str
        Path to the generated report
    """
    report_lines = []

    report_lines.append("# Comprehensive Reconstruction Parameter Sweep Analysis")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Overview
    report_lines.append("## Overview")
    report_lines.append("")
    report_lines.append(f"- **Total experiments**: {len(results_df)}")
    report_lines.append(f"- **Unique directories**: {results_df['dir_name'].nunique()}")
    report_lines.append(f"- **TX counts analyzed**: {sorted(results_df['tx_count'].unique())}")
    report_lines.append(f"- **Strategies tested**: {results_df['strategy'].nunique()}")
    report_lines.append(f"- **Selection methods**: {results_df['selection_method'].unique().tolist()}")
    report_lines.append(f"- **Whitening Configs**: {results_df['whitening_config'].unique().tolist()}")
    report_lines.append("")

    # Universal Analysis
    report_lines.append("## Universal Analysis (Across All TX Counts)")
    report_lines.append("")

    # Best overall strategy
    best_row = universal_summary.iloc[0]
    report_lines.append(f"### Best Overall Strategy")
    report_lines.append("")
    report_lines.append(f"- **Strategy**: `{best_row['strategy']}`")
    report_lines.append(f"- **Selection Method**: `{best_row['selection_method']}`")
    report_lines.append(f"- **Whitening Config**: `{best_row['whitening_config']}`")
    report_lines.append(f"- **Mean ALE**: {best_row['ale_mean']:.2f} m (+-{best_row['ale_std']:.2f})")
    report_lines.append(f"- **Mean Pd**: {best_row['pd_mean']*100:.1f}% (+-{best_row['pd_std']*100:.1f})")
    report_lines.append(f"- **Mean Precision**: {best_row['precision_mean']*100:.1f}%")
    if 'n_est_mean' in best_row:
        report_lines.append(f"- **Mean N Est (Filtered)**: {best_row['n_est_mean']:.2f} (raw: {best_row.get('n_est_raw_mean', 0):.2f})")
    if 'count_error_mean' in best_row:
        report_lines.append(f"- **Mean Count Error**: {best_row['count_error_mean']:.2f}")
    report_lines.append(f"- **Experiments**: {int(best_row['ale_count'])}")
    report_lines.append("")

    # Top 10 strategies table
    report_lines.append("### Top 10 Strategies (by Mean ALE)")
    report_lines.append("")
    report_lines.append("| Rank | Strategy | Selection | Power Filter | Threshold | Whitening Config | Mean ALE (m) | Mean Pd (%) | N |")
    report_lines.append("|------|----------|-----------|--------------|-----------|------------------|--------------|-------------|---|")
    for i, row in universal_summary.head(10).iterrows():
        rank = universal_summary.index.get_loc(i) + 1
        pf_str = "Yes" if row['power_filtering'] else "No"
        thresh_str = f"{row['power_threshold']}" if row['power_filtering'] else "-"
        report_lines.append(
            f"| {rank} | {row['strategy']} | {row['selection_method']} | {pf_str} | {thresh_str} | "
            f"{row['whitening_config']} | {row['ale_mean']:.2f} | "
            f"{row['pd_mean']*100:.1f} | {int(row['ale_count'])} |"
        )
    report_lines.append("")

    # Selection Method Comparison
    report_lines.append("### Selection Method Comparison")
    report_lines.append("")

    # Selection configurations: (method, use_power_filtering)
    selection_configs = [
        ('max', False),
        ('cluster', False),
        ('max', True),
        ('cluster', True),
    ]

    for method, use_pf in selection_configs:
        pf_suffix = " + PF" if use_pf else ""
        method_name = f"{method}{pf_suffix}"

        method_df = results_df[
            (results_df['selection_method'] == method) &
            (results_df['power_filtering'] == use_pf)
        ]

        if len(method_df) > 0:
            avg_ale = method_df['ale'].mean()
            avg_pd = method_df['pd'].mean()
            report_lines.append(f"- **{method_name}**: Mean ALE = {avg_ale:.2f} m, Mean Pd = {avg_pd*100:.1f}%")
    report_lines.append("")

    # Whitening Config Comparison
    report_lines.append("### Whitening Config Comparison")
    report_lines.append("")

    for config_name in results_df['whitening_config'].unique():
        config_df = results_df[results_df['whitening_config'] == config_name]
        if len(config_df) > 0:
            avg_ale = config_df['ale'].mean()
            avg_pd = config_df['pd'].mean()
            report_lines.append(f"- **{config_name}**: Mean ALE = {avg_ale:.2f} m, Mean Pd = {avg_pd*100:.1f}%")
    report_lines.append("")

    # Fixed vs Dynamic Comparison
    report_lines.append("### Fixed vs Dynamic Sigma Noise Comparison")
    report_lines.append("")

    fixed_df = results_df[results_df['strategy'].str.startswith('fixed')]
    dynamic_df = results_df[~results_df['strategy'].str.startswith('fixed')]

    if len(fixed_df) > 0 and len(dynamic_df) > 0:
        fixed_ale = fixed_df['ale'].mean()
        dynamic_ale = dynamic_df['ale'].mean()
        fixed_pd = fixed_df['pd'].mean()
        dynamic_pd = dynamic_df['pd'].mean()

        report_lines.append(f"- **Fixed strategies**: Mean ALE = {fixed_ale:.2f} m, Mean Pd = {fixed_pd*100:.1f}%")
        report_lines.append(f"- **Dynamic strategies**: Mean ALE = {dynamic_ale:.2f} m, Mean Pd = {dynamic_pd*100:.1f}%")

        if dynamic_ale < fixed_ale:
            improvement = (fixed_ale - dynamic_ale) / fixed_ale * 100
            report_lines.append(f"- **Conclusion**: Dynamic strategies outperform fixed by {improvement:.1f}% in ALE")
        else:
            degradation = (dynamic_ale - fixed_ale) / fixed_ale * 100
            report_lines.append(f"- **Conclusion**: Fixed strategies outperform dynamic by {degradation:.1f}% in ALE")
    report_lines.append("")

    # Per TX Count Analysis
    report_lines.append("## Analysis by TX Count")
    report_lines.append("")

    for tx_count in sorted(tx_count_summaries.keys()):
        summary = tx_count_summaries[tx_count]

        report_lines.append(f"### TX Count = {tx_count}")
        report_lines.append("")

        if len(summary) > 0:
            best = summary.iloc[0]
            report_lines.append(f"**Best Strategy**: `{best['strategy']}` with `{best['selection_method']}` and `{best['whitening_config']}`")
            report_lines.append(f"- Mean ALE: {best['ale_mean']:.2f} m (+-{best['ale_std']:.2f})")
            report_lines.append(f"- Mean Pd: {best['pd_mean']*100:.1f}%")
            report_lines.append(f"- Mean Precision: {best['precision_mean']*100:.1f}%")
            if 'n_est_mean' in best:
                report_lines.append(f"- Mean N (Est/Raw): {best['n_est_mean']:.1f} / {best.get('n_est_raw_mean', 0):.1f}")
            if 'count_error_mean' in best:
                report_lines.append(f"- Mean Count Error: {best['count_error_mean']:.2f}")
            report_lines.append("")

            # Top 5 for this TX count
            report_lines.append("| Strategy | Selection | Whitening Config | Mean ALE | Mean Pd |")
            report_lines.append("|----------|-----------|------------------|----------|---------|")
            for _, row in summary.head(5).iterrows():
                report_lines.append(
                    f"| {row['strategy']} | {row['selection_method']} | {row['whitening_config']} | "
                    f"{row['ale_mean']:.2f} | {row['pd_mean']*100:.1f}% |"
                )
            report_lines.append("")


    # Per TX Set Analysis
    if tx_set_summary is not None:
        report_lines.append("## Analysis by Transmitter Set")
        report_lines.append("")

        for tx_set in sorted(tx_set_summary['transmitters'].unique()):
            set_summary = tx_set_summary[tx_set_summary['transmitters'] == tx_set]

            report_lines.append(f"### Transmitters: {tx_set}")
            report_lines.append("")

            if len(set_summary) > 0:
                best = set_summary.iloc[0]
                report_lines.append(f"**Best Strategy**: `{best['strategy']}` with `{best['selection_method']}`")
                report_lines.append(f"- Mean ALE: {best['ale_mean']:.2f} m")
                report_lines.append(f"- Mean Pd: {best['pd_mean']*100:.1f}%")
                if 'n_est_mean' in best:
                     report_lines.append(f"- Mean N Est: {best['n_est_mean']:.1f}")
                report_lines.append("")

                # Top 3 for this set
                report_lines.append("| Strategy | Selection | Mean ALE | Mean Pd | Precision | Count Err |")
                report_lines.append("|----------|-----------|----------|---------|-----------|-----------|")
                for _, row in set_summary.head(3).iterrows():
                    prec_str = f"{row['precision_mean']*100:.1f}%" if 'precision_mean' in row else "-"
                    err_str = f"{row['count_error_mean']:.2f}" if 'count_error_mean' in row else "-"
                    report_lines.append(
                        f"| {row['strategy']} | {row['selection_method']} | "
                        f"{row['ale_mean']:.2f} | {row['pd_mean']*100:.1f}% | {prec_str} | {err_str} |"
                    )
                report_lines.append("")

    # GLRT Score Analysis
    report_lines.append("## GLRT Score Analysis")
    report_lines.append("")

    glrt_analysis = analyze_glrt_score_correlation(results_df)
    report_lines.append(f"**Summary**: {glrt_analysis['summary']}")
    report_lines.append("")

    if glrt_analysis['overall_correlations']:
        report_lines.append("### Correlations (Overall)")
        report_lines.append("| Metric Pair | Type | Correlation (r) | p-value |")
        report_lines.append("|-------------|------|----------------|---------|")

        corrs = glrt_analysis['overall_correlations']
        if 'final_score_vs_ale_pearson' in corrs:
            c = corrs['final_score_vs_ale_pearson']
            report_lines.append(f"| Final Score vs ALE | Pearson | {c['r']:.3f} | {c['p_value']:.4e} |")
        if 'final_score_vs_ale_spearman' in corrs:
            c = corrs['final_score_vs_ale_spearman']
            report_lines.append(f"| Final Score vs ALE | Spearman | {c['r']:.3f} | {c['p_value']:.4e} |")
        if 'score_reduction_vs_ale' in corrs:
            c = corrs['score_reduction_vs_ale']
            report_lines.append(f"| Score Reduction vs ALE | Pearson | {c['r']:.3f} | {c['p_value']:.4e} |")
        if 'n_iterations_vs_ale' in corrs:
            c = corrs['n_iterations_vs_ale']
            report_lines.append(f"| N Iterations vs ALE | Pearson | {c['r']:.3f} | {c['p_value']:.4e} |")
        report_lines.append("")

        report_lines.append("> **Note**: Strong positive correlation means higher score -> higher ALE (bad). Strong negative means higher score -> lower ALE (good). Ideally we want the score to be a good quality indicator (negative correlation).")
        report_lines.append("")

    if len(glrt_analysis['per_directory_analysis']) > 0:
        report_lines.append("### Best ALE vs Lowest GLRT Score (Per Directory)")
        report_lines.append("")
        report_lines.append(f"**Matching Rate**: {glrt_analysis['matching_rate']*100:.1f}% of directories had their best ALE result from the configuration that produced the lowest GLRT score.")
        report_lines.append("")

        # Show top 5 mismatches (largest ALE difference)
        mismatches = [item for item in glrt_analysis['per_directory_analysis'] if not item['configs_match']]
        mismatches.sort(key=lambda x: x['lowest_glrt_ale'] - x['best_ale'], reverse=True)

        if mismatches:
            report_lines.append("#### Top Mismatches (Where lowest GLRT score misled the most)")
            report_lines.append("| Directory | Best ALE Config | Best ALE (m) | Lowest GLRT Config | Lowest GLRT ALE (m) | Diff (m) |")
            report_lines.append("|-----------|-----------------|--------------|--------------------|---------------------|----------|")

            for item in mismatches[:5]:
                diff = item['lowest_glrt_ale'] - item['best_ale']
                report_lines.append(
                    f"| {item['dir_name']} | {item['best_ale_config']} | {item['best_ale']:.2f} | "
                    f"{item['lowest_glrt_config']} | {item['lowest_glrt_ale']:.2f} | +{diff:.2f} |"
                )
            report_lines.append("")

    # Conclusions
    report_lines.append("## Summary and Recommendations")
    report_lines.append("")

    # Find best strategy per TX count
    best_per_count = {}
    for tx_count, summary in tx_count_summaries.items():
        if len(summary) > 0:
            best_per_count[tx_count] = {
                'strategy': summary.iloc[0]['strategy'],
                'selection': summary.iloc[0]['selection_method'],
                'whitening_config': summary.iloc[0]['whitening_config'],
                'ale': summary.iloc[0]['ale_mean'],
            }

    # Check if same strategy is best across counts
    best_strategies = [v['strategy'] for v in best_per_count.values()]
    best_selections = [v['selection'] for v in best_per_count.values()]
    best_whitening_configs = [v['whitening_config'] for v in best_per_count.values()]

    if len(set(best_strategies)) == 1:
        report_lines.append(f"- **Consistent winner**: `{best_strategies[0]}` performs best across all TX counts")
    else:
        report_lines.append("- **Mixed results**: Different strategies perform best at different TX counts:")
        for tx_count, info in best_per_count.items():
            report_lines.append(f"  - TX={tx_count}: `{info['strategy']}` (ALE={info['ale']:.2f}m)")

    report_lines.append("")

    if len(set(best_selections)) == 1:
        report_lines.append(f"- **Selection method**: `{best_selections[0]}` consistently outperforms")
    else:
        report_lines.append("- **Selection method**: Results vary by TX count")

    report_lines.append("")

    if len(set(best_whitening_configs)) == 1:
        report_lines.append(f"- **Whitening Config**: `{best_whitening_configs[0]}` consistently outperforms")
    else:
        report_lines.append("- **Whitening Config**: Results vary by TX count")

    report_lines.append("")

    # Write report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    return str(report_path)
