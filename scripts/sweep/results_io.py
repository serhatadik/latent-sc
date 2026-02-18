"""CSV I/O: append results and save BIC-specific results."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from .constants import DESIRED_COLUMN_ORDER


def append_results_to_csv(results: List[Dict], output_dir: Path):
    """
    Append a batch of results to the main results CSV file.
    Handles header creation if file doesn't exist.
    """
    if not results:
        return

    csv_path = output_dir / 'all_results.csv'
    df = pd.DataFrame(results)

    # Reorder columns to ensure consistent header
    existing_cols = list(df.columns)
    ordered_cols = []

    # Add desired columns if they are present in the dataframe
    for col in DESIRED_COLUMN_ORDER:
        if col in existing_cols:
            ordered_cols.append(col)

    # Add remaining columns
    for col in existing_cols:
        if col not in ordered_cols:
            ordered_cols.append(col)

    # Apply reordering
    df = df[ordered_cols]

    # Check if file exists to determine if header is needed
    file_exists = csv_path.exists()

    # Use append mode 'a'
    # Locking is not implemented here, assuming main thread calls this sequentially
    try:
        df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
    except Exception as e:
        print(f"Warning: Failed to append results to CSV: {e}")


def save_bic_results_csv(results_df: pd.DataFrame, output_dir: Path):
    """
    Save a simplified BIC-only results CSV with key metrics from combinatorial selection.

    The CSV contains:
    - dir_name: Directory name
    - transmitters: TX identifiers
    - seed: Random seed
    - strategy: GLRT strategy
    - whitening_config: Whitening configuration
    - tx_count: True TX count
    - combo_n_tx: Number of TXs in optimal combination
    - combo_ale: Average Localization Error from BIC selection
    - combo_pd: Probability of Detection from BIC selection
    - combo_precision: Precision from BIC selection
    - combo_count_error: |true_tx_count - estimated_tx_count|

    Parameters
    ----------
    results_df : pd.DataFrame
        Full results dataframe
    output_dir : Path
        Output directory
    """
    # Select only BIC-relevant columns (including strategy and whitening config)
    bic_columns = [
        'dir_name',
        'transmitters',
        'seed',
        'strategy',
        'whitening_config',
        'tx_count',
        'combo_n_tx',
        'combo_ale',
        'combo_pd',
        'combo_precision',
        'combo_count_error',
    ]

    # Filter to columns that exist
    available_columns = [col for col in bic_columns if col in results_df.columns]

    if len(available_columns) == 0:
        print("Warning: No BIC columns found in results")
        return

    bic_df = results_df[available_columns].copy()

    # Sort by dir_name, transmitters, seed, strategy, whitening_config
    sort_cols = [col for col in ['dir_name', 'transmitters', 'seed', 'strategy', 'whitening_config'] if col in bic_df.columns]
    if sort_cols:
        bic_df = bic_df.sort_values(sort_cols)

    # Save to CSV
    csv_path = output_dir / 'all_results_bic.csv'
    bic_df.to_csv(csv_path, index=False)
    print(f"BIC results saved to: {csv_path}")
    print(f"  Total rows: {len(bic_df)}")

    # Print summary statistics
    if 'combo_ale' in bic_df.columns:
        valid_ale = bic_df['combo_ale'].dropna()
        if len(valid_ale) > 0:
            print(f"  Mean ALE: {valid_ale.mean():.2f} m")
    if 'combo_pd' in bic_df.columns:
        valid_pd = bic_df['combo_pd'].dropna()
        if len(valid_pd) > 0:
            print(f"  Mean Pd: {valid_pd.mean()*100:.1f}%")
    if 'combo_precision' in bic_df.columns:
        valid_prec = bic_df['combo_precision'].dropna()
        if len(valid_prec) > 0:
            print(f"  Mean Precision: {valid_prec.mean()*100:.1f}%")
    if 'combo_count_error' in bic_df.columns:
        valid_ce = bic_df['combo_count_error'].dropna()
        if len(valid_ce) > 0:
            print(f"  Mean Count Error: {valid_ce.mean():.2f}")

    return bic_df
