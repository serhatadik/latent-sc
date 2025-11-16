"""Temporal visualization functions for spectrum occupancy analysis.

This module contains functions to create visualizations for temporal analysis:
- Time-of-day comparison plots (Figure 5)
- Seasonal comparison plots (Figure 5)
- Temporal metric bar charts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Set publication-quality defaults
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14
sns.set_theme()


def plot_temporal_duty_cycles(temporal_metrics, monitor_name, band_name, save_path=None):
    """
    Plot duty cycles across time-of-day periods and seasons.

    Parameters
    ----------
    temporal_metrics : dict
        Dictionary with 'time_of_day' and 'season' keys containing metrics
    monitor_name : str
        Name of the monitoring station
    band_name : str
        Frequency band name
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, axes : matplotlib figure and axis objects
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Time of Day
    ax1 = axes[0]
    time_periods = list(temporal_metrics['time_of_day'].keys())
    duty_cycles_tod = [temporal_metrics['time_of_day'][period]['duty_cycle']
                       for period in time_periods]

    colors_tod = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars1 = ax1.bar(time_periods, duty_cycles_tod, color=colors_tod, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Duty Cycle (%)', fontsize=14)
    ax1.set_xlabel('Time of Day', fontsize=14)
    ax1.set_title(f'Duty Cycle by Time of Day\n{monitor_name} @ {band_name} MHz', fontsize=16)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, duty_cycles_tod):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12)

    # Plot 2: Seasons
    ax2 = axes[1]
    seasons = list(temporal_metrics['season'].keys())
    duty_cycles_season = [temporal_metrics['season'][season]['duty_cycle']
                          for season in seasons]

    colors_season = ['#95E1D3', '#F38181', '#AA96DA', '#FCBAD3']
    bars2 = ax2.bar(seasons, duty_cycles_season, color=colors_season, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Duty Cycle (%)', fontsize=14)
    ax2.set_xlabel('Season', fontsize=14)
    ax2.set_title(f'Duty Cycle by Season\n{monitor_name} @ {band_name} MHz', fontsize=16)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars2, duty_cycles_season):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


def plot_temporal_power(temporal_metrics, monitor_name, band_name, save_path=None):
    """
    Plot average occupied power across time-of-day periods and seasons.

    Parameters
    ----------
    temporal_metrics : dict
        Dictionary with 'time_of_day' and 'season' keys containing metrics
    monitor_name : str
        Name of the monitoring station
    band_name : str
        Frequency band name
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, axes : matplotlib figure and axis objects
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Time of Day
    ax1 = axes[0]
    time_periods = list(temporal_metrics['time_of_day'].keys())
    avg_power_tod = [temporal_metrics['time_of_day'][period]['avg_power_occupied']
                     for period in time_periods]

    colors_tod = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars1 = ax1.bar(time_periods, avg_power_tod, color=colors_tod, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Average Power When Occupied (dBm)', fontsize=14)
    ax1.set_xlabel('Time of Day', fontsize=14)
    ax1.set_title(f'Average Power by Time of Day\n{monitor_name} @ {band_name} MHz', fontsize=16)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, avg_power_tod):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=12)

    # Plot 2: Seasons
    ax2 = axes[1]
    seasons = list(temporal_metrics['season'].keys())
    avg_power_season = [temporal_metrics['season'][season]['avg_power_occupied']
                        for season in seasons]

    colors_season = ['#95E1D3', '#F38181', '#AA96DA', '#FCBAD3']
    bars2 = ax2.bar(seasons, avg_power_season, color=colors_season, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Average Power When Occupied (dBm)', fontsize=14)
    ax2.set_xlabel('Season', fontsize=14)
    ax2.set_title(f'Average Power by Season\n{monitor_name} @ {band_name} MHz', fontsize=16)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars2, avg_power_season):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


def plot_temporal_variance(temporal_metrics, monitor_name, band_name, save_path=None):
    """
    Plot signal variation across time-of-day periods and seasons.

    Parameters
    ----------
    temporal_metrics : dict
        Dictionary with 'time_of_day' and 'season' keys containing metrics
    monitor_name : str
        Name of the monitoring station
    band_name : str
        Frequency band name
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, axes : matplotlib figure and axis objects
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Time of Day
    ax1 = axes[0]
    time_periods = list(temporal_metrics['time_of_day'].keys())
    variance_tod = [temporal_metrics['time_of_day'][period]['signal_variation']
                    for period in time_periods]

    colors_tod = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars1 = ax1.bar(time_periods, variance_tod, color=colors_tod, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Signal Variation (dB²)', fontsize=14)
    ax1.set_xlabel('Time of Day', fontsize=14)
    ax1.set_title(f'Signal Variation by Time of Day\n{monitor_name} @ {band_name} MHz', fontsize=16)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, variance_tod):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12)

    # Plot 2: Seasons
    ax2 = axes[1]
    seasons = list(temporal_metrics['season'].keys())
    variance_season = [temporal_metrics['season'][season]['signal_variation']
                       for season in seasons]

    colors_season = ['#95E1D3', '#F38181', '#AA96DA', '#FCBAD3']
    bars2 = ax2.bar(seasons, variance_season, color=colors_season, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Signal Variation (dB²)', fontsize=14)
    ax2.set_xlabel('Season', fontsize=14)
    ax2.set_title(f'Signal Variation by Season\n{monitor_name} @ {band_name} MHz', fontsize=16)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars2, variance_season):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


def plot_all_temporal_metrics(temporal_metrics, monitor_name, band_name, save_path=None):
    """
    Create comprehensive temporal analysis plot (Figure 5 in paper).

    Combines all three metrics (duty cycle, power, variation) in a single figure.

    Parameters
    ----------
    temporal_metrics : dict
        Dictionary with 'time_of_day' and 'season' keys containing metrics
    monitor_name : str
        Name of the monitoring station
    band_name : str
        Frequency band name
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, axes : matplotlib figure and axis objects
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # Extract data
    time_periods = list(temporal_metrics['time_of_day'].keys())
    seasons = list(temporal_metrics['season'].keys())

    duty_cycles_tod = [temporal_metrics['time_of_day'][period]['duty_cycle']
                       for period in time_periods]
    duty_cycles_season = [temporal_metrics['season'][season]['duty_cycle']
                          for season in seasons]

    avg_power_tod = [temporal_metrics['time_of_day'][period]['avg_power_occupied']
                     for period in time_periods]
    avg_power_season = [temporal_metrics['season'][season]['avg_power_occupied']
                        for season in seasons]

    variance_tod = [temporal_metrics['time_of_day'][period]['signal_variation']
                    for period in time_periods]
    variance_season = [temporal_metrics['season'][season]['signal_variation']
                       for season in seasons]

    colors_tod = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    colors_season = ['#95E1D3', '#F38181', '#AA96DA', '#FCBAD3']

    # Row 1: Duty Cycle
    axes[0, 0].bar(time_periods, duty_cycles_tod, color=colors_tod, alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Duty Cycle (%)', fontsize=14)
    axes[0, 0].set_title('Duty Cycle by Time of Day', fontsize=14)
    axes[0, 0].grid(axis='y', alpha=0.3)

    axes[0, 1].bar(seasons, duty_cycles_season, color=colors_season, alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('Duty Cycle (%)', fontsize=14)
    axes[0, 1].set_title('Duty Cycle by Season', fontsize=14)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Row 2: Average Power
    axes[1, 0].bar(time_periods, avg_power_tod, color=colors_tod, alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Avg Power (dBm)', fontsize=14)
    axes[1, 0].set_title('Average Power by Time of Day', fontsize=14)
    axes[1, 0].grid(axis='y', alpha=0.3)

    axes[1, 1].bar(seasons, avg_power_season, color=colors_season, alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Avg Power (dBm)', fontsize=14)
    axes[1, 1].set_title('Average Power by Season', fontsize=14)
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Row 3: Signal Variation
    axes[2, 0].bar(time_periods, variance_tod, color=colors_tod, alpha=0.7, edgecolor='black')
    axes[2, 0].set_ylabel('Variation (dB²)', fontsize=14)
    axes[2, 0].set_xlabel('Time of Day', fontsize=14)
    axes[2, 0].set_title('Signal Variation by Time of Day', fontsize=14)
    axes[2, 0].grid(axis='y', alpha=0.3)

    axes[2, 1].bar(seasons, variance_season, color=colors_season, alpha=0.7, edgecolor='black')
    axes[2, 1].set_ylabel('Variation (dB²)', fontsize=14)
    axes[2, 1].set_xlabel('Season', fontsize=14)
    axes[2, 1].set_title('Signal Variation by Season', fontsize=14)
    axes[2, 1].grid(axis='y', alpha=0.3)

    fig.suptitle(f'Temporal Analysis: {monitor_name} @ {band_name} MHz',
                fontsize=18, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes
