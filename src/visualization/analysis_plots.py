"""Analysis visualization functions.

This module contains functions for creating analysis plots:
- Power histograms with thresholds (Figure 2)
- Regression plots (Figure 6)
- Correlation heatmaps (Table III)
- MSE and error analysis plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Set publication-quality defaults
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14


def plot_power_histogram(power_data, threshold, band_start, band_end,
                         monitor_name, bins=20, save_path=None):
    """
    Create Figure 2: Power level histogram with occupancy threshold.

    Parameters
    ----------
    power_data : np.ndarray or pd.Series
        Power measurements in dB
    threshold : float
        Occupancy threshold in dB
    band_start : float
        Start frequency of band in MHz
    band_end : float
        End frequency of band in MHz
    monitor_name : str
        Name of monitoring station
    bins : int
        Number of histogram bins (default: 20)
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram
    counts, bin_edges, patches = ax.hist(power_data, bins=bins, color='skyblue',
                                        edgecolor='black', alpha=0.7, label='Power Distribution')

    # Add threshold line
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
              label=f'Threshold = {threshold:.1f} dB')

    # Calculate duty cycle
    occupied_count = np.sum(power_data > threshold)
    total_count = len(power_data)
    duty_cycle = (occupied_count / total_count) * 100

    # Add text box with statistics
    textstr = f'Duty Cycle: {duty_cycle:.2f}%\nOccupied: {occupied_count:,}\nTotal: {total_count:,}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=props)

    ax.set_xlabel('Power (dB)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'Power Distribution @ {band_start}-{band_end} MHz\n{monitor_name}',
                fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_frequency_vs_power_scatter(frequency_data, power_data, threshold,
                                   band_start, band_end, monitor_name,
                                   save_path=None):
    """
    Create scatter plot of frequency vs power with threshold line.

    Parameters
    ----------
    frequency_data : np.ndarray or pd.Series
        Frequency measurements in MHz
    power_data : np.ndarray or pd.Series
        Power measurements in dB
    threshold : float
        Occupancy threshold in dB
    band_start : float
        Start frequency of band in MHz
    band_end : float
        End frequency of band in MHz
    monitor_name : str
        Name of monitoring station
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    # Create scatter plot
    ax.scatter(frequency_data, power_data, marker='*', s=0.3, alpha=0.5)

    # Add threshold line
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2,
              label=f'Threshold = {threshold:.1f} dB')

    ax.set_xlabel('Frequency (MHz)', fontsize=16)
    ax.set_ylabel('Power (dB)', fontsize=16)
    ax.set_title(f'Spectrum Occupancy @ {monitor_name}', fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_variance_regression(rssi_data, variance_data, model, band_name,
                             save_path=None):
    """
    Create Figure 6: Variance vs RSSI regression plot.

    Shows the nonlinear relationship between RSSI and signal variation
    with polynomial regression fit.

    Parameters
    ----------
    rssi_data : np.ndarray
        RSSI values (observed signal strength) in dB
    variance_data : np.ndarray
        Signal variation values in dB²
    model : sklearn.pipeline.Pipeline
        Trained regression model (PolynomialFeatures + Lasso)
    band_name : str
        Frequency band name
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot actual data points
    ax.scatter(rssi_data, variance_data, s=100, alpha=0.7, edgecolors='black',
              label='Observed Data', zorder=3)

    # Generate smooth curve for fitted model
    rssi_range = np.linspace(rssi_data.min() - 5, rssi_data.max() + 5, 200)
    variance_pred = model.predict(rssi_range.reshape(-1, 1))

    # Only plot positive predictions
    valid_mask = variance_pred >= 0
    ax.plot(rssi_range[valid_mask], variance_pred[valid_mask], 'r-',
           linewidth=2, label='Polynomial Regression Fit', zorder=2)

    # Add gridlines
    ax.grid(True, alpha=0.3, zorder=1)

    # Labels and title
    ax.set_xlabel('RSSI (dB)', fontsize=14)
    ax.set_ylabel('Signal Variation (dB²)', fontsize=14)
    ax.set_title(f'Variance Prediction Model @ {band_name} MHz', fontsize=16)
    ax.legend(fontsize=12)

    # Add R² score if available
    from sklearn.metrics import r2_score
    variance_pred_actual = model.predict(rssi_data.reshape(-1, 1))
    r2 = r2_score(variance_data, variance_pred_actual)

    textstr = f'R² = {r2:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=props)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_correlation_heatmap(rssi_data, variance_data, duty_cycle_data,
                             band_name, save_path=None):
    """
    Create correlation heatmap for Table III metrics.

    Parameters
    ----------
    rssi_data : np.ndarray
        RSSI values in dB
    variance_data : np.ndarray
        Signal variation values in dB²
    duty_cycle_data : np.ndarray
        Duty cycle values in %
    band_name : str
        Frequency band name
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    import pandas as pd

    # Create DataFrame
    df = pd.DataFrame({
        'RSSI (dB)': rssi_data,
        'Variance (dB²)': variance_data,
        'Duty Cycle (%)': duty_cycle_data
    })

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
               center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
               ax=ax, vmin=-1, vmax=1)

    ax.set_title(f'Correlation Matrix @ {band_name} MHz', fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_mse_comparison(mse_values, monitor_names, band_name, save_path=None):
    """
    Plot MSE values across monitoring stations.

    Parameters
    ----------
    mse_values : np.ndarray
        MSE values for each monitoring station
    monitor_names : list
        Names of monitoring stations
    band_name : str
        Frequency band name
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar plot
    x_pos = np.arange(len(monitor_names))
    bars = ax.bar(x_pos, mse_values, color='steelblue', alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, val in zip(bars, mse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # Add mean line
    mean_mse = np.mean(mse_values)
    ax.axhline(mean_mse, color='red', linestyle='--', linewidth=2,
              label=f'Mean MSE = {mean_mse:.2f}')

    ax.set_ylabel('Mean Squared Error (dB²)', fontsize=14)
    ax.set_xlabel('Monitoring Station', fontsize=14)
    ax.set_title(f'Prediction MSE by Station @ {band_name} MHz', fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(monitor_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_prediction_vs_actual(predicted, actual, monitor_names, band_name,
                              save_path=None):
    """
    Plot predicted vs actual signal strength values.

    Parameters
    ----------
    predicted : np.ndarray
        Predicted signal strength values in dB
    actual : np.ndarray
        Actual observed signal strength values in dB
    monitor_names : list
        Names of monitoring stations
    band_name : str
        Frequency band name
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(actual, predicted, s=150, alpha=0.7, edgecolors='black')

    # Add perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
           label='Perfect Prediction')

    # Add labels for each point
    for i, name in enumerate(monitor_names):
        ax.annotate(name, (actual[i], predicted[i]), fontsize=9,
                   xytext=(5, 5), textcoords='offset points')

    # Calculate RMSE
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    textstr = f'RMSE = {rmse:.2f} dB'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=props)

    ax.set_xlabel('Actual Signal Strength (dB)', fontsize=14)
    ax.set_ylabel('Predicted Signal Strength (dB)', fontsize=14)
    ax.set_title(f'Prediction Accuracy @ {band_name} MHz', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax
