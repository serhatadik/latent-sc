"""Spatial visualization functions for georeferenced plots.

This module contains functions to create publication-quality spatial visualizations
for spectrum occupancy analysis, including:
- Transmit power maps (Figure 3a)
- Probability mass function maps (Figure 3b)
- Predicted signal strength maps (Figure 3c)
- Combined power/duty cycle maps (Figures 4, 7)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from ..utils.conversions import lin_to_dB

# Set publication-quality defaults
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14


def plot_transmit_power_map(transmit_power_map, data_points, observed_powers,
                              UTM_lat, UTM_long, band_name, save_path=None):
    """
    Create Figure 3a: Distribution of estimated transmit power.

    Parameters
    ----------
    transmit_power_map : np.ndarray
        2D array of estimated transmit power values (in dBX)
    data_points : np.ndarray
        Array of shape (N, 2) with monitoring station pixel coordinates
    observed_powers : np.ndarray
        Array of observed power values at monitoring stations (in dBX)
    UTM_lat : np.ndarray
        UTM latitude coordinates for ticks
    UTM_long : np.ndarray
        UTM longitude coordinates for ticks
    band_name : str
        Frequency band name (e.g., "3610_3650")
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    x = np.linspace(0, transmit_power_map.shape[1], transmit_power_map.shape[1], endpoint=False)
    y = np.linspace(0, transmit_power_map.shape[0], transmit_power_map.shape[0], endpoint=False)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.gca()

    # Main contour plot
    cf = plt.contourf(X, Y, transmit_power_map, 100, cmap='hot')

    # Right colorbar for estimated transmit power
    cbar = plt.colorbar(cf, label='Estimated Transmit Power [dBX]')
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(label='Estimated Transmit Power [dBX]', size=18)

    # Scatter plot for observed data points
    scatter = plt.scatter(data_points[:, 0], data_points[:, 1],
                         c=observed_powers, s=150, edgecolor='green',
                         linewidth=2, cmap='hot', label='Spectrum Monitoring Locations')

    # Left colorbar for observed signal strength
    scatter_cbar = plt.colorbar(scatter, label='Observed Signal Strength [dBX]', location='left')
    scatter_cbar.ax.tick_params(labelsize=18)
    scatter_cbar.set_label(label='Observed Signal Strength [dBX]', size=18)

    plt.legend()

    # Set UTM ticks
    interval = max(1, len(UTM_lat) // 5)
    tick_values = list(range(0, len(UTM_lat), interval))
    tick_labels = ['{:.1f}'.format(lat) for lat in UTM_lat[::interval]]
    plt.xticks(ticks=tick_values, labels=tick_labels, fontsize=14, rotation=0)

    interval = max(1, len(UTM_long) // 5)
    tick_values = list(range(0, len(UTM_long), interval))
    tick_labels = ['{:.1f}'.format(lat) for lat in UTM_long[::interval]]
    plt.yticks(ticks=[0] + tick_values[1:], labels=[""] + tick_labels[1:], fontsize=14, rotation=90)

    # Adjust colorbar positions
    cbar.ax.set_position([0.77, 0.1, 0.04, 0.8])
    scatter_cbar.ax.set_position([0.18, 0.1, 0.04, 0.8])

    plt.xlabel('UTM$_E$ [m]', fontsize=18, labelpad=10)
    plt.ylabel('UTM$_N$ [m]', fontsize=18, labelpad=10)
    plt.title('Estimated Transmit Power @ ' + band_name.split('-')[0] + '-' +
              band_name.split('-')[1] + ' MHz', fontsize=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_pmf_map(pmf, data_points, observed_powers, UTM_lat, UTM_long,
                 band_name, save_path=None):
    """
    Create Figure 3b: 2D probability mass function of transmitter location.

    Parameters
    ----------
    pmf : np.ndarray
        2D probability mass function
    data_points : np.ndarray
        Array of shape (N, 2) with monitoring station pixel coordinates
    observed_powers : np.ndarray
        Array of observed power values at monitoring stations
    UTM_lat : np.ndarray
        UTM latitude coordinates for ticks
    UTM_long : np.ndarray
        UTM longitude coordinates for ticks
    band_name : str
        Frequency band name
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    x = np.linspace(0, pmf.shape[1], pmf.shape[1], endpoint=False)
    y = np.linspace(0, pmf.shape[0], pmf.shape[0], endpoint=False)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.gca()

    # Main contour plot
    cf = plt.contourf(X, Y, pmf, 100, cmap='hot')

    # Right colorbar for probability
    cbar = plt.colorbar(cf, label='Probability Mass')
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(label='Probability Mass', size=18)

    # Scatter plot for observed data points
    scatter = plt.scatter(data_points[:, 0], data_points[:, 1],
                         c=observed_powers, s=150, edgecolor='green',
                         linewidth=2, cmap='hot', label='Spectrum Monitoring Locations')

    # Left colorbar for observed signal strength
    scatter_cbar = plt.colorbar(scatter, label='Observed Signal Strength [dBX]', location='left')
    scatter_cbar.ax.tick_params(labelsize=18)
    scatter_cbar.set_label(label='Observed Signal Strength [dBX]', size=18)

    plt.legend()

    # Set UTM ticks
    interval = max(1, len(UTM_lat) // 5)
    tick_values = list(range(0, len(UTM_lat), interval))
    tick_labels = ['{:.1f}'.format(lat) for lat in UTM_lat[::interval]]
    plt.xticks(ticks=tick_values, labels=tick_labels, fontsize=14, rotation=0)

    interval = max(1, len(UTM_long) // 5)
    tick_values = list(range(0, len(UTM_long), interval))
    tick_labels = ['{:.1f}'.format(lat) for lat in UTM_long[::interval]]
    plt.yticks(ticks=[0] + tick_values[1:], labels=[""] + tick_labels[1:], fontsize=14, rotation=90)

    # Adjust colorbar positions
    cbar.ax.set_position([0.77, 0.1, 0.04, 0.8])
    scatter_cbar.ax.set_position([0.18, 0.1, 0.04, 0.8])

    plt.xlabel('UTM$_E$ [m]', fontsize=18, labelpad=10)
    plt.ylabel('UTM$_N$ [m]', fontsize=18, labelpad=10)
    plt.title('2D PMF of Transmitter Location @ ' + band_name.split('-')[0] + '-' +
              band_name.split('-')[1] + ' MHz', fontsize=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_signal_estimates_map(signal_estimates, data_points, observed_powers,
                               UTM_lat, UTM_long, band_name, save_path=None):
    """
    Create Figure 3c: 2D predictions of signal strength.

    Parameters
    ----------
    signal_estimates : np.ndarray
        2D array of estimated signal strength (linear scale)
    data_points : np.ndarray
        Array of shape (N, 2) with monitoring station pixel coordinates
    observed_powers : np.ndarray
        Array of observed power values at monitoring stations (in dBX)
    UTM_lat : np.ndarray
        UTM latitude coordinates for ticks
    UTM_long : np.ndarray
        UTM longitude coordinates for ticks
    band_name : str
        Frequency band name
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    # Convert to dB for visualization
    signal_estimates_dB = lin_to_dB(signal_estimates)

    x = np.linspace(0, signal_estimates.shape[1], signal_estimates.shape[1], endpoint=False)
    y = np.linspace(0, signal_estimates.shape[0], signal_estimates.shape[0], endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Determine common color scale
    min_strength = min(signal_estimates_dB.min(), observed_powers.min())
    max_strength = max(signal_estimates_dB.max(), observed_powers.max())

    colorbar_ticks = np.linspace(min_strength, max_strength, num=7)
    colorbar_tick_labels = ['{:.1f}'.format(tick) for tick in colorbar_ticks]

    fig = plt.figure(figsize=(13, 8))
    ax = fig.gca()

    # Main contour plot
    cf = plt.contourf(X, Y, signal_estimates_dB, 100, cmap='hot',
                     vmin=min_strength, vmax=max_strength)

    # Right colorbar for predicted signal strength
    cbar = plt.colorbar(cf, label='Predicted Signal Strength [dBX]')
    cbar.ax.tick_params(labelsize=18)
    cbar.set_ticks(colorbar_ticks)
    cbar.set_label(label='Predicted Signal Strength [dBX]', size=18)
    cbar.set_ticklabels(colorbar_tick_labels)

    # Scatter plot for observed data points
    scatter = plt.scatter(data_points[:, 0], data_points[:, 1],
                         c=observed_powers, s=150, edgecolor='green',
                         linewidth=2, cmap='hot', label='Spectrum Monitoring Locations',
                         vmin=min_strength, vmax=max_strength)

    # Left colorbar for observed signal strength
    scatter_cbar = plt.colorbar(scatter, label='Observed Signal Strength [dBX]', location='left')
    scatter_cbar.ax.tick_params(labelsize=18)
    scatter_cbar.set_ticks(colorbar_ticks)
    scatter_cbar.set_label(label='Observed Signal Strength [dBX]', size=18)
    scatter_cbar.set_ticklabels(colorbar_tick_labels)

    plt.legend()

    # Set UTM ticks
    interval = max(1, len(UTM_lat) // 5)
    tick_values = list(range(0, len(UTM_lat), interval))
    tick_labels = ['{:.1f}'.format(lat) for lat in UTM_lat[::interval]]
    plt.xticks(ticks=tick_values, labels=tick_labels, fontsize=14, rotation=0)

    interval = max(1, len(UTM_long) // 5)
    tick_values = list(range(0, len(UTM_long), interval))
    tick_labels = ['{:.1f}'.format(lat) for lat in UTM_long[::interval]]
    plt.yticks(ticks=[0] + tick_values[1:], labels=[""] + tick_labels[1:], fontsize=14, rotation=90)

    # Adjust colorbar positions
    cbar.ax.set_position([0.77, 0.1, 0.04, 0.8])
    scatter_cbar.ax.set_position([0.18, 0.1, 0.04, 0.8])

    plt.xlabel('UTM$_E$ [m]', fontsize=18, labelpad=10)
    plt.ylabel('UTM$_N$ [m]', fontsize=18, labelpad=10)
    plt.title('2D Predictions of Signal Strength @ ' + band_name.split('-')[0] + '-' +
              band_name.split('-')[1] + ' MHz', fontsize=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def draw_aggregated_squares(ax, aggregated_power, aggregated_duty_cycle, block_size):
    """
    Helper function to draw aggregated squares for power/duty cycle visualization.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw on
    aggregated_power : np.ndarray
        Aggregated power values (in dB)
    aggregated_duty_cycle : np.ndarray
        Aggregated duty cycle values (0-100%)
    block_size : int
        Size of aggregation blocks in pixels
    """
    for i in range(aggregated_power.shape[0]):
        for j in range(aggregated_power.shape[1]):
            power_val = aggregated_power[i, j]
            duty_cycle_val = aggregated_duty_cycle[i, j]

            # Calculate square size based on duty cycle
            square_size = block_size * (duty_cycle_val / 100)
            offset = (block_size - square_size) / 2

            # Position
            x = j * block_size + offset
            y = i * block_size + offset

            # Color based on power (normalize to colormap range)
            color_val = (power_val + 120) / 40  # Adjust normalization as needed
            color = plt.cm.viridis(np.clip(color_val, 0, 1))

            # Draw rectangle
            rect = Rectangle((x, y), square_size, square_size,
                           facecolor=color, edgecolor='none')
            ax.add_patch(rect)


def aggregate_data(data, block_size):
    """
    Aggregate 2D data into blocks by averaging.

    Parameters
    ----------
    data : np.ndarray
        2D array to aggregate
    block_size : int
        Size of blocks for aggregation

    Returns
    -------
    aggregated : np.ndarray
        Aggregated data
    """
    h, w = data.shape
    h_blocks = h // block_size
    w_blocks = w // block_size

    aggregated = np.zeros((h_blocks, w_blocks))

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = data[i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size]
            aggregated[i, j] = np.mean(block)

    return aggregated


def plot_power_duty_cycle_combined(power_map, duty_cycle_map, data_points,
                                   buildings_map, UTM_lat, UTM_long,
                                   band_name, block_size=5, save_path=None):
    """
    Create Figure 4/7: Combined power and duty cycle visualization.

    Power is shown as color (viridis colormap), duty cycle as square size.
    Buildings are shown as gray background.

    Parameters
    ----------
    power_map : np.ndarray
        2D array of power values (in dB)
    duty_cycle_map : np.ndarray
        2D array of duty cycle values (0-100%)
    data_points : np.ndarray
        Array of shape (N, 2) with monitoring station pixel coordinates
    buildings_map : np.ndarray
        2D array of building heights or presence
    UTM_lat : np.ndarray
        UTM latitude coordinates for ticks
    UTM_long : np.ndarray
        UTM longitude coordinates for ticks
    band_name : str
        Frequency band name
    block_size : int
        Size of aggregation blocks (default: 5)
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    x = np.linspace(0, power_map.shape[1], power_map.shape[1], endpoint=False)
    y = np.linspace(0, power_map.shape[0], power_map.shape[0], endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Aggregate data
    aggregated_power = aggregate_data(power_map, block_size)
    aggregated_duty_cycle = aggregate_data(duty_cycle_map, block_size)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.gca()

    # Plot buildings as background
    plt.contourf(X, Y, buildings_map, 100, cmap='gist_gray', alpha=0.5)

    # Draw aggregated squares
    draw_aggregated_squares(ax, aggregated_power, aggregated_duty_cycle, block_size)

    # Scatter plot for monitoring locations
    scatter = plt.scatter(data_points[:, 0], data_points[:, 1],
                         c='purple', s=40, edgecolors='white',
                         linewidths=2, label='Spectrum Monitoring Locations')

    # Set UTM ticks
    interval = max(1, len(UTM_lat) // 5)
    tick_values = list(range(0, len(UTM_lat), interval))
    tick_labels = ['{:.1f}'.format(lat) for lat in UTM_lat[::interval]]
    plt.xticks(ticks=tick_values, labels=tick_labels, fontsize=14, rotation=0)

    interval = max(1, len(UTM_long) // 5)
    tick_values = list(range(0, len(UTM_long), interval))
    tick_labels = ['{:.1f}'.format(lat) for lat in UTM_long[::interval]]
    plt.yticks(ticks=[0] + tick_values[1:], labels=[""] + tick_labels[1:], fontsize=14, rotation=90)

    ax.set_xlim([0, power_map.shape[1]])
    ax.set_ylim([0, power_map.shape[0]])

    plt.xlabel('UTM$_E$ [m]', fontsize=18, labelpad=10)
    plt.ylabel('UTM$_N$ [m]', fontsize=18, labelpad=10)
    plt.title('Spatial Spectrum Occupancy @ ' + band_name.split('-')[0] + '-' +
              band_name.split('-')[1] + ' MHz', fontsize=20)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_variation_confidence_combined(variation_map, confidence_map, data_points,
                                       UTM_lat, UTM_long, band_name, save_path=None):
    """
    Create Figure 7: Signal variation and confidence level maps.

    Parameters
    ----------
    variation_map : np.ndarray
        2D array of signal variation values (in dB²)
    confidence_map : np.ndarray
        2D array of confidence level values (0-1)
    data_points : np.ndarray
        Array of shape (N, 2) with monitoring station pixel coordinates
    UTM_lat : np.ndarray
        UTM latitude coordinates for ticks
    UTM_long : np.ndarray
        UTM longitude coordinates for ticks
    band_name : str
        Frequency band name
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig, axes : matplotlib figure and axis objects
    """
    fig, axes = plt.subplots(1, 2, figsize=(26, 8))

    x = np.linspace(0, variation_map.shape[1], variation_map.shape[1], endpoint=False)
    y = np.linspace(0, variation_map.shape[0], variation_map.shape[0], endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Plot 1: Signal Variation
    ax1 = axes[0]
    cf1 = ax1.contourf(X, Y, variation_map, 100, cmap='viridis')
    cbar1 = plt.colorbar(cf1, ax=ax1)
    cbar1.set_label('Signal Variation [dB²]', fontsize=18)
    cbar1.ax.tick_params(labelsize=14)

    ax1.scatter(data_points[:, 0], data_points[:, 1],
               c='red', s=100, edgecolors='white', linewidths=2,
               label='Monitoring Locations')

    ax1.set_xlabel('UTM$_E$ [m]', fontsize=18)
    ax1.set_ylabel('UTM$_N$ [m]', fontsize=18)
    ax1.set_title('Signal Variation @ ' + band_name.split('-')[0] + '-' +
                  band_name.split('-')[1] + ' MHz', fontsize=20)
    ax1.legend()

    # Plot 2: Confidence Level
    ax2 = axes[1]
    cf2 = ax2.contourf(X, Y, confidence_map, 100, cmap='hot')
    cbar2 = plt.colorbar(cf2, ax=ax2)
    cbar2.set_label('Confidence Level', fontsize=18)
    cbar2.ax.tick_params(labelsize=14)

    ax2.scatter(data_points[:, 0], data_points[:, 1],
               c='green', s=100, edgecolors='white', linewidths=2,
               label='Monitoring Locations')

    ax2.set_xlabel('UTM$_E$ [m]', fontsize=18)
    ax2.set_ylabel('UTM$_N$ [m]', fontsize=18)
    ax2.set_title('Confidence Level @ ' + band_name.split('-')[0] + '-' +
                  band_name.split('-')[1] + ' MHz', fontsize=20)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes
