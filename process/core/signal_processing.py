"""
Signal processing utilities for RF measurements.
Extracted from read_samples.py, Fading_Analysis.py, and fading_analysis_rot.py
"""
import numpy as np
from typing import Tuple, Dict, List


def find_indices_outside(arr: np.ndarray, a: float, b: float) -> List[int]:
    """
    Find indices of array elements outside the range (a, b).

    Args:
        arr: Input array
        a: Lower bound (exclusive)
        b: Upper bound (exclusive)

    Returns:
        List of indices where arr[i] is not in (a, b)
    """
    indices = [i for i, x in enumerate(arr) if not (a < x < b)]
    return indices


def compute_psd(iq_samples: np.ndarray, sample_rate: float, center_freq: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density (PSD) from IQ samples.

    This function applies a Hamming window, performs FFT, and computes PSD in dB.

    Args:
        iq_samples: Complex IQ samples
        sample_rate: Sampling rate in Hz
        center_freq: Center frequency in Hz

    Returns:
        Tuple of (frequencies, PSD_shifted) where:
            - frequencies: Frequency array centered around center_freq
            - PSD_shifted: PSD in dB (10*log10(W/Hz)), frequency-shifted
    """
    nsamps = len(iq_samples)

    # Apply Hamming window
    x = iq_samples * np.hamming(nsamps)

    # Compute PSD
    PSD = np.abs(np.fft.fft(x)) ** 2 / (nsamps * sample_rate)
    PSD_log = 10.0 * np.log10(PSD)
    PSD_shifted = np.fft.fftshift(PSD_log)

    # Compute frequency array
    f = np.arange(sample_rate / -2.0, sample_rate / 2.0, sample_rate / nsamps)
    f += center_freq

    return f, PSD_shifted


def extract_channel_power(
    frequencies: np.ndarray,
    psd: np.ndarray,
    channel_min: float,
    channel_max: float
) -> float:
    """
    Extract maximum power for a specific RF channel.

    This function masks frequencies outside the channel range and finds the
    maximum PSD value within the channel.

    Args:
        frequencies: Frequency array
        psd: Power spectral density array
        channel_min: Minimum frequency of the channel
        channel_max: Maximum frequency of the channel

    Returns:
        Maximum power (in dB) within the channel
    """
    # Find indices outside the channel range
    idx = find_indices_outside(frequencies, channel_min, channel_max)

    # Create a copy and mask out-of-channel values
    psd_channel = psd.copy()
    psd_channel[idx] = -200  # Very low value to exclude from max

    # Find maximum power in channel
    arg_max = np.argmax(psd_channel)

    return psd_channel[arg_max]


def process_sample_to_powers(
    iq_samples: np.ndarray,
    sample_rate: float,
    center_freq: float,
    rf_channels: Dict[str, Tuple[float, float]]
) -> Dict[str, float]:
    """
    Process IQ samples and extract power for all RF channels.

    This is the main processing function that combines PSD computation
    and channel power extraction for all channels.

    Args:
        iq_samples: Complex IQ samples
        sample_rate: Sampling rate in Hz
        center_freq: Center frequency in Hz
        rf_channels: Dictionary mapping channel names to (min_freq, max_freq) tuples

    Returns:
        Dictionary mapping channel names to power values (dB)
    """
    # Compute PSD
    frequencies, psd = compute_psd(iq_samples, sample_rate, center_freq)

    # Extract power for each channel
    powers = {}
    for channel_name, (channel_min, channel_max) in rf_channels.items():
        power = extract_channel_power(frequencies, psd, channel_min, channel_max)
        powers[channel_name] = power

    return powers
