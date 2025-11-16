"""Unit conversion utilities for dB and linear scales."""

import numpy as np


def dB_to_lin(pow_dB):
    """
    Convert power from dB to linear scale.

    Parameters
    ----------
    pow_dB : float or array-like
        Power in decibels

    Returns
    -------
    float or ndarray
        Power in linear scale

    Examples
    --------
    >>> dB_to_lin(30)
    1000.0
    >>> dB_to_lin(np.array([0, 10, 20]))
    array([  1.,  10., 100.])
    """
    return 10 ** (np.array(pow_dB) / 10)


def lin_to_dB(pow_lin):
    """
    Convert power from linear to dB scale.

    Parameters
    ----------
    pow_lin : float or array-like
        Power in linear scale

    Returns
    -------
    float or ndarray
        Power in decibels

    Examples
    --------
    >>> lin_to_dB(1000)
    30.0
    >>> lin_to_dB(np.array([1, 10, 100]))
    array([ 0., 10., 20.])

    Notes
    -----
    Handles zero and negative values by replacing them with a small epsilon
    to avoid log(0) errors.
    """
    pow_lin = np.array(pow_lin)
    # Handle zeros and negative values
    pow_lin = np.where(pow_lin <= 0, 1e-12, pow_lin)
    return 10 * np.log10(pow_lin)
