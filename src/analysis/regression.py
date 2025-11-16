"""Nonlinear regression for predicting variance from RSSI."""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


def train_variance_regression(rssi_data, variance_data, degree=3, alpha=0.05):
    """
    Train polynomial regression model to predict variance from RSSI.

    Creates the model shown in Figure 6 of the paper.

    Parameters
    ----------
    rssi_data : array-like
        RSSI values in dB
    variance_data : array-like
        Variance values in dB²
    degree : int, optional
        Degree of polynomial features (default: 3)
    alpha : float, optional
        Lasso regularization strength (default: 0.05)

    Returns
    -------
    sklearn.pipeline.Pipeline
        Trained regression model

    Examples
    --------
    >>> rssi = np.array([-85, -95, -96, -101, -93, -100, -89, -100, -96, -89])
    >>> variance = np.array([21, 18, 26, 10, 23, 13, 15, 16, 22, 23])
    >>> model = train_variance_regression(rssi, variance)
    >>> predictions = model.predict(rssi.reshape(-1, 1))
    >>> len(predictions) == len(rssi)
    True

    Notes
    -----
    Uses polynomial features with Lasso regularization to model the
    nonlinear relationship between RSSI and variance observed in the paper.
    """
    # Create polynomial regression model with Lasso
    model = make_pipeline(
        PolynomialFeatures(degree),
        Lasso(alpha=alpha, max_iter=10000)
    )

    # Reshape data
    X = np.array(rssi_data).reshape(-1, 1)
    y = np.array(variance_data)

    # Train model
    model.fit(X, y)

    return model


def predict_variance_map(rssi_map, model):
    """
    Predict variance at all map locations using regression model.

    Parameters
    ----------
    rssi_map : ndarray
        Map of RSSI values in dB
    model : sklearn.pipeline.Pipeline
        Trained regression model

    Returns
    -------
    ndarray
        Predicted variance map in dB²

    Examples
    --------
    >>> rssi_data = np.array([-85, -95, -96])
    >>> var_data = np.array([21, 18, 26])
    >>> model = train_variance_regression(rssi_data, var_data)
    >>> rssi_map = np.random.randn(10, 10) * 5 - 90
    >>> var_map = predict_variance_map(rssi_map, model)
    >>> var_map.shape == rssi_map.shape
    True

    Notes
    -----
    Negative variance predictions are clipped to 0 as variance
    cannot be negative.
    """
    original_shape = rssi_map.shape

    # Flatten map for prediction
    rssi_flat = rssi_map.ravel().reshape(-1, 1)

    # Predict
    variance_flat = model.predict(rssi_flat)

    # Reshape back to map
    variance_map = variance_flat.reshape(original_shape)

    # Clip negative values to 0
    variance_map = np.maximum(variance_map, 0)

    return variance_map


def evaluate_regression(model, X_test, y_test):
    """
    Evaluate regression model performance.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained model
    X_test : array-like
        Test RSSI values
    y_test : array-like
        Test variance values

    Returns
    -------
    dict
        Dictionary containing:
        - 'mse': Mean squared error
        - 'rmse': Root mean squared error
        - 'r2': R² score
        - 'predictions': Model predictions

    Examples
    --------
    >>> rssi_train = np.array([-85, -95, -96, -101])
    >>> var_train = np.array([21, 18, 26, 10])
    >>> rssi_test = np.array([-90, -100])
    >>> var_test = np.array([19, 12])
    >>> model = train_variance_regression(rssi_train, var_train)
    >>> results = evaluate_regression(model, rssi_test, var_test)
    >>> 'mse' in results
    True
    """
    X_test = np.array(X_test).reshape(-1, 1)
    y_test = np.array(y_test)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred
    }


def combine_band_data(band_data_dict):
    """
    Combine RSSI and variance data from multiple frequency bands.

    Parameters
    ----------
    band_data_dict : dict
        Dictionary mapping band names to {'rssi': [...], 'variance': [...]}

    Returns
    -------
    tuple of ndarray
        (combined_rssi, combined_variance)

    Examples
    --------
    >>> data = {
    ...     'band1': {'rssi': [-85, -90], 'variance': [20, 15]},
    ...     'band2': {'rssi': [-95, -100], 'variance': [18, 12]}
    ... }
    >>> rssi, var = combine_band_data(data)
    >>> len(rssi) == 4
    True
    """
    all_rssi = []
    all_variance = []

    for band_name, data in band_data_dict.items():
        all_rssi.extend(data['rssi'])
        all_variance.extend(data['variance'])

    return np.array(all_rssi), np.array(all_variance)


def fill_negative_variance_with_idw(variance_map, rssi_map, model,
                                     max_distance=200):
    """
    Fill negative variance predictions using IDW interpolation.

    Used in the paper when regression predicts negative variance
    at low or high RSSI regions.

    Parameters
    ----------
    variance_map : ndarray
        Variance map with some negative values
    rssi_map : ndarray
        RSSI map
    model : sklearn.pipeline.Pipeline
        Regression model
    max_distance : float, optional
        Maximum distance for IDW (default: 200)

    Returns
    -------
    ndarray
        Variance map with negative values filled by IDW

    Notes
    -----
    This implements the approach described in Section II.D where
    proxels with negative variance estimates use IDW interpolation instead.
    """
    from ..interpolation.idw import idw_with_distance_threshold

    # Find pixels with positive variance (valid predictions)
    positive_mask = variance_map > 0

    if not np.any(positive_mask):
        return variance_map  # No positive values to interpolate from

    # Get coordinates and values of positive pixels
    rows_pos, cols_pos = np.where(positive_mask)
    values_pos = variance_map[positive_mask]

    # Find pixels needing interpolation
    negative_mask = variance_map <= 0

    if not np.any(negative_mask):
        return variance_map  # No negative values to fix

    # Interpolate using IDW
    interpolated = idw_with_distance_threshold(
        cols_pos, rows_pos, values_pos,
        variance_map.shape, max_distance
    )

    # Fill negative values with interpolated values
    result = variance_map.copy()
    result[negative_mask] = interpolated[negative_mask]

    # If interpolation still gave NaN (no neighbors), keep as 0
    result = np.nan_to_num(result, nan=0.0)

    return result
