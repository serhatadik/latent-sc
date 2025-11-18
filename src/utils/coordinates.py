"""Coordinate system utilities and distance calculations."""

import numpy as np
from pyproj import Transformer


def serialize_index(row, col, num_cols):
    """
    Convert 2D matrix indices to 1D serialized index.

    Parameters
    ----------
    row : int
        Row index
    col : int
        Column index
    num_cols : int
        Number of columns in the matrix

    Returns
    -------
    int
        Serialized 1D index

    Examples
    --------
    >>> serialize_index(2, 3, 10)
    23
    """
    return row * num_cols + col


def deserialize_index(serialized_index, num_cols):
    """
    Convert 1D serialized index back to 2D matrix indices.

    Parameters
    ----------
    serialized_index : int
        Serialized 1D index
    num_cols : int
        Number of columns in the matrix

    Returns
    -------
    tuple of int
        (row, col) indices

    Examples
    --------
    >>> deserialize_index(23, 10)
    (2, 3)
    """
    row = serialized_index // num_cols
    col = serialized_index % num_cols
    return row, col


def euclidean_distance(point1, point2, scale=1.0):
    """
    Calculate Euclidean distance between two points.

    Parameters
    ----------
    point1 : array-like
        First point coordinates (x, y) or (row, col)
    point2 : array-like
        Second point coordinates (x, y) or (row, col)
    scale : float, optional
        Scaling factor to convert indices to meters (default: 1.0)

    Returns
    -------
    float
        Euclidean distance in scaled units

    Examples
    --------
    >>> euclidean_distance([0, 0], [3, 4])
    5.0
    >>> euclidean_distance([0, 0], [3, 4], scale=5)
    25.0
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.sqrt(np.sum((point2 - point1) ** 2)) * scale
    # Minimum distance of 1 to avoid division by zero
    return max(distance, 1.0)


def compute_distance_matrix(locations, grid_points, scale=1.0):
    """
    Compute distance matrix between all locations and grid points.

    Parameters
    ----------
    locations : ndarray of shape (n_locations, 2)
        Coordinates of known locations
    grid_points : ndarray of shape (n_points, 2)
        Coordinates of grid points
    scale : float, optional
        Scaling factor for distances (default: 1.0)

    Returns
    -------
    ndarray of shape (n_locations, n_points)
        Distance matrix where element [i, j] is distance from location i to grid point j

    Examples
    --------
    >>> locs = np.array([[0, 0], [1, 1]])
    >>> grid = np.array([[0, 0], [2, 2], [1, 0]])
    >>> distances = compute_distance_matrix(locs, grid)
    >>> distances.shape
    (2, 3)
    """
    n_locations = len(locations)
    n_points = len(grid_points)
    distances = np.zeros((n_locations, n_points))

    for i, loc in enumerate(locations):
        for j, point in enumerate(grid_points):
            distances[i, j] = euclidean_distance(loc, point, scale)

    return distances


def create_grid(map_shape, resolution=1):
    """
    Create coordinate grid for a map.

    Parameters
    ----------
    map_shape : tuple of int
        (height, width) of the map in pixels
    resolution : float, optional
        Grid resolution in meters per pixel (default: 1)

    Returns
    -------
    tuple of ndarray
        (X, Y) meshgrid arrays

    Examples
    --------
    >>> X, Y = create_grid((100, 150), resolution=5)
    >>> X.shape, Y.shape
    ((100, 150), (100, 150))
    """
    height, width = map_shape
    x = np.arange(0, width) * resolution
    y = np.arange(0, height) * resolution
    X, Y = np.meshgrid(x, y)
    return X, Y


def compute_pairwise_distances(points, scale=1.0):
    """
    Compute pairwise distances between all points.

    Parameters
    ----------
    points : ndarray of shape (n_points, 2)
        Array of point coordinates
    scale : float, optional
        Scaling factor (default: 1.0)

    Returns
    -------
    ndarray of shape (n_points, n_points)
        Symmetric matrix of pairwise distances

    Examples
    --------
    >>> points = np.array([[0, 0], [1, 0], [0, 1]])
    >>> dists = compute_pairwise_distances(points)
    >>> dists[0, 1]  # Distance from point 0 to point 1
    1.0
    """
    n_points = len(points)
    distances = np.zeros((n_points, n_points))

    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = euclidean_distance(points[i], points[j], scale)
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def utm_to_pixel(utm_e, utm_n, map_data):
    """
    Convert UTM coordinates to pixel grid coordinates.

    Parameters
    ----------
    utm_e : float or array_like
        UTM Easting coordinate(s) in meters
    utm_n : float or array_like
        UTM Northing coordinate(s) in meters
    map_data : dict
        Map data dictionary from load_slc_map() containing:
        - 'UTM_lat': Array of UTM northing values for each row
        - 'UTM_long': Array of UTM easting values for each column

    Returns
    -------
    tuple of (int or ndarray, int or ndarray)
        (col, row) pixel coordinates
        Note: Returns (col, row) to match [x, y] convention

    Examples
    --------
    >>> utm_e, utm_n = 425000.0, 4512000.0
    >>> col, row = utm_to_pixel(utm_e, utm_n, map_data)
    """
    utm_e = np.atleast_1d(utm_e)
    utm_n = np.atleast_1d(utm_n)

    # Find closest pixel indices
    col_idx = np.argmin(np.abs(map_data['UTM_long'][:, None] - utm_e), axis=0)
    row_idx = np.argmin(np.abs(map_data['UTM_lat'][:, None] - utm_n), axis=0)

    # Return as scalar if input was scalar
    if col_idx.size == 1:
        return int(col_idx[0]), int(row_idx[0])
    else:
        return col_idx, row_idx


def pixel_to_utm(col, row, map_data):
    """
    Convert pixel grid coordinates to UTM coordinates.

    Parameters
    ----------
    col : int or array_like
        Column index (x coordinate)
    row : int or array_like
        Row index (y coordinate)
    map_data : dict
        Map data dictionary from load_slc_map()

    Returns
    -------
    tuple of (float or ndarray, float or ndarray)
        (utm_e, utm_n) UTM coordinates in meters

    Examples
    --------
    >>> col, row = 100, 200
    >>> utm_e, utm_n = pixel_to_utm(col, row, map_data)
    """
    col = np.atleast_1d(col)
    row = np.atleast_1d(row)

    # Clip to valid range
    col = np.clip(col, 0, len(map_data['UTM_long']) - 1)
    row = np.clip(row, 0, len(map_data['UTM_lat']) - 1)

    utm_e = map_data['UTM_long'][col]
    utm_n = map_data['UTM_lat'][row]

    # Return as scalar if input was scalar
    if utm_e.size == 1:
        return float(utm_e[0]), float(utm_n[0])
    else:
        return utm_e, utm_n


def pixel_array_to_utm(pixel_coords, map_data):
    """
    Convert array of pixel coordinates to UTM coordinates.

    Parameters
    ----------
    pixel_coords : ndarray
        Nx2 array of [col, row] pixel coordinates
    map_data : dict
        Map data dictionary from load_slc_map()

    Returns
    -------
    ndarray
        Nx2 array of [utm_e, utm_n] coordinates

    Examples
    --------
    >>> pixel_coords = np.array([[100, 200], [150, 250]])
    >>> utm_coords = pixel_array_to_utm(pixel_coords, map_data)
    """
    pixel_coords = np.atleast_2d(pixel_coords)
    utm_coords = np.zeros_like(pixel_coords, dtype=float)

    for i in range(len(pixel_coords)):
        utm_e, utm_n = pixel_to_utm(pixel_coords[i, 0], pixel_coords[i, 1], map_data)
        utm_coords[i] = [utm_e, utm_n]

    return utm_coords


def utm_array_to_pixel(utm_coords, map_data):
    """
    Convert array of UTM coordinates to pixel coordinates.

    Parameters
    ----------
    utm_coords : ndarray
        Nx2 array of [utm_e, utm_n] coordinates
    map_data : dict
        Map data dictionary from load_slc_map()

    Returns
    -------
    ndarray
        Nx2 array of [col, row] pixel coordinates

    Examples
    --------
    >>> utm_coords = np.array([[425000.0, 4512000.0], [425100.0, 4512100.0]])
    >>> pixel_coords = utm_array_to_pixel(utm_coords, map_data)
    """
    utm_coords = np.atleast_2d(utm_coords)
    pixel_coords = np.zeros_like(utm_coords, dtype=int)

    for i in range(len(utm_coords)):
        col, row = utm_to_pixel(utm_coords[i, 0], utm_coords[i, 1], map_data)
        pixel_coords[i] = [col, row]

    return pixel_coords


def latlon_to_utm(latitude, longitude, utm_zone=12, northern=True):
    """
    Convert latitude/longitude to UTM coordinates.

    Parameters
    ----------
    latitude : float or array_like
        Latitude in decimal degrees
    longitude : float or array_like
        Longitude in decimal degrees (negative for West)
    utm_zone : int, optional
        UTM zone number (default: 12 for Salt Lake City)
    northern : bool, optional
        True if Northern hemisphere (default: True)

    Returns
    -------
    tuple of (float or ndarray, float or ndarray)
        (utm_e, utm_n) UTM coordinates in meters

    Examples
    --------
    >>> lat, lon = 40.76414, -111.84759  # Bookstore location
    >>> utm_e, utm_n = latlon_to_utm(lat, lon, utm_zone=12)
    >>> isinstance(utm_e, float) and isinstance(utm_n, float)
    True

    Notes
    -----
    Salt Lake City is in UTM Zone 12N.
    The default settings are configured for this region.
    """
    latitude = np.atleast_1d(latitude)
    longitude = np.atleast_1d(longitude)

    # Create transformer from WGS84 (EPSG:4326) to UTM
    hemisphere = 'north' if northern else 'south'
    utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    # Transform (note: pyproj expects lon, lat order)
    utm_e, utm_n = transformer.transform(longitude, latitude)

    # Return as scalar if input was scalar
    if np.asarray(utm_e).size == 1:
        return float(utm_e), float(utm_n)
    else:
        return np.asarray(utm_e), np.asarray(utm_n)


def latlon_to_pixel(latitude, longitude, map_data, utm_zone=12, northern=True):
    """
    Convert latitude/longitude to pixel coordinates.

    This is a convenience function that combines latlon_to_utm and utm_to_pixel.

    Parameters
    ----------
    latitude : float or array_like
        Latitude in decimal degrees
    longitude : float or array_like
        Longitude in decimal degrees (negative for West)
    map_data : dict
        Map data dictionary from load_slc_map()
    utm_zone : int, optional
        UTM zone number (default: 12 for Salt Lake City)
    northern : bool, optional
        True if Northern hemisphere (default: True)

    Returns
    -------
    tuple of (int or ndarray, int or ndarray)
        (col, row) pixel coordinates

    Examples
    --------
    >>> lat, lon = 40.76414, -111.84759  # Bookstore location
    >>> col, row = latlon_to_pixel(lat, lon, map_data, utm_zone=12)
    >>> isinstance(col, (int, np.integer)) and isinstance(row, (int, np.integer))
    True

    Notes
    -----
    This function is useful for directly converting sensor locations
    specified in lat/lon to pixel coordinates for the localization algorithm.
    """
    # Convert lat/lon to UTM
    utm_e, utm_n = latlon_to_utm(latitude, longitude, utm_zone, northern)

    # Convert UTM to pixel
    col, row = utm_to_pixel(utm_e, utm_n, map_data)

    return col, row


def latlon_array_to_pixel(latlon_coords, map_data, utm_zone=12, northern=True):
    """
    Convert array of lat/lon coordinates to pixel coordinates.

    Parameters
    ----------
    latlon_coords : ndarray
        Nx2 array of [longitude, latitude] coordinates in decimal degrees
    map_data : dict
        Map data dictionary from load_slc_map()
    utm_zone : int, optional
        UTM zone number (default: 12 for Salt Lake City)
    northern : bool, optional
        True if Northern hemisphere (default: True)

    Returns
    -------
    ndarray
        Nx2 array of [col, row] pixel coordinates

    Examples
    --------
    >>> latlon_coords = np.array([[-111.84759, 40.76414], [-111.83814, 40.76770]])
    >>> pixel_coords = latlon_array_to_pixel(latlon_coords, map_data)
    >>> pixel_coords.shape
    (2, 2)

    Notes
    -----
    Input format is [longitude, latitude] to match the [x, y] convention.
    """
    latlon_coords = np.atleast_2d(latlon_coords)
    pixel_coords = np.zeros((len(latlon_coords), 2), dtype=int)

    for i in range(len(latlon_coords)):
        lon, lat = latlon_coords[i]
        col, row = latlon_to_pixel(lat, lon, map_data, utm_zone, northern)
        pixel_coords[i] = [col, row]

    return pixel_coords
